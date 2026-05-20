import os

import ffmpeg
from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers
from speech.aggregator import aggregate_results
from speech.models import (
    SpeechAnalysis, SpeechReport, SpeechSilence, SpeechFiller,
    SpeechSession, SpeechSessionReport, SpeechVideoFile,
)
from speech.serializers import (
    InterviewUploadSerializer, InterviewUploadResponseSerializer,
    SpeechSessionResponseSerializer,
)

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def _extract_audio(video_path: str) -> str:
    wav_path = video_path + "_audio.wav"
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


def _analyze_single(file_path: str, file_name: str, order: int, session: SpeechSession) -> dict:
    ext = os.path.splitext(file_name)[1].lower()
    extracted_path = None
    try:
        if ext in VIDEO_EXTENSIONS:
            extracted_path = _extract_audio(file_path)
            analyze_path = extracted_path
        else:
            analyze_path = file_path

        result = analyze(analyze_path)
        segments = result.pop("segments", None)

        volume = analyze_volume(analyze_path, segments=segments)
        filler = detect_fillers(analyze_path)

        result["volume"] = volume
        result["filler"] = filler

        speech_analysis = SpeechAnalysis.objects.create(
            session=session,
            file_name=file_name,
            order=order,
        )

        SpeechReport.objects.create(
            analysis=speech_analysis,
            transcript=result["transcript"],
            syllable_count=result["syllable_count"],
            duration_sec=result["duration_sec"],
            spm=result["spm"],
            pace=result["pace"],
            avg_db=volume["avg_db"],
            max_db=volume["max_db"],
            min_db=volume["min_db"],
            std_db=volume["std_db"],
            volume_level=volume["volume_level"],
            filler_count=filler["filler_count"],
            frequent_fillers=filler["frequent_fillers"],
            volume_timeline=volume["volume_timeline"],
        )

        SpeechSilence.objects.bulk_create([
            SpeechSilence(
                analysis=speech_analysis,
                start=s["start"],
                end=s["end"],
                duration=s["duration"],
            )
            for s in result["silences"]
        ])

        SpeechFiller.objects.bulk_create([
            SpeechFiller(
                analysis=speech_analysis,
                start=f["start"],
                end=f["end"],
                duration=f["duration"],
                filler_type=f["type"],
            )
            for f in filler["fillers"]
        ])

        return {
            "order": order,
            "file_name": file_name,
            **result,
        }

    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)


# ── 1단계: 면접 영상 업로드 ────────────────────────────────────────

@extend_schema(
    summary="1단계: 면접 영상 업로드",
    description=(
        "면접이 끝난 후 영상 파일 n개를 업로드합니다. "
        "파일은 서버에 저장되며, 분석은 시작되지 않습니다. "
        "응답으로 받은 session_id를 분석 요청 시 사용하세요."
    ),
    request={'multipart/form-data': InterviewUploadSerializer},
    responses={201: InterviewUploadResponseSerializer},
    tags=['Speech Analysis'],
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_interview(request):
    video_files = request.FILES.getlist("videos")

    if not video_files:
        return Response(
            {"error": "영상/음성 파일을 'videos' 키로 하나 이상 전송해 주세요."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    for f in video_files:
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return Response(
                {"error": f"지원하지 않는 파일 형식입니다: {f.name}. 지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    session = SpeechSession.objects.create(video_count=len(video_files))

    save_dir = os.path.join(settings.MEDIA_ROOT, "interviews", str(session.id))
    os.makedirs(save_dir, exist_ok=True)

    for order, video_file in enumerate(video_files, start=1):
        ext = os.path.splitext(video_file.name)[1].lower()
        save_path = os.path.join(save_dir, f"{order}{ext}")
        with open(save_path, "wb") as dest:
            for chunk in video_file.chunks():
                dest.write(chunk)

        SpeechVideoFile.objects.create(
            session=session,
            file_name=video_file.name,
            file_path=save_path,
            order=order,
        )

    return Response(
        {"session_id": session.id, "video_count": session.video_count},
        status=status.HTTP_201_CREATED,
    )


# ── 2단계: 분석 시작 ───────────────────────────────────────────────

@extend_schema(
    summary="2단계: 면접 분석 시작",
    description=(
        "사용자가 '나의 면접 결과 보기'를 클릭하면 호출합니다. "
        "업로드된 영상을 순서대로 분석하고, 항목별 집계 결과를 반환합니다."
    ),
    parameters=[
        OpenApiParameter(
            name="session_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            description="1단계 업로드에서 받은 session_id",
        )
    ],
    responses={200: SpeechSessionResponseSerializer},
    tags=['Speech Analysis'],
)
@api_view(["POST"])
def analyze_session(request, session_id: int):
    try:
        session = SpeechSession.objects.get(id=session_id)
    except SpeechSession.DoesNotExist:
        return Response({"error": "존재하지 않는 세션입니다."}, status=status.HTTP_404_NOT_FOUND)

    if hasattr(session, "report"):
        return Response({"error": "이미 분석이 완료된 세션입니다."}, status=status.HTTP_400_BAD_REQUEST)

    video_files = session.video_files.all()
    if not video_files.exists():
        return Response({"error": "업로드된 영상이 없습니다."}, status=status.HTTP_400_BAD_REQUEST)

    individual_results = []
    try:
        for video_file in video_files:
            result = _analyze_single(
                file_path=video_file.file_path,
                file_name=video_file.file_name,
                order=video_file.order,
                session=session,
            )
            individual_results.append(result)

        summary = aggregate_results(individual_results)

        SpeechSessionReport.objects.create(
            session=session,
            avg_spm=summary["avg_spm"],
            pace=summary["pace"],
            avg_db=summary["avg_db"],
            max_db=summary["max_db"],
            min_db=summary["min_db"],
            avg_std_db=summary["avg_std_db"],
            volume_level=summary["volume_level"],
            total_filler_count=summary["total_filler_count"],
            frequent_fillers=summary["frequent_fillers"],
            total_silence_count=summary["total_silence_count"],
            avg_silence_duration=summary["avg_silence_duration"],
        )

        return Response({
            "session_id": session.id,
            "video_count": session.video_count,
            "individual": individual_results,
            "summary": summary,
        })

    except Exception as e:
        return Response(
            {"error": f"분석 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
