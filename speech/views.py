import os
import tempfile

import ffmpeg
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema

from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers
from speech.aggregator import aggregate_results
from speech.models import SpeechAnalysis, SpeechReport, SpeechSilence, SpeechFiller, SpeechSession, SpeechSessionReport
from speech.serializers import VideoUploadSerializer, SpeechSessionResponseSerializer

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio(video_path: str) -> str:
    wav_path = video_path + "_audio.wav"
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


def _analyze_single(audio_file, order: int, session: SpeechSession) -> dict:
    ext = os.path.splitext(audio_file.name)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    extracted_path = None
    try:
        if ext in VIDEO_EXTENSIONS:
            extracted_path = extract_audio(tmp_path)
            analyze_path = extracted_path
        else:
            analyze_path = tmp_path

        result = analyze(analyze_path)
        segments = result.pop("segments", None)

        volume = analyze_volume(analyze_path, segments=segments)
        filler = detect_fillers(analyze_path)

        result["volume"] = volume
        result["filler"] = filler

        speech_analysis = SpeechAnalysis.objects.create(
            session=session,
            file_name=audio_file.name,
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
            "file_name": audio_file.name,
            **result,
        }

    finally:
        os.unlink(tmp_path)
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)


@extend_schema(
    summary="음성/영상 분석 (다중 파일)",
    description="영상 또는 음성 파일을 여러 개 업로드하면 각각 분석 후 전체 집계 결과를 반환합니다.",
    request={'multipart/form-data': VideoUploadSerializer},
    responses={200: SpeechSessionResponseSerializer},
    tags=['Speech Analysis']
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_videos(request):
    video_files = request.FILES.getlist("videos")

    if not video_files:
        return Response(
            {"error": "영상/음성 파일을 'videos' 키로 하나 이상 전송해 주세요."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    session = SpeechSession.objects.create(video_count=len(video_files))

    individual_results = []
    try:
        for order, audio_file in enumerate(video_files, start=1):
            result = _analyze_single(audio_file, order=order, session=session)
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
            "video_count": len(video_files),
            "individual": individual_results,
            "summary": summary,
        })

    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response(
            {"error": f"분석 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
