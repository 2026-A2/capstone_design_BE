import os

import ffmpeg
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from speech.transcriber import analyze
from speech.volume import analyze_volume
from speech.filler import detect_fillers
from speech.aggregator import aggregate_results
from speech.models import SpeechAnalysis, SpeechReport, SpeechSilence, SpeechFiller, SpeechInterviewReport
from speech.serializers import SpeechSessionResponseSerializer

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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


def _analyze_single(question) -> dict:
    file_path = question.video_path.path
    file_name = os.path.basename(file_path)
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

        speech_analysis = SpeechAnalysis.objects.create(question=question)

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
            "order": question.order,
            "file_name": file_name,
            **result,
        }

    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.unlink(extracted_path)


@extend_schema(
    summary="면접 음성 분석",
    description=(
        "사용자가 '나의 면접 결과 보기'를 클릭하면 호출합니다. "
        "InterviewQuestion에 저장된 영상을 순서대로 분석하고 항목별 집계 결과를 반환합니다."
    ),
    parameters=[
        OpenApiParameter(
            name="interview_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            description="분석할 Interview ID",
        )
    ],
    responses={200: SpeechSessionResponseSerializer},
    tags=['Speech Analysis'],
)
@api_view(["GET"])
def analyze_interview(request, interview_id: int):
    from interview.models import Interview

    try:
        interview = Interview.objects.get(id=interview_id)
    except Interview.DoesNotExist:
        return Response({"error": "존재하지 않는 면접입니다."}, status=status.HTTP_404_NOT_FOUND)

    if hasattr(interview, "speech_report"):
        return Response({"error": "이미 분석이 완료된 면접입니다."}, status=status.HTTP_400_BAD_REQUEST)

    questions = interview.questions.all()
    if not questions.exists():
        return Response({"error": "등록된 질문이 없습니다."}, status=status.HTTP_400_BAD_REQUEST)

    individual_results = []
    try:
        for question in questions:
            result = _analyze_single(question)
            individual_results.append(result)

        summary = aggregate_results(individual_results)

        SpeechInterviewReport.objects.create(
            interview=interview,
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

        individual_filtered = [
            {
                "order": r["order"],
                "file_name": r["file_name"],
                "transcript": r["transcript"],
                "silences": r["silences"],
                "volume_timeline": r["volume"]["volume_timeline"],
                "trailing_off": r["volume"]["trailing_off"],
                "fillers": r["filler"]["fillers"],
            }
            for r in individual_results
        ]

        return Response({
            "summary": summary,
            "interview_id": interview.id,
            "video_count": questions.count(),
            "individual": individual_filtered,
        })

    except Exception as e:
        return Response(
            {"error": f"분석 중 오류가 발생했습니다: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
