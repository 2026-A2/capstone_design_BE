from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from speech.analyzer import analyze_question
from speech.aggregator import aggregate_results
from speech.models import SpeechInterviewReport
from speech.serializers import SpeechSessionResponseSerializer


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
            result = analyze_question(question)
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
