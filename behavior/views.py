from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

class BehaviorAnalyzeView(APIView):
    @extend_schema(
        summary="면접 행동 분석",
        description="동영상을 업로드하여 눈 깜빡임, 미소 등을 분석합니다.",
        responses={200: str}
    )
    def post(self, request):
        return Response({"message": "분석 완료!"})