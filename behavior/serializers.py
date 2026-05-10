from rest_framework import serializers
from .models import BehaviorReport

# [요청용] 영상 업로드 시 사용
class VideoUploadSerializer(serializers.Serializer):
    # help_text는 DRF 기본 인자이므로 에러가 나지 않습니다.
    video_file = serializers.FileField(help_text="녹화된 답변 영상 파일 (mp4, webm)")
    question_text = serializers.CharField(
        help_text="질문 텍스트 (예: 지원동기에 대해 말씀해주세요.)"
    )

# [응답용] 최종 리포트 데이터
class BehaviorReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = BehaviorReport
        fields = '__all__'
        # 여기서도 example 인자 대신 help_text를 활용하거나 
        # views.py의 @extend_schema(examples=[...])를 통해 예시를 보여주는 것이 안전합니다.
        extra_kwargs = {
            'focus_rate': {'help_text': '정면 응시율 (예: 85.5)'},
            'overall_score': {'help_text': '종합 점수 (예: 88)'},
        }

# [응답용] 누적 추이 조회용
class CumulativeReportSerializer(serializers.Serializer):
    date = serializers.DateTimeField(source='interview.created_at', format="%Y-%m-%d")
    overall_score = serializers.IntegerField()
    focus_rate = serializers.FloatField()