from rest_framework import serializers
from .models import Interview
from behavior.models import BehaviorReport

# [요청용] 면접 생성
class InterviewCreateSerializer(serializers.ModelSerializer):
    question_count = serializers.IntegerField(
        min_value=2, 
        max_value=5, 
        default=3,
        help_text="2~5개 사이 입력"
    )

    class Meta:
        model = Interview
        fields = ['interview_type', 'question_count', 'cover_letter', 'job_group']
        extra_kwargs = {
            'interview_type': {'help_text': 'RESUME 또는 JOB', 'default': 'RESUME'},
            'cover_letter': {'default': '안녕하세요. 신입 개발자 지원자입니다...'},
            'job_group': {'default': 'IT/개발'}
        }

# [응답용] 기본 정보
class InterviewResponseSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(source='id', default=1)
    interview_type = serializers.CharField(default='RESUME')
    question_count = serializers.IntegerField(default=3)
    status = serializers.CharField(default='pending')

# [응답용] 12개 분석 지표 (현실적인 수치 반영)
class BehaviorReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = BehaviorReport
        fields = [
            'focus_rate', 'left_gaze_rate', 'right_gaze_rate', 'blinks_per_min',
            'nod_count', 'shoulder_stability', 'lr_sway_count', 'fb_sway_count',
            'total_smile_rate', 'start_smile_status', 'end_smile_status', 'overall_score'
        ]
        # Swagger 예시값 강제 지정
        extra_kwargs = {
            'focus_rate': {'default': 85.5},
            'left_gaze_rate': {'default': 7.2},
            'right_gaze_rate': {'default': 7.3},
            'blinks_per_min': {'default': 12.0},
            'nod_count': {'default': 5},
            'shoulder_stability': {'default': 92.0},
            'lr_sway_count': {'default': 2},
            'fb_sway_count': {'default': 1},
            'total_smile_rate': {'default': 45.0},
            'start_smile_status': {'default': True},
            'end_smile_status': {'default': True},
            'overall_score': {'default': 88}
        }