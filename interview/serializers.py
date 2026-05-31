# interview/serializers.py
from rest_framework import serializers
from .models import Interview

# ===========================================================================
# [1번 API 요청용] 면접 세션 생성
# ===========================================================================

class InterviewCreateSerializer(serializers.Serializer):

    interview_type = serializers.ChoiceField(
        choices=['RESUME', 'JOB'],
        help_text="면접 유형 (RESUME 또는 JOB)"
    )

    question_count = serializers.IntegerField(
        min_value=2,
        max_value=5,
        default=5,
        help_text="사용자가 설정한 질문 개수 (2~5개)"
    )

    resume_content = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="자소서 내용"
    )

    job_category = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="type이 JOB일 때 필수 입력"
    )

    video_file = serializers.FileField(
        help_text="초기 캘리브레이션 영상 파일"
    )

    def validate(self, data):

        interview_type = data.get(
            'interview_type'
        )

        if interview_type == 'RESUME':

            if not data.get(
                'resume_content'
            ):
                raise serializers.ValidationError(
                    "RESUME 면접은 resume_content가 필요합니다."
                )

        elif interview_type == 'JOB':

            if not data.get(
                'job_category'
            ):
                raise serializers.ValidationError(
                    "JOB 면접은 job_category가 필요합니다."
                )

        return data
# ===========================================================================
# [1번 API 응답용] 면접 세션 생성 완료 반환 데이터
# ===========================================================================

class InterviewResponseSerializer(serializers.ModelSerializer):
    interview_id = serializers.IntegerField(source='id', help_text="생성된 면접 세션의 고유 ID")
    interview_type = serializers.CharField(help_text="면접 유형 (RESUME 또는 JOB)")
    question_count = serializers.IntegerField(help_text="해당 면접에서 진행할 총 질문 개수")
    status = serializers.CharField(help_text="현재 면접 세션 상태 (CALIBRATED)") # 파일 처리가 동시에 끝나므로 초기 상태가 CALIBRATED가 됩니다!

    class Meta:
        model = Interview
        fields = ['interview_id', 'interview_type', 'question_count', 'status']


# ===========================================================================
# [4번 API 응답용] 전체 면접 리포트 요약 목록의 단일 아이템
# ===========================================================================
class InterviewListElementSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(help_text="면접 세션 ID")
    interview_type = serializers.CharField(help_text="면접 유형 (RESUME / JOB)")
    created_at = serializers.CharField(help_text="면접 생성 날짜 (YYYY-MM-DD)")
    question_count = serializers.IntegerField(help_text="총 질문 개수")
    status = serializers.CharField(help_text="현재 면접 세션 상태")


# ===========================================================================
# [3번 API 응답용] 방금 마친 면접 결과 상세 리포트 내 하위 구조들
# ===========================================================================
class ReportBehaviorSerializer(serializers.Serializer):
    gaze_front_ratio = serializers.FloatField(help_text="정면 응시율 (%)")
    gaze_deviation_ratio = serializers.FloatField(help_text="시선 이탈률 (%)")
    body_sway_per_min = serializers.FloatField(help_text="분당 몸 흔들림 횟수")
    shoulder_stability = serializers.FloatField(help_text="어깨 안정성 점수 (%)")
    blink_per_min = serializers.FloatField(help_text="분당 눈 깜빡임 횟수")
    nod_per_min = serializers.FloatField(help_text="분당 고개 끄덕임 횟수")
    smile_ratio = serializers.FloatField(help_text="미소율 (%)")

class ReportSpeechSerializer(serializers.Serializer):
    avg_spm = serializers.FloatField(help_text="평균 말하기 속도 (음절/분)")
    pace = serializers.CharField(help_text="말하기 속도 레벨 (빠름/보통/느림)")
    avg_db = serializers.FloatField(help_text="평균 음량 (dB)")
    volume_level = serializers.CharField(help_text="음량 레벨 (크다/보통/작다)")
    total_filler_count = serializers.IntegerField(help_text="전체 필러 횟수")
    frequent_fillers = serializers.ListField(child=serializers.CharField(), help_text="자주 사용한 필러 단어 목록")
    total_silence_count = serializers.IntegerField(help_text="침묵 횟수")
    avg_silence_duration = serializers.FloatField(help_text="평균 침묵 지속 시간 (초)")
    transcript = serializers.CharField(help_text="전체 답변 스크립트 (질문별 줄바꿈 구분)")

class FinalAnalysisResultSerializer(serializers.Serializer):
    behavior = ReportBehaviorSerializer()
    speech = ReportSpeechSerializer()

# [3번 API 최종 응답 구조]
class FinalReportSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(help_text="면접 세션 ID")
    interview_type = serializers.CharField(help_text="면접 유형 (RESUME / JOB)")
    status = serializers.CharField(help_text="면접 완료 상태 (COMPLETED)")
    question_count = serializers.IntegerField(help_text="질문 개수")
    total_video_duration = serializers.IntegerField(help_text="전체 영상 시간 (초 단위)")
    created_at = serializers.CharField(help_text="면접 일시 (ISO 포맷팅)")
    analysis_result = FinalAnalysisResultSerializer()

# [5번 API 최종 응답 구조]
class TrendItemSerializer(serializers.Serializer):

    interview_id = serializers.IntegerField()
    date = serializers.CharField()
    interview_type = serializers.CharField()
    gaze_front_ratio = serializers.FloatField()
    gaze_deviation_ratio = serializers.FloatField()
    body_sway_per_min = serializers.FloatField()
    shoulder_stability = serializers.FloatField()
    blink_per_min = serializers.FloatField()
    nod_per_min = serializers.FloatField()
    smile_ratio = serializers.FloatField()
    avg_spm = serializers.FloatField(required=False, help_text="평균 말하기 속도 (음절/분)")
    pace = serializers.CharField(required=False, help_text="말하기 속도 레벨 (빠름/보통/느림)")
    avg_db = serializers.FloatField(required=False, help_text="평균 음량 (dB)")
    volume_level = serializers.CharField(required=False, help_text="음량 레벨 (크다/보통/작다)")
    total_filler_count = serializers.IntegerField(required=False, help_text="전체 필러 횟수")
    total_silence_count = serializers.IntegerField(required=False, help_text="침묵 횟수")


class CumulativeTrendsResponseSerializer(serializers.Serializer):
    total_interview_count = serializers.IntegerField()
    trends = TrendItemSerializer(many=True)

class ResumeListSerializer(serializers.Serializer):
    resume_id = serializers.IntegerField()
    title = serializers.CharField()
    created_at = serializers.CharField()

class ResumeDetailSerializer(serializers.Serializer):
    resume_id = serializers.IntegerField()
    title = serializers.CharField()
    content = serializers.CharField()
    created_at = serializers.CharField()

class ResumeCreateSerializer(serializers.Serializer):

    title = serializers.CharField()

    content = serializers.CharField()


class ResumeUpdateSerializer(serializers.Serializer):

    title = serializers.CharField(
        required=False
    )

    content = serializers.CharField(
        required=False
    )


class DeleteResponseSerializer(serializers.Serializer):
    message = serializers.CharField()