# behavior/serializers.py
from rest_framework import serializers

# ===========================================================================
# [1-2번 API 요청/응답용] 캘리브레이션 영상 처리
# ===========================================================================
class CalibrationUploadSerializer(serializers.Serializer):
    video_file = serializers.FileField(
        help_text="약 5~10초 내외의 초기 환경 측정용 녹화 영상 파일 (.mp4 등)"
    )

class CalibrationResponseSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(help_text="면접 세션 ID")
    status = serializers.CharField(help_text="세션 상태 (CALIBRATED)")
    message = serializers.CharField(help_text="처리 완료 메시지")


# ===========================================================================
# [2번 API 요청/응답용] 질문별 답변 영상 업로드 및 분석
# ===========================================================================
class VideoUploadSerializer(serializers.Serializer):
    question_order = serializers.IntegerField(
        help_text="질문 순서 번호 (예: 1)", 
        min_value=1
    )
    question_text = serializers.CharField(
        max_length=500, 
        help_text="질문 텍스트 (예: 자기소개를 해주세요.)"
    )
    video_file = serializers.FileField(
        help_text="녹화된 영상 파일 (.mp4 등)"
    )

class VideoUploadResponseSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(help_text="면접 세션 ID")
    question_order = serializers.IntegerField(help_text="질문 순서 번호")
    status = serializers.CharField(help_text="분석 시작 상태 (ANALYZING)")
    message = serializers.CharField(help_text="처리 완료 메시지")


# ===========================================================================
# [5번 API 응답용] 누적 트렌드 데이터 구조
# ===========================================================================
class CumulativeTrendElementSerializer(serializers.Serializer):
    interview_id = serializers.IntegerField(help_text="면접 세션 ID")
    date = serializers.CharField(help_text="면접 실시 날짜 (YYYY-MM-DD)")
    interview_type = serializers.CharField(help_text="면접 유형 (RESUME / JOB)")
    gaze_front_ratio = serializers.FloatField(help_text="정면 응시율 (%)")
    gaze_deviation_ratio = serializers.FloatField(help_text="시선 이탈률 (%)")
    body_sway_per_min = serializers.FloatField(help_text="분당 몸 흔들림 횟수")
    shoulder_stability = serializers.FloatField(help_text="어깨 안정성 (%)")
    blink_per_min = serializers.FloatField(help_text="분당 눈 깜빡임 횟수")
    nod_per_min = serializers.FloatField(help_text="분당 고개 끄덕임 횟수")
    smile_ratio = serializers.FloatField(help_text="미소율 (%)")
    avg_spm = serializers.FloatField(required=False, help_text="평균 말하기 속도 (음절/분)")
    pace = serializers.CharField(required=False, help_text="말하기 속도 레벨 (빠름/보통/느림)")
    avg_db = serializers.FloatField(required=False, help_text="평균 음량 (dB)")
    volume_level = serializers.CharField(required=False, help_text="음량 레벨 (크다/보통/작다)")
    total_filler_count = serializers.IntegerField(required=False, help_text="전체 필러 횟수")
    total_silence_count = serializers.IntegerField(required=False, help_text="침묵 횟수")

# [5번 API 최종 응답 구조]
class CumulativeTrendsResponseSerializer(serializers.Serializer):
    total_interview_count = serializers.IntegerField(help_text="총 완료된 면접 회차 수")
    trends = CumulativeTrendElementSerializer(many=True, help_text="시계열 트렌드 리스트")