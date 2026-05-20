from rest_framework import serializers


# ── 1단계: 영상 업로드 ──────────────────────────────────────────────

class InterviewUploadSerializer(serializers.Serializer):
    videos = serializers.ListField(
        child=serializers.FileField(),
        help_text="면접 영상 파일 리스트 (mp4, mov, avi, mkv, webm, wav, mp3, m4a)",
    )


class InterviewUploadResponseSerializer(serializers.Serializer):
    session_id = serializers.IntegerField(help_text="생성된 세션 ID (분석 요청 시 사용)")
    video_count = serializers.IntegerField(help_text="업로드된 영상 수")


# ── 2단계: 분석 결과 ────────────────────────────────────────────────

class SilenceSerializer(serializers.Serializer):
    start = serializers.FloatField()
    end = serializers.FloatField()
    duration = serializers.FloatField()


class FillerDetailSerializer(serializers.Serializer):
    start = serializers.FloatField()
    end = serializers.FloatField()
    duration = serializers.FloatField()
    type = serializers.CharField()


class FillerSerializer(serializers.Serializer):
    filler_count = serializers.IntegerField()
    frequent_fillers = serializers.ListField(child=serializers.CharField())
    fillers = FillerDetailSerializer(many=True)


class VolumeSerializer(serializers.Serializer):
    avg_db = serializers.FloatField()
    max_db = serializers.FloatField()
    min_db = serializers.FloatField()
    std_db = serializers.FloatField()
    volume_level = serializers.CharField()
    trailing_off = serializers.ListField()
    volume_timeline = serializers.ListField()


class IndividualAnalysisSerializer(serializers.Serializer):
    order = serializers.IntegerField(help_text="영상 순서 (1부터 시작)")
    file_name = serializers.CharField()
    transcript = serializers.CharField(help_text="해당 영상 전사 텍스트")
    silences = SilenceSerializer(many=True, help_text="침묵 구간 상세 목록")
    volume_timeline = serializers.ListField(help_text="음량 타임라인")
    trailing_off = serializers.ListField(help_text="말끝 흐림 구간 목록")
    fillers = FillerDetailSerializer(many=True, help_text="습관어 상세 목록")


class SessionSummarySerializer(serializers.Serializer):
    avg_spm = serializers.FloatField(help_text="전체 가중 평균 말 속도 (음절/분)")
    pace = serializers.CharField(help_text="평균 말 속도 수준 (느림/보통/빠름/매우 빠름)")
    avg_db = serializers.FloatField(help_text="전체 평균 음량 (dB)")
    max_db = serializers.FloatField(help_text="전체 최대 음량 (dB)")
    min_db = serializers.FloatField(help_text="전체 최소 음량 (dB)")
    avg_std_db = serializers.FloatField(help_text="음량 표준편차 평균")
    volume_level = serializers.CharField(help_text="평균 음량 수준")
    total_filler_count = serializers.IntegerField(help_text="전체 습관어 횟수 합산")
    frequent_fillers = serializers.ListField(child=serializers.CharField(), help_text="빈도 순 습관어 목록")
    total_silence_count = serializers.IntegerField(help_text="전체 침묵 구간 수")
    avg_silence_duration = serializers.FloatField(help_text="침묵 구간 평균 길이 (초)")


class SpeechSessionResponseSerializer(serializers.Serializer):
    summary = SessionSummarySerializer()
    session_id = serializers.IntegerField()
    video_count = serializers.IntegerField()
    individual = IndividualAnalysisSerializer(many=True)
