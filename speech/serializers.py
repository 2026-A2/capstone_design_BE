from rest_framework import serializers


class AudioUploadSerializer(serializers.Serializer):
    audio = serializers.FileField(help_text="음성 또는 영상 파일 (wav, mp3, m4a, mp4, mov, avi, mkv, webm)")


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


class SpeechAnalysisResponseSerializer(serializers.Serializer):
    transcript = serializers.CharField()
    syllable_count = serializers.IntegerField()
    duration_sec = serializers.FloatField()
    spm = serializers.FloatField(help_text="분당 음절 수")
    pace = serializers.CharField(help_text="말 속도 (느림/보통/빠름/매우 빠름)")
    silences = SilenceSerializer(many=True)
    volume = VolumeSerializer()
    filler = FillerSerializer()
