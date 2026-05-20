from django.db import models


class SpeechSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    video_count = models.IntegerField()

    class Meta:
        db_table = "speech_session"


class SpeechVideoFile(models.Model):
    session = models.ForeignKey(SpeechSession, on_delete=models.CASCADE, related_name="video_files")
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512)
    order = models.IntegerField()

    class Meta:
        db_table = "speech_video_file"
        ordering = ["order"]


class SpeechAnalysis(models.Model):
    session = models.ForeignKey(SpeechSession, on_delete=models.CASCADE, related_name="analyses", null=True, blank=True)
    file_name = models.CharField(max_length=255)
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default="completed")

    class Meta:
        db_table = "speech_analysis"


class SpeechReport(models.Model):
    analysis = models.OneToOneField(SpeechAnalysis, on_delete=models.CASCADE, related_name="report")
    transcript = models.TextField()
    syllable_count = models.IntegerField()
    duration_sec = models.FloatField()
    spm = models.FloatField()
    pace = models.CharField(max_length=20)
    avg_db = models.FloatField()
    max_db = models.FloatField()
    min_db = models.FloatField()
    std_db = models.FloatField()
    volume_level = models.CharField(max_length=20)
    filler_count = models.IntegerField()
    frequent_fillers = models.JSONField(default=list)
    volume_timeline = models.JSONField(default=list)

    class Meta:
        db_table = "speech_report"


class SpeechSilence(models.Model):
    analysis = models.ForeignKey(SpeechAnalysis, on_delete=models.CASCADE, related_name="silences")
    start = models.FloatField()
    end = models.FloatField()
    duration = models.FloatField()

    class Meta:
        db_table = "speech_silence"


class SpeechFiller(models.Model):
    analysis = models.ForeignKey(SpeechAnalysis, on_delete=models.CASCADE, related_name="fillers")
    start = models.FloatField()
    end = models.FloatField()
    duration = models.FloatField()
    filler_type = models.CharField(max_length=10)

    class Meta:
        db_table = "speech_filler"


class SpeechSessionReport(models.Model):
    session = models.OneToOneField(SpeechSession, on_delete=models.CASCADE, related_name="report")
    avg_spm = models.FloatField()
    pace = models.CharField(max_length=20)
    avg_db = models.FloatField()
    max_db = models.FloatField()
    min_db = models.FloatField()
    avg_std_db = models.FloatField()
    volume_level = models.CharField(max_length=20)
    total_filler_count = models.IntegerField()
    frequent_fillers = models.JSONField(default=list)
    total_silence_count = models.IntegerField()
    avg_silence_duration = models.FloatField()

    class Meta:
        db_table = "speech_session_report"
