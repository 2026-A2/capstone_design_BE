from django.db import models


class SpeechAnalysis(models.Model):
    question = models.OneToOneField(
        'interview.InterviewQuestion',
        on_delete=models.CASCADE,
        related_name='speech_analysis',
    )
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


class SpeechInterviewReport(models.Model):
    interview = models.OneToOneField(
        'interview.Interview',
        on_delete=models.CASCADE,
        related_name='speech_report',
    )
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
        db_table = "speech_interview_report"
