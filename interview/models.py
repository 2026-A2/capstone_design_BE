# interview/models.py
from django.db import models

class Resume(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Interview(models.Model):

    TYPE_CHOICES = [
        ('RESUME', '자소서 기반'),
        ('JOB', '직무 기반')
    ]

    STATUS_CHOICES = [
        ('CREATED', '생성됨'),
        ('CALIBRATED', '캘리브레이션 완료'),
        ('PROGRESS', '진행중'),
        ('ANALYZING', '분석중'),
        ('COMPLETED', '완료'),
        ('FAILED', '실패')
    ]

    interview_type = models.CharField(
        max_length=10,
        choices=TYPE_CHOICES
    )

    question_count = models.IntegerField()

    resume_content = models.TextField(
        null=True,
        blank=True
    )

    job_category = models.CharField(
        max_length=100,
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='CREATED'
    )

    calibration_config = models.JSONField(
        null=True,
        blank=True,
        help_text="초기 캘리브레이션 기준값"
    )

    def __str__(self):
        return f"[{self.id}] {self.interview_type} - {self.status}"



class InterviewQuestion(models.Model):
    STATUS_CHOICES = [('PENDING', '대기'), ('ANALYZING', '분석중'), ('COMPLETED', '완료'), ('FAILED', '실패')]
    interview = models.ForeignKey(Interview, on_delete=models.CASCADE, related_name='questions')
    question_text = models.CharField(max_length=500)
    order = models.IntegerField()
    video_path = models.FileField(upload_to='videos/%Y/%m/%d/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    video_duration = models.FloatField(default=0.0)  # 분당 계산을 위해 영상 길이(초) 필수 저장

    # ================= [딱 필요한 6가지 핵심 지표 (행동)] =================
    gaze_front_ratio = models.FloatField(default=0.0)      # 1. 정면 응시율 (%)
    gaze_deviation_ratio = models.FloatField(default=0.0)  # 2. 시선 이탈률 (%)
    body_sway_count = models.IntegerField(default=0)       # 3. 몸 흔들림 총 횟수 (원천 데이터)
    shoulder_stability_ratio = models.FloatField(default=0.0)# 4. 어깨 안정도 (%)
    blink_count = models.IntegerField(default=0)           # 5. 눈 깜빡임 총 횟수 (원천 데이터)
    nod_count = models.IntegerField(default=0)             # 6. 고개 끄덕임 총 횟수 (원천 데이터)
    smile_ratio = models.FloatField(default=0.0)           # 7. 미소율 (%)

    class Meta:
        ordering = ['order']
        unique_together = ('interview', 'order')