from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

#면접 기본 정보 테이블 
class Interview(models.Model):
    TYPE_CHOICES = [('RESUME', '자소서 기반'), ('JOB', '직무 기반')]
    STATUS_CHOICES = [
        ('pending', '대기'), 
        ('processing', '분석중'), 
        ('completed', '완료'), 
        ('failed', '실패')
    ]

    #면접 회차별 식별 제목
    title = models.CharField(max_length = 200)
    interview_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    #질문 개수 
    question_count = models.IntegerField(
        validators=[MinValueValidator(2), MaxValueValidator(5)], help_text="사용자가 설정한 질문 개수")
    
    #자소서 혹은 직무 - 둘 중 하나만 들어올 수 있음 (null=True)
    cover_letter = models.TextField(null = True, blank = True)
    job_group = models.CharField(max_length = 100, null = True, blank = True)
    #면접 실시 시간 
    created_at = models.DateTimeField(auto_now_add = True)

    # 전체 프로세스 상태 - 기본값 (pending)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return f"[{self.id}] {self.title}"


 #AI가 생성한 질문들을 저장 (질문당 영상 저장됨)
class InterviewQuestion(models.Model):
        
    STATUS_CHOICES = [
        ('pending', '대기'), 
        ('processing', '분석중'), 
        ('completed', '완료'), 
        ('failed', '실패')
    ]
    interview = models.ForeignKey(Interview, on_delete=models.CASCADE, related_name='questions')
    question_text = models.CharField(max_length=500, help_text="AI가 생성한 질문 내용")
    order = models.IntegerField(help_text="질문 순서 (1번 질문, 2번 질문...)")

    # 질문당 영상이 생성되므로 여기에 저장 (질문마다 영상이 생성되는 구조 반영)
    video_path = models.FileField(upload_to='videos/%Y/%m/%d/', null=True, blank=True)
    # 개별 질문 분석 상태 (질문 1은 완료, 질문 2는 분석 중... 일 수 있음)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.interview.title} - Q{self.order}: {self.question_text[:20]}..."

