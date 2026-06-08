# behavior/models.py
from django.db import models
from interview.models import InterviewQuestion

class BehaviorDetail(models.Model):
    question = models.ForeignKey(InterviewQuestion, on_delete=models.CASCADE, related_name='details')
    timestamp = models.FloatField()
    
    # 실시간 프레임별 핵심 상태 변수
    gaze_direction = models.CharField(max_length=10, default='center') # center, left, right 등
    is_swaying = models.BooleanField(default=False)
    is_blink = models.BooleanField(default=False)
    is_nodding = models.BooleanField(default=False)
    is_smiling = models.BooleanField(default=False)
    shoulder_stable_frame = models.BooleanField(default=True) # 해당 프레임 어깨 안정 여부
