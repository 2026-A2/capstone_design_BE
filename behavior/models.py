from django.db import models
from interview.models import Interview, InterviewQuestion

#초 단위 상세 분석 데이터 DB (1:N)
class BehaviorDetail(models.Model):
    # 질문별 구분은 하되, 리포트는 하나로 합칠 수 있게 ForeignKey 유지
    question = models.ForeignKey(InterviewQuestion, on_delete=models.CASCADE, related_name='behavior_details')
    timestamp = models.FloatField() #영상 내 시간(초)

    # 1. 시선 및 고개 (수직/수평 세분화)
    gaze_x = models.FloatField(help_text = "눈동자 수평 위치")
    gaze_y = models.FloatField(help_text = "눈동자 수직 위치 (내리깔기 감지)")
    head_yaw = models.FloatField(help_text = "고개 좌우 회전")
    head_pitch = models.FloatField(help_text = "고개 상하 각도 (끄덕임/시선하락)")
    head_roll = models.FloatField(help_text = "고개 좌우 기울기 (갸우뚱)")
    
    # 2. 어깨 및 몸통 (흔들림 수치화)
    shoulder_tilt = models.FloatField(help_text = "실시간 어깨 기울기 각도")
    shoulder_width = models.FloatField(help_text = "화면상 어깨 너비 (앞뒤 흔들림 감지용)")
    center_x = models.FloatField(help_text = "어깨 중앙점 X좌표 (좌우 흔들림 감지용)")

    # 3. 상태 판정 (로직에서 판단한 결과값)
    is_smiling = models.BooleanField(default=False)
    is_blink = models.BooleanField(default=False, help_text="해당 프레임에서 눈을 감았는지")
    # "왼쪽 응시율 00%"를 계산하기 위해 방향 상태를 저장
    gaze_direction = models.CharField(max_length=10, default='center') # left, right, center, up, down
    is_swaying = models.BooleanField(default=False)

#최종 분석 리포트용 / 질문(Question)이 아닌 면접(Interview)과 1:1 연결
class BehaviorReport(models.Model):
    interview = models.OneToOneField(Interview, on_delete=models.CASCADE, related_name='behavior_report')
    
    # 시선 리포트 - 왼쪽/오른쪽 응시율 , 눈깜빡임 
    focus_rate = models.FloatField(help_text="전체 정면 응시율")
    left_gaze_rate = models.FloatField(help_text="왼쪽 응시 비중")
    right_gaze_rate = models.FloatField(help_text="오른쪽 응시 비중")
    blinks_per_min = models.FloatField()
    
    # 자세 리포트 - 고개 , 몸통 좌우/앞뒤 흔들림 횟수와 안정성
    nod_count = models.IntegerField(help_text="고개 끄덕임 총 횟수")
    shoulder_stability = models.FloatField(help_text="어깨 수평 유지율(%)")
    lr_sway_count = models.IntegerField(help_text="좌우 흔들림 감지 횟수")
    fb_sway_count = models.IntegerField(help_text="앞뒤 흔들림 감지 횟수")
    is_swaying = models.BooleanField(default=False, help_text="해당 프레임에서 몸의 흔들림이 감지되었는지")
    
    # 표정 리포트 
    total_smile_rate = models.FloatField()
    start_smile_status = models.BooleanField()
    end_smile_status = models.BooleanField()
    
    #점수제 - 유지할 지 고민중 
    overall_score = models.IntegerField()
