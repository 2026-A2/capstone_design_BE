from django.urls import path
from . import views

urlpatterns = [
    # 영상 및 질문 업로드
    path('analyze/<int:interview_id>/<int:order>/', views.process_video_analysis, name='process_video_analysis'),
    
    # 특정 면접의 통합 리포트 조회
    path('report/individual/<int:interview_id>/', views.get_individual_report, name='get_individual_report'),
    
    # 전체 면접 점수 추이 조회
    path('report/cumulative/', views.get_cumulative_report, name='get_cumulative_report'),
]