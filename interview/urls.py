from django.urls import path
from . import views
from . import views as interview_views       
from behavior import views as behavior_views


urlpatterns = [
    # 1. 면접 세션 생성 (POST) & 4. 역대 면접 리포트 목록 조회 (GET)
    path('', views.interview_base_handler, name='interview_base_handler'),

    # 5. [누적 리포트 창] 회차별 변화 트렌드 데이터 조회 (GET)
    path('trends/', behavior_views.get_interview_trends, name='get_interview_trends'),

    # 2. 질문별 답변 영상 업로드 및 분석 요청 (POST)
    path('<int:interview_id>/questions/', behavior_views.process_video_analysis, name='process_video_analysis'),
    
    # 3. 방금 마친 면접 결과 상세 리포트 조회 (GET)
    path('<int:interview_id>/report/', interview_views.get_final_report , name='get_individual_report'),
]