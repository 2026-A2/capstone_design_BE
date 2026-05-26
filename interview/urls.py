from django.urls import path
from . import views

urlpatterns = [
    # 세션 생성
    path('', views.create_interview, name='create_interview'),
    
    
    path('<int:interview_id>/finalize/', views.get_final_report, name='get_final_report'),
    
    # 면접 기록 목록
    path('list/', views.list_interviews, name='list_interviews'),
]