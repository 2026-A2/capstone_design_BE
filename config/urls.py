"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from interview import views as interview_views
from drf_spectacular.views import (SpectacularAPIView, SpectacularJSONAPIView, SpectacularYAMLAPIView, SpectacularSwaggerView, SpectacularRedocView)
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('speech.urls')),

    path("json/", SpectacularJSONAPIView.as_view(), name="schema-json"),
    path("yaml/", SpectacularYAMLAPIView.as_view(), name="swagger-yaml"),
    path("swagger/", SpectacularSwaggerView.as_view(url_name="schema-json"), name="swagger-ui"),
    path("redoc/", SpectacularRedocView.as_view(url_name="schema-json"), name="redoc"),

    path('interviews/', include('interview.urls')),

   # 6. 저장된 자소서 목록 조회 (GET)
    path('resumes/', interview_views.get_resume_list, name='get_resume_list'),
    # 7. 자소서 상세 조회 (GET)
    path('resumes/<int:resume_id>/', interview_views.get_resume_detail, name='get_resume_detail'),
    # 8. 자소서 삭제 (DELETE)
    path('resumes/<int:resume_id>/delete/', interview_views.delete_resume, name='delete_resume'),

]

# 미디어 파일(영상 등) 접근 설정
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
