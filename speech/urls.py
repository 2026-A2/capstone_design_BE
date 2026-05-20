from django.urls import path
from . import views

urlpatterns = [
    path("analyze/", views.analyze_videos, name="analyze_videos"),
]
