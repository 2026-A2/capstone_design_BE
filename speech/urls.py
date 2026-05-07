from django.urls import path
from . import views

urlpatterns = [
    path("analyze/", views.analyze_audio, name="analyze_audio"),
]
