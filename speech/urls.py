from django.urls import path
from . import views

urlpatterns = [
    path("interview/<int:interview_id>/analyze/", views.analyze_interview, name="analyze_interview"),
]
