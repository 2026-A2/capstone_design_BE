from django.urls import path
from . import views

urlpatterns = [
    path("interview/", views.upload_interview, name="upload_interview"),
    path("interview/<int:session_id>/analyze/", views.analyze_session, name="analyze_session"),
]
