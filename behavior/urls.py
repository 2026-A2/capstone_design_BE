from django.urls import path
from .views import BehaviorAnalyzeView

urlpatterns = [
    path('analyze/', BehaviorAnalyzeView.as_view(), name='analyze'),
]