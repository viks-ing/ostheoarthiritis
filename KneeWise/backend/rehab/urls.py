"""
rehab/urls.py - URL Routing for Rehabilitation Session REST Endpoints
"""

from django.urls import path
from .views import (
    SessionListCreateView,
    SessionDetailView,
    SessionCompleteView,
    SetCreateView,
    RehabProcessFrameView,
)

app_name = "rehab"

urlpatterns = [
    path("sessions/", SessionListCreateView.as_view(), name="session_list_create"),
    path("sessions/<uuid:pk>/", SessionDetailView.as_view(), name="session_detail"),
    path("sessions/<uuid:pk>/complete/", SessionCompleteView.as_view(), name="session_complete"),
    path("sessions/<uuid:pk>/sets/", SetCreateView.as_view(), name="set_create"),
    path("process-frame/", RehabProcessFrameView.as_view(), name="process_frame"),
]
