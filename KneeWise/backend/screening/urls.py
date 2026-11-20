"""
screening/urls.py - Screening API URL Routing
"""

from django.urls import path
from .views import (
    XRayUploadView,
    ScreeningPredictView,
    ScreeningHistoryView,
    ScreeningDetailView,
    PatientListCreateView,
    ProtectedXRayMediaView,
)

app_name = "screening"

urlpatterns = [
    path("upload/", XRayUploadView.as_view(), name="xray-upload"),
    path("predict/", ScreeningPredictView.as_view(), name="screening-predict"),
    path("predict/<uuid:pk>/", ScreeningPredictView.as_view(), name="screening-predict-detail"),
    path("history/", ScreeningHistoryView.as_view(), name="screening-history"),
    path("history/<uuid:pk>/", ScreeningDetailView.as_view(), name="screening-detail"),
    path("xrays/<uuid:pk>/", ProtectedXRayMediaView.as_view(), name="xray-media"),
    path("<uuid:pk>/", ScreeningDetailView.as_view(), name="screening-detail-direct"),
    path("patients/", PatientListCreateView.as_view(), name="patient-list-create"),
]
