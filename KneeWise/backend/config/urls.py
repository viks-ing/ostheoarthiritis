"""
config/urls.py - Main URL Routing Configuration for KneeWise Backend
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    # Screening and Diagnostic API Endpoints
    path("api/screening/", include("screening.urls", namespace="screening")),
    # Rehabilitation API Endpoints
    path("api/rehab/", include("rehab.urls", namespace="rehab")),
    # Authentication & JWT Token Endpoints
    path("api/auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]

# SEC-01 Remediation: Direct unauthenticated public media serving is disabled to protect patient radiographs.
# All medical images must be accessed via the authenticated, permission-checked endpoint:
# GET /api/screening/xrays/<uuid>/

