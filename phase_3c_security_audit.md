# KneeWise — Phase 3C.0: Security Audit & Production Readiness Assessment

**Target System:** KneeWise AI Knee Osteoarthritis Screening & Rehabilitation Platform  
**Architecture:** React + TypeScript Frontend $\longrightarrow$ Django REST Framework (Port 8080) $\longrightarrow$ FastAPI ML Service (Port 8000) $\longrightarrow$ Frozen EfficientNet-B0 Checkpoint  
**Audit Type:** Read-Only Source Code & Configuration Security Review  
**Date:** 2026-11-20  
**Status:** Audit Complete — No Source Code Modified (Read-Only Compliance)  

---

## Executive Summary

A comprehensive, read-only security audit of the KneeWise platform was conducted across the Django REST Framework backend, the standalone FastAPI machine learning microservice, and associated infrastructure configurations.

While Phase 3B established clean architectural decoupling, deterministic inference, and robust unit-level error translation, this audit identified **critical security vulnerabilities** and **operational readiness gaps** that must be remediated prior to staging and production deployment. Most notably:
1. **Unprotected Direct Access to Radiographs (Critical):** Uploaded knee X-rays stored in `media/` are served without authentication when static file serving is active, exposing Protected Health Information (PHI).
2. **Zero Service-to-Service Authentication (Critical):** FastAPI accepts unauthenticated inference requests from any client on port 8000 with wildcard CORS (`allow_origins=["*"]`), bypassing Django authentication and audit logs.
3. **Missing Token Revocation & Excessive JWT Lifetime (High):** JWT token blacklisting is uninstalled, no logout endpoint exists, and access tokens persist for 2 hours.
4. **Lack of Rate Limiting & Unbounded Pagination (High):** Neither Django nor FastAPI enforces throttling, creating high vulnerability to credential brute-forcing, image upload flooding, and memory exhaustion via unpaginated screening history queries.
5. **No PHI Access Audit Logging (High / Regulatory):** Access to patient records and screening histories is not recorded, violating HIPAA / GDPR compliance standards for medical decision-support systems.
6. **Insecure Secrets & Source Control Exposure (Critical):** `SECRET_KEY` falls back to a hardcoded insecure string, `DEBUG` defaults to `True`, and `.gitignore` fails to exclude `.env`, `db.sqlite3`, or `media/`.

---

## Detailed Audit Findings by Security Category

### 1. Django Authentication

#### Current Implementation
- **Authentication Classes:** Configured in `KneeWise/backend/config/settings.py` (lines 110–122) with `rest_framework_simplejwt.authentication.JWTAuthentication` and `rest_framework.authentication.SessionAuthentication`.
- **Permission Enforcement:** `REST_FRAMEWORK['DEFAULT_PERMISSION_CLASSES']` enforces `rest_framework.permissions.IsAuthenticated`.
- **JWT Routing:** Configured in `KneeWise/backend/config/urls.py` (lines 18–20) via `TokenObtainPairView` (`/api/auth/token/`) and `TokenRefreshView` (`/api/auth/token/refresh/`).
- **Token Parameters:** `ACCESS_TOKEN_LIFETIME = timedelta(hours=2)`, `REFRESH_TOKEN_LIFETIME = timedelta(days=7)`, header prefix `Bearer`.

#### Relevant File Paths
- `KneeWise/backend/config/settings.py` (lines 33, 110–129)
- `KneeWise/backend/config/urls.py` (lines 9–12, 19–20)
- `KneeWise/backend/screening/views.py` (lines 47, 72, 175, 366, 398)

#### Security Weaknesses
1. **No Token Revocation / Blacklisting:** `rest_framework_simplejwt.token_blacklist` is absent from `INSTALLED_APPS`. There is no token blacklist table and no `/api/auth/logout/` endpoint. Compromised tokens cannot be revoked before their expiration window.
2. **Excessive Access Token Lifetime:** An access token lifetime of 2 hours is excessive for a clinical system handling sensitive radiographs and diagnostic inferences. Industry standard for medical apps is 15–30 minutes.
3. **No User Self-Service / Registration Control:** No user registration (`/api/auth/register/`) or password-reset endpoints exist. Clinicians must be provisioned manually via Django superuser CLI.
4. **Unnecessary `SessionAuthentication` in API Mode:** Retaining `SessionAuthentication` alongside JWT creates CSRF exposure for browser clients interacting with session cookies.

#### Severity
**High**

#### Recommended Remediation
- Add `rest_framework_simplejwt.token_blacklist` to `INSTALLED_APPS` and apply migrations.
- Implement a dedicated `/api/auth/logout/` endpoint accepting the refresh token and adding it to the blacklist table.
- Reduce `ACCESS_TOKEN_LIFETIME` to 15 minutes (`timedelta(minutes=15)`).
- Remove `SessionAuthentication` from `DEFAULT_AUTHENTICATION_CLASSES` for production API traffic.
- Implement structured clinician onboarding and password reset endpoints with strong complexity validators.

#### Existing Tests Covering Feature
- `test_unauthenticated_request_rejection` in `KneeWise/backend/screening/tests.py` (lines 121–134) verifies that unauthenticated requests return HTTP 401 on upload, predict, and history endpoints.

#### Missing Security Tests
- Test verifying that expired JWT access tokens are rejected with HTTP 401.
- Test verifying that tampered or forged JWT signatures are rejected.
- Test verifying that blacklisted refresh tokens cannot be used to obtain new access tokens.
- Test verifying that logged-out tokens are immediately invalid.

---

### 2. Authorization and Patient Isolation

#### Current Implementation
- **View-Level Permissions:** All views enforce `permission_classes = [IsAuthenticated]`.
- **Procedural Patient Isolation:**
  - `PatientListCreateView`: Non-staff clinicians are restricted to `Patient.objects.filter(assigned_clinician=request.user)`.
  - `XRayUploadView`: Verifies `patient.assigned_clinician == request.user`.
  - `ScreeningPredictView`: Validates that the requesting clinician is the assigned clinician of the patient linked to the X-ray.
  - `ScreeningHistoryView`: Filters predictions via `patient__assigned_clinician=request.user`.
  - `ScreeningDetailView`: Rejects access (`HTTP 403`) if `prediction.patient.assigned_clinician != request.user`.
- **Admin Isolation:** Staff and superusers (`request.user.is_staff or request.user.is_superuser`) can view and manage all patient records.
- **Admin Immutability:** `ScreeningPredictionAdmin` in `admin.py` sets all fields to `readonly_fields`.

#### Relevant File Paths
- `KneeWise/backend/screening/views.py` (lines 45–64, 119–141, 194–200, 231–236, 360–411)
- `KneeWise/backend/screening/models.py` (lines 29–64, 107–177)
- `KneeWise/backend/screening/serializers.py` (lines 16–37)
- `KneeWise/backend/screening/admin.py` (lines 9–37)

#### Security Weaknesses
1. **Unprotected Direct Static Media Access (Critical PHI Leakage):** In `config/urls.py` (lines 24–25), media files are served via `static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)`. Any unauthenticated actor who obtains or guesses the file path can directly download raw knee radiographs without logging in or proving clinician-patient authorization.
2. **Predictable Sequential Patient Directory Paths:** `models.xray_upload_path` formats paths as `xrays/patient_{instance.patient.id}/<uuid>.png`. Because `patient.id` is a sequential auto-increment integer, an attacker can enumerate patient folder IDs.
3. **Imperative Authorization vs Declarative Object Permissions:** Authorization checks are hardcoded procedurally inside view methods rather than utilizing DRF `BasePermission` classes (`has_object_permission`). This pattern is brittle and prone to omission during endpoint expansion.
4. **Permanent Hard Cascade Deletion of Diagnostic Data:** In `models.py`, `Patient` deletion cascades (`on_delete=models.CASCADE`) to `UploadedXRay` and `ScreeningPrediction`. If a clinician or admin deletes a patient, all historical radiographs and AI screening predictions are permanently erased from the database, violating medical record retention regulations.
5. **Writable `assigned_clinician` in `PatientSerializer`:** `assigned_clinician` is present in `fields` but absent from `read_only_fields`, creating a risk of clinician assignment tampering if the serializer is reused for updates.

#### Severity
**Critical**

#### Recommended Remediation
- Eliminate public static media serving of radiographs. Implement an authenticated streaming view (e.g. `GET /api/screening/xrays/<uuid>/`) protected by clinician-patient permission checks, utilizing `X-Accel-Redirect` (Nginx) or pre-signed URLs (S3).
- Replace sequential database IDs in file storage paths with UUIDs or cryptographic hashes (e.g., `xrays/patients/<patient_uuid>/<image_uuid>.png`).
- Encapsulate clinician authorization into reusable DRF permission classes (`IsAssignedClinicianOrAdmin`).
- Replace `on_delete=models.CASCADE` on `Patient` foreign keys with `models.PROTECT` or implement an audit-compliant soft-deletion architecture (`is_deleted`, `deleted_at`, `deleted_by`).
- Add `assigned_clinician` to `read_only_fields` in `PatientSerializer`.

#### Existing Tests Covering Feature
- `test_patient_ownership_validation` in `tests.py` (lines 197–207) verifies HTTP 403 when Clinician A attempts to upload an X-ray for Clinician B's patient.
- `test_unauthorized_prediction_detail_access` in `tests.py` (lines 381–414) verifies HTTP 403 when Clinician B attempts to view Clinician A's patient prediction detail.
- `test_screening_history_retrieval` in `tests.py` (lines 318–379) verifies history queries only return the requesting clinician's patients.

#### Missing Security Tests
- Test verifying that unauthenticated direct HTTP GET requests to `/media/xrays/...` return HTTP 401/403.
- Test verifying that Clinician B cannot access radiograph image files belonging to Clinician A's patient.
- Test verifying that attempting to delete a patient record with existing screening predictions raises a protected error or performs a soft delete.
- Test verifying that Clinician A cannot reassign a patient to Clinician B via the patient endpoint.

---

### 3. X-ray Upload Security

#### Current Implementation
- **File Validation Checks:**
  - File presence check (`if "file" not in request.FILES`).
  - Zero-byte file rejection (`if upload_file.size == 0`).
  - Maximum size enforcement: `MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024` (25 MB).
  - Extension whitelisting: `ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}`.
  - Image integrity verification: PIL `Image.open(upload_file).verify()`.
- **Filename Sanitization:** `xray_upload_path` uses `uuid.uuid4().hex` for storage naming, preventing client-controlled path traversal.
- **Client Filename Storage:** The original filename is saved separately as metadata in `UploadedXRay.original_filename`.

#### Relevant File Paths
- `KneeWise/backend/screening/models.py` (lines 16–26, 65–105)
- `KneeWise/backend/screening/views.py` (lines 41–43, 75–117, 237–260)
- `KneeWise/backend/config/settings.py` (lines 104–106, 139–141)

#### Security Weaknesses
1. **Client-Provided Content-Type Trusted:** Line 154 of `views.py` sets `mime_type=upload_file.content_type`. The client-supplied `Content-Type` header is not verified against actual file magic numbers.
2. **Shallow Header-Only Integrity Check:** PIL `img.verify()` validates that headers conform to an image format, but does not fully decode pixel data, inspect for trailing malicious payloads, or strip malicious EXIF metadata / polyglot scripts.
3. **High In-Memory File Buffering (DoS Risk):** `FILE_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024` (25 MB) in `settings.py` causes Django to buffer uploads up to 25 MB directly in server RAM (`InMemoryUploadedFile`). Concurrent uploads of large files can rapidly induce an Out-of-Memory (OOM) crash on hardware with 8 GB RAM.
4. **Orphaned File Storage on Deletion:** Django models do not automatically delete underlying files from `MEDIA_ROOT` when model instances are deleted, leading to storage leakage.

#### Severity
**High**

#### Recommended Remediation
- Implement true magic-byte MIME validation using `python-magic` or `filetype` to inspect the file header bytes before processing.
- Re-encode uploaded radiographs through PIL (opening and saving to a fresh buffer) to strip EXIF metadata, comments, and any embedded polyglot code.
- Reduce `FILE_UPLOAD_MAX_MEMORY_SIZE` to 2.5 MB (`2621440` bytes) so that large files are streamed to temporary disk files rather than consuming physical RAM.
- Connect Django `post_delete` signals on `UploadedXRay` to ensure physical image files are purged from disk/storage when records are deleted.

#### Existing Tests Covering Feature
- `test_successful_xray_upload` in `tests.py` (lines 136–159) verifies valid PNG upload and UUID storage naming.
- `test_invalid_file_extension_rejection` in `tests.py` (lines 161–171) verifies rejection of `.pdf` files.
- `test_empty_file_rejection` in `tests.py` (lines 173–183) verifies rejection of 0-byte files.
- `test_corrupted_image_rejection` in `tests.py` (lines 185–195) verifies rejection of corrupted image headers.

#### Missing Security Tests
- Test verifying rejection of MIME-spoofed files (e.g. PHP/HTML script with a `.png` extension and `image/png` header).
- Test verifying that malicious EXIF tags or embedded scripts are stripped upon ingestion.
- Test verifying strict rejection when file size exceeds the 25 MB boundary limit.
- Test verifying path traversal characters (`../../`) in original filenames are neutralized.

---

### 4. FastAPI Protection

#### Current Implementation
- **Microservice Location:** `KneeWise/ml/src/api.py`.
- **Lifespan Initialization:** Loads `KneeWiseInferenceEngine` once on startup, verifying model weights against the expected SHA-256 checksum (`d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b`).
- **Endpoints:** `GET /health` (diagnostic status) and `POST /predict` (image inference).
- **Communication Client:** `KneeWise/backend/screening/services.py` dispatches requests using `httpx.Client` with timeout handling (`KNEEWISE_FASTAPI_TIMEOUT = 15.0`).

#### Relevant File Paths
- `KneeWise/ml/src/api.py` (lines 30–73, 75–174)
- `KneeWise/backend/screening/services.py` (lines 41–122)
- `KneeWise/backend/config/settings.py` (lines 134–138)

#### Security Weaknesses
1. **Zero Service-to-Service Authentication (Critical):** FastAPI has no authentication layer. Any client on the network can directly query `POST /predict` or `GET /health` without presenting a token, API key, or certificate. This completely bypasses Django's authorization checks, rate limits, and audit logs.
2. **Insecure Wildcard CORS Configuration:** Lines 66–72 of `api.py` add `CORSMiddleware` with `allow_origins=["*"]` and `allow_credentials=True`. This allows arbitrary third-party web origins to issue requests to the inference microservice.
3. **Public Exposure of OpenAPI Documentation:** Default Swagger UI (`/docs`), ReDoc (`/redoc`), and schema (`/openapi.json`) are accessible without restriction.
4. **Event Loop Blocking During Inference (DoS Risk):** In `api.py` line 161, `engine.predict` runs CPU/GPU PyTorch inference synchronously inside an `async def` handler on the main asyncio event loop. During inference (~40–80 ms), the event loop is blocked, causing latency spikes and potential denial of service under concurrent traffic.
5. **Unrestricted Network Binding:** The runbook suggests binding to `0.0.0.0`, exposing the unauthenticated service to the local network or public internet.

#### Severity
**Critical**

#### Recommended Remediation
- Implement an API Key / Shared Secret header (`X-KneeWise-Service-Token`) validated on every FastAPI request via a FastAPI dependency. Store the secret in Django and FastAPI environment variables.
- Remove wildcard CORS (`allow_origins=["*"]`) from FastAPI. In production, FastAPI should only accept internal network traffic from the Django host.
- Disable Swagger and OpenAPI in production: `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`.
- Bind FastAPI strictly to `127.0.0.1` (or an isolated Docker container network).
- Offload synchronous PyTorch inference to a worker thread using `starlette.concurrency.run_in_threadpool` or define the route as a synchronous endpoint (`def predict_radiograph`).

#### Existing Tests Covering Feature
- `KneeWise/ml/src/test_api.py` (tests `GET /health` and `POST /predict` via Starlette `TestClient`).
- `test_fastapi_timeout_handling`, `test_fastapi_unavailable_handling`, and `test_invalid_prediction_response_handling` in `tests.py` verify Django error handling when FastAPI fails.

#### Missing Security Tests
- Test verifying that FastAPI rejects requests lacking the `X-KneeWise-Service-Token` header with HTTP 401/403.
- Test verifying that Swagger `/docs` and OpenAPI schema endpoints return HTTP 404 in production mode.
- Test verifying that concurrent inference requests do not block the `/health` diagnostic endpoint.

---

### 5. API Protection

#### Current Implementation
- Standard DRF parsers: `MultiPartParser`, `FormParser`, `JSONParser`.
- Standard Django CSRF middleware enabled in `MIDDLEWARE`.
- Range validation on threshold parameter: $0.05 \le \tau \le 0.95$.

#### Relevant File Paths
- `KneeWise/backend/config/settings.py` (lines 38–47, 110–122, 131–133)
- `KneeWise/backend/screening/views.py` (lines 45–411)

#### Security Weaknesses
1. **Completely Missing Rate Limiting / Throttling:** Neither DRF `DEFAULT_THROTTLE_CLASSES` nor FastAPI has rate limiting configured. The application is vulnerable to brute-force credential stuffing on `/api/auth/token/`, disk exhaustion on `/api/screening/upload/`, and compute exhaustion on `/api/screening/predict/`.
2. **Unbounded History Query / Lack of Pagination:** `ScreeningHistoryView` executes `qs.count()` and `serializer.data` on the complete unpaginated queryset. In a clinical facility with thousands of records, querying history will cause memory spikes, database contention, and API gateway timeouts.
3. **Permissive CORS in Django:** `CORS_ALLOW_ALL_ORIGINS = True` is hardcoded in `settings.py` (line 132). Any website can make cross-origin requests to the Django backend.
4. **Bypassing Serializers for Request Ingestion:** Views unpack request parameters manually (`request.data.get(...)`, `request.FILES[...]`) rather than using structured DRF input serializers, leading to inconsistent validation error formats.
5. **Missing HTTP Security Headers:** Essential production security headers (`SECURE_SSL_REDIRECT`, `SECURE_HSTS_SECONDS`, `SECURE_CONTENT_TYPE_NOSNIFF`, `X_FRAME_OPTIONS = "DENY"`, `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE`) are absent from `settings.py`.

#### Severity
**High**

#### Recommended Remediation
- Configure DRF throttling:
  ```python
  "DEFAULT_THROTTLE_CLASSES": [
      "rest_framework.throttling.AnonRateThrottle",
      "rest_framework.throttling.UserRateThrottle",
  ],
  "DEFAULT_THROTTLE_RATES": {
      "anon": "10/minute",
      "user": "120/minute",
      "uploads": "20/minute",
      "predict": "30/minute",
  }
  ```
- Enforce global pagination:
  ```python
  "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
  "PAGE_SIZE": 25,
  ```
- Disable `CORS_ALLOW_ALL_ORIGINS = True` and configure `CORS_ALLOWED_ORIGINS` strictly from environment variables.
- Refactor views to validate inputs using dedicated DRF serializers (`XRayUploadSerializer`, `ScreeningPredictRequestSerializer`).
- Configure security headers in `settings.py` for non-debug deployments.

#### Existing Tests Covering Feature
- `test_screening_history_retrieval` in `tests.py` verifies record counts and filtering, but does not test pagination or throttling.

#### Missing Security Tests
- Test verifying that rapid requests to `/api/auth/token/` trigger HTTP 429 (Too Many Requests).
- Test verifying that `/api/screening/history/` paginates results and respects page size limits.
- Test verifying that disallowed CORS origins receive no `Access-Control-Allow-Origin` response header.
- Test verifying security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, HSTS) are present in responses.

---

### 6. Secrets Management

#### Current Implementation
- Environment variables are read using `os.environ.get()` for `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `KNEEWISE_FASTAPI_URL`, `KNEEWISE_FASTAPI_TIMEOUT`, and `KNEEWISE_DEFAULT_THRESHOLD`.

#### Relevant File Paths
- `KneeWise/backend/config/settings.py` (lines 13–21, 134–138)
- `KneeWise/.gitignore` and `.gitignore`

#### Security Weaknesses
1. **Hardcoded Insecure Fallback Secret Key:** Line 15 of `settings.py` falls back to `"django-insecure-kneewise-medical-ai-screening-platform-key-dev-only"`. If the environment variable is omitted in deployment, the system silently operates with a compromised, publicly known key.
2. **Insecure Default `DEBUG = True`:** Line 18 of `settings.py` defaults to `True`. An unconfigured production instance will run with debug mode enabled, exposing detailed stack traces, internal paths, and SQL queries upon any error.
3. **Flawed `.gitignore` Rules (Risk of Leaking PHI and Credentials):**
   - The `.gitignore` files in the repository root and `KneeWise/` do **not** exclude `.env` files.
   - They do **not** exclude SQLite database files (`db.sqlite3`).
   - They do **not** exclude the `media/` directory (`media/xrays/`).
   - If Git is initialized, private keys, local user credentials, patient records, and patient radiographs would be committed to source control!

#### Severity
**Critical**

#### Recommended Remediation
- Require `DJANGO_SECRET_KEY` in production; raise `django.core.exceptions.ImproperlyConfigured` if unset when `DEBUG=False`.
- Default `DEBUG` to `False` (`os.environ.get("DJANGO_DEBUG", "False").lower() in ("true", "1", "t")`).
- Immediately update root and subproject `.gitignore` files to exclude:
  ```gitignore
  .env
  .env.*
  *.sqlite3
  KneeWise/backend/db.sqlite3
  KneeWise/backend/media/
  KneeWise/backend/staticfiles/
  *.pem
  *.key
  ```

#### Existing Tests Covering Feature
- None.

#### Missing Security Tests
- Test verifying that Django raises `ImproperlyConfigured` on startup if `DEBUG=False` and `DJANGO_SECRET_KEY` is empty or matches the insecure default.
- Automated repository check verifying that `.env`, `db.sqlite3`, and `media/` are ignored by Git.

---

### 7. Audit Logging

#### Current Implementation
- Standard Python `logging` used in `screening.views` and `screening.services`.
- Error events (network timeouts, FastAPI connection errors, corrupted images) are logged.
- Django Admin tracks changes via standard `django_admin_log`.

#### Relevant File Paths
- `KneeWise/backend/screening/views.py` (lines 11, 39, 207, 314)
- `KneeWise/backend/screening/services.py` (lines 7, 13, 75, 79, 83, 88, 92, 96, 103, 109, 114, 119)
- `KneeWise/backend/config/settings.py` (no `LOGGING` dict defined)

#### Security Weaknesses
1. **Zero Access Logging for Protected Health Information (HIPAA / GDPR Gap):** No audit log is emitted when a clinician views patient demographics, accesses screening histories, or retrieves prediction records. In healthcare, every access to patient data must produce an immutable audit trail.
2. **Authentication Events Not Logged:** Failed login attempts, brute-force patterns, password changes, and token refresh events are not tracked.
3. **Missing Centralized `LOGGING` Configuration in Django:** No `LOGGING` dictionary is defined in `settings.py`. Logs write to standard output with no rotation, structured formatting, or destination routing.
4. **Internal Path Exposure in Error Logs:** Line 207 logs internal file paths (`uploaded_xray.image.name`), and line 133 echoes requested patient identifiers in 404 responses.

#### Severity
**High**

#### Recommended Remediation
- Create a dedicated clinical audit logging service (`AuditLog` model or specialized audit logger) that records every PHI interaction:
  `{timestamp, clinician_id, patient_id, action, endpoint, client_ip, user_agent, outcome}`.
- Configure Django's `LOGGING` dictionary in `settings.py` with structured JSON formatters and rotating file handlers (`RotatingFileHandler`).
- Integrate signal listeners (`user_logged_in`, `user_login_failed`) to record all authentication lifecycle events.
- Sanitize error messages returned to clients to prevent internal path and identifier leakage.

#### Existing Tests Covering Feature
- None.

#### Missing Security Tests
- Test verifying that retrieving patient predictions generates a corresponding audit log entry.
- Test verifying that failed authentication attempts emit a security warning log.
- Test verifying that patient names and medical identifiers are masked in application debug logs.

---

### 8. Backup and Recovery

#### Current Implementation
- **Database:** Local SQLite (`KneeWise/backend/db.sqlite3`).
- **Media Storage:** Local filesystem (`KneeWise/backend/media/xrays/`).

#### Relevant File Paths
- `KneeWise/backend/config/settings.py` (lines 70–75, 104–106)

#### Security Weaknesses
1. **SQLite Unsuitable for Production Deployment:** SQLite does not support concurrent write operations under multi-clinician load, lacks point-in-time recovery (PITR), and offers no network access controls. Database locking errors (`OperationalError: database is locked`) will occur under moderate concurrent usage.
2. **Zero Automated Database Backups:** No backup scripts, dump procedures, or snapshot routines exist. A filesystem failure or accidental deletion will cause permanent data loss.
3. **No Redundancy for Patient Radiographs:** Uploaded X-rays reside on a single local filesystem without backup, replication, or versioning.
4. **Undefined RTO and RPO:** No formal Recovery Time Objective (RTO) or Recovery Point Objective (RPO) has been established.

#### Severity
**High**

#### Recommended Remediation
- Migrate the database engine to managed PostgreSQL (e.g. AWS RDS PostgreSQL, Google Cloud SQL, or on-premises PostgreSQL) with automated daily snapshots and WAL archiving for point-in-time recovery.
- Migrate media storage to an S3-compatible object storage service (AWS S3, Google Cloud Storage, Azure Blob Storage) using `django-storages` with server-side encryption (AES-256 / KMS) and cross-region replication.
- Formulate a Disaster Recovery Plan defining RTO (< 1 hour) and RPO (< 15 minutes) with scheduled recovery drills.

#### Existing Tests Covering Feature
- None.

#### Missing Security Tests
- Test verifying database backup and restoration to a clean staging environment.
- Test verifying object storage upload, download, and permission validation.

---

## Vulnerability Summary & Severity Matrix

| ID | Category | Finding | Severity | Impact |
| :--- | :--- | :--- | :---: | :--- |
| **SEC-01** | Authorization | Unprotected direct static media access to radiographs | **Critical** | Unauthenticated exposure of Protected Health Information (PHI) |
| **SEC-02** | FastAPI | Zero service-to-service authentication on port 8000 | **Critical** | Direct unauthenticated model invocation bypassing DRF |
| **SEC-03** | FastAPI | Wildcard CORS (`allow_origins=["*"]`) on ML service | **Critical** | Arbitrary cross-origin interaction with inference microservice |
| **SEC-04** | Secrets | Insecure default fallback secret key & `DEBUG=True` | **Critical** | Predictable session signatures and debug traceback leakage |
| **SEC-05** | Secrets | Gitignore omits `.env`, `db.sqlite3`, and `media/` | **Critical** | Accidental version control commit of keys, DB, and PHI |
| **SEC-06** | Authorization | Permanent cascade deletion of diagnostic records | **Critical** | Destruction of medical screening history & HIPAA non-compliance |
| **SEC-07** | Authentication | Missing JWT token blacklist and logout endpoint | **High** | Inability to revoke compromised tokens for 2 hours |
| **SEC-08** | Authentication | Excessive 2-hour JWT access token lifetime | **High** | Extended exposure window upon credential/token theft |
| **SEC-09** | API Protection | Completely missing rate limiting / throttling | **High** | Brute force, upload flooding, and compute exhaustion DoS |
| **SEC-10** | API Protection | Unbounded queryset & lack of pagination on history | **High** | Server memory exhaustion and gateway timeouts on large datasets |
| **SEC-11** | Upload | Client `Content-Type` trusted without magic-byte check | **High** | Potential execution/storage of disguised malicious payloads |
| **SEC-12** | Upload | High in-memory upload buffering (25 MB RAM buffer) | **High** | Server Out-of-Memory (OOM) crash under concurrent uploads |
| **SEC-13** | Audit Logging | Complete absence of PHI access & auth audit logging | **High** | Inability to audit unauthorized access to medical records |
| **SEC-14** | Backup | Single SQLite file & local filesystem storage | **High** | Complete data loss upon disk corruption; database lock contention |
| **SEC-15** | FastAPI | Synchronous inference inside async route blocks event loop | **Medium** | Event loop starvation under concurrent inference traffic |
| **SEC-16** | API Protection | Missing production HTTP security headers | **Medium** | Clickjacking, MIME sniffing, and missing HSTS enforcement |

---

## Prioritized Implementation Plan

Remediation should proceed in three structured phases, adhering strictly to the constraint of keeping the frozen model checkpoint intact and preserving all passing tests.

### Phase 1: Critical Security Controls (Immediate — Pre-Staging Gate)
1. **Secure Radiograph Access (SEC-01):**
   - Disable direct static media serving of radiographs.
   - Implement an authenticated media download endpoint enforcing clinician-patient permission checks.
2. **FastAPI Service-to-Service Authentication (SEC-02, SEC-03):**
   - Add a shared secret header (`X-KneeWise-Service-Token`) validated by FastAPI.
   - Remove wildcard CORS from FastAPI; restrict network binding to `127.0.0.1`.
   - Disable Swagger/OpenAPI docs in production.
3. **Secrets & Source Control Hardening (SEC-04, SEC-05):**
   - Require `DJANGO_SECRET_KEY` when `DEBUG=False`.
   - Default `DEBUG` to `False`.
   - Update `.gitignore` to strictly exclude `.env*`, `*.sqlite3`, and `media/`.
4. **Medical Record Retention Protection (SEC-06):**
   - Change `on_delete=models.CASCADE` on `Patient` foreign keys to `models.PROTECT`.

### Phase 2: High-Priority Hardening (Pre-Production Gate)
1. **Authentication & Token Lifecycle (SEC-07, SEC-08):**
   - Install `rest_framework_simplejwt.token_blacklist`.
   - Implement `/api/auth/logout/` endpoint.
   - Reduce `ACCESS_TOKEN_LIFETIME` to 15 minutes.
   - Remove `SessionAuthentication` from default API classes.
2. **API Throttling & Pagination (SEC-09, SEC-10):**
   - Configure DRF `AnonRateThrottle` (10/min) and `UserRateThrottle` (120/min).
   - Configure global `PageNumberPagination` (`PAGE_SIZE = 25`).
3. **Upload Sanitization & Buffer Reduction (SEC-11, SEC-12):**
   - Implement magic-byte validation (`python-magic` / `filetype`).
   - Re-encode uploaded images via PIL to strip unverified metadata.
   - Reduce `FILE_UPLOAD_MAX_MEMORY_SIZE` to 2.5 MB.
4. **Clinical Audit Logging (SEC-13):**
   - Implement structured PHI access logging recording user, patient, timestamp, and IP.
   - Configure centralized Django `LOGGING` dictionary with file rotation.

### Phase 3: Infrastructure & Operational Resilience (Production Deployment)
1. **Database & Storage Migration (SEC-14):**
   - Migrate from SQLite to managed PostgreSQL.
   - Migrate local media to S3-compatible encrypted object storage.
2. **FastAPI Concurrency Optimization (SEC-15):**
   - Offload PyTorch inference calls to a threadpool (`run_in_threadpool`).
3. **HTTP Security Headers (SEC-16):**
   - Enable HSTS, `X-Content-Type-Options: nosniff`, and `X-Frame-Options: DENY`.
4. **Security Regression Test Suite:**
   - Implement automated security integration tests covering token revocation, rate limiting, BOLA, and upload sanitization.

---
*Report compiled in read-only mode. No application files or model weights were altered.*
