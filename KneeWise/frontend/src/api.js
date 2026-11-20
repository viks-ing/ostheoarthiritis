/**
 * api.js - KneeWise Frontend API Service Layer
 * Bridges React UI with Django REST Backend (8080) and FastAPI ML Microservice (8000)
 */

export const DJANGO_BASE_URL = 'http://127.0.0.1:8080';
export const FASTAPI_BASE_URL = 'http://127.0.0.1:8000';

// Local storage keys
const TOKEN_KEY = 'kneewise_access_token';
const REFRESH_KEY = 'kneewise_refresh_token';
const USER_KEY = 'kneewise_user';

export const authStorage = {
  getToken: () => localStorage.getItem(TOKEN_KEY),
  getRefreshToken: () => localStorage.getItem(REFRESH_KEY),
  setTokens: (access, refresh) => {
    localStorage.setItem(TOKEN_KEY, access);
    if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
  },
  getUser: () => {
    const raw = localStorage.getItem(USER_KEY);
    try {
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  },
  setUser: (user) => localStorage.setItem(USER_KEY, JSON.stringify(user)),
  clear: () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  },
};

/**
 * Performs authenticated requests to Django backend with automatic Bearer token injection.
 */
export async function apiFetch(endpoint, options = {}) {
  const token = authStorage.getToken();
  const headers = {
    ...(options.headers || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Do not set Content-Type if sending FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const url = endpoint.startsWith('http') ? endpoint : `${DJANGO_BASE_URL}${endpoint}`;
  const response = await fetch(url, { ...options, headers });

  if (response.status === 401) {
    // Attempt token refresh or clear
    authStorage.clear();
  }

  return response;
}

// -------------------------------------------------------------
// Authentication Endpoints
// -------------------------------------------------------------

export async function login(username, password) {
  const response = await fetch(`${DJANGO_BASE_URL}/api/auth/token/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Authentication failed' }));
    throw new Error(err.detail || 'Invalid clinician credentials');
  }

  const data = await response.json();
  authStorage.setTokens(data.access, data.refresh);
  authStorage.setUser({ username });
  return data;
}

export async function checkFastApiHealth() {
  try {
    const response = await fetch(`${FASTAPI_BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) return { online: false, data: null };
    const data = await response.json();
    return { online: true, data };
  } catch (err) {
    return { online: false, error: err.message };
  }
}

export async function checkDjangoHealth() {
  try {
    // Probe unauthenticated route or root
    const response = await fetch(`${DJANGO_BASE_URL}/api/auth/token/`, {
      method: 'OPTIONS',
      signal: AbortSignal.timeout(3000),
    });
    return { online: response.status < 500 };
  } catch (err) {
    return { online: false, error: err.message };
  }
}

// -------------------------------------------------------------
// Patients Directory Endpoints
// -------------------------------------------------------------

export async function getPatients() {
  const res = await apiFetch('/api/screening/patients/');
  if (!res.ok) throw new Error(`Failed to fetch patients (${res.status})`);
  const data = await res.json();
  return Array.isArray(data) ? data : data.results || [];
}

export async function createPatient(patientData) {
  const res = await apiFetch('/api/screening/patients/', {
    method: 'POST',
    body: JSON.stringify(patientData),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to create patient' }));
    throw new Error(JSON.stringify(err));
  }
  return await res.json();
}

// -------------------------------------------------------------
// X-ray Screening & AI Inference Endpoints
// -------------------------------------------------------------

export async function uploadXray(formData) {
  const res = await apiFetch('/api/screening/upload/', {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || JSON.stringify(err));
  }
  return await res.json();
}

export async function triggerPrediction(xrayId, threshold = 0.30) {
  const res = await apiFetch(`/api/screening/predict/${xrayId}/?threshold=${threshold}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || 'Inference service error');
  }
  return await res.json();
}

export async function fetchProtectedXrayBlob(xrayId) {
  const res = await apiFetch(`/api/screening/xrays/${xrayId}/`);
  if (!res.ok) throw new Error(`Could not load radiograph (${res.status})`);
  return await res.blob();
}

// -------------------------------------------------------------
// Rehabilitation Sessions Endpoints
// -------------------------------------------------------------

export async function getRehabSessions(patientId = null) {
  const query = patientId ? `?patient=${patientId}` : '';
  const res = await apiFetch(`/api/rehab/sessions/${query}`);
  if (!res.ok) throw new Error(`Failed to fetch rehab sessions (${res.status})`);
  const data = await res.json();
  return Array.isArray(data) ? data : data.results || [];
}

export async function createRehabSession(sessionData) {
  const res = await apiFetch('/api/rehab/sessions/', {
    method: 'POST',
    body: JSON.stringify(sessionData),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start rehab session' }));
    throw new Error(JSON.stringify(err));
  }
  return await res.json();
}

export async function completeRehabSession(sessionId, completionData) {
  const res = await apiFetch(`/api/rehab/sessions/${sessionId}/complete/`, {
    method: 'POST',
    body: JSON.stringify(completionData),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to complete session' }));
    throw new Error(JSON.stringify(err));
  }
  return await res.json();
}
