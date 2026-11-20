import React, { useState } from 'react';
import { DJANGO_BASE_URL, FASTAPI_BASE_URL, authStorage } from '../api';

export default function ApiInspectorView() {
  const [method, setMethod] = useState('GET');
  const [url, setUrl] = useState(`${FASTAPI_BASE_URL}/health`);
  const [body, setBody] = useState('');
  const [responseStatus, setResponseStatus] = useState(null);
  const [responseHeaders, setResponseHeaders] = useState({});
  const [responseData, setResponseData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const presets = [
    { label: 'FastAPI Health Check', method: 'GET', url: `${FASTAPI_BASE_URL}/health`, body: '' },
    { label: 'Django Patients List', method: 'GET', url: `${DJANGO_BASE_URL}/api/screening/patients/`, body: '' },
    { label: 'Django Rehab Sessions', method: 'GET', url: `${DJANGO_BASE_URL}/api/rehab/sessions/`, body: '' },
    {
      label: 'Django JWT Auth',
      method: 'POST',
      url: `${DJANGO_BASE_URL}/api/auth/token/`,
      body: JSON.stringify({ username: 'dr_sharma', password: 'SecurePass123!' }, null, 2),
    },
  ];

  const handleSelectPreset = (p) => {
    setMethod(p.method);
    setUrl(p.url);
    setBody(p.body);
  };

  const handleExecute = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponseStatus(null);
    setResponseData(null);

    const token = authStorage.getToken();
    const headers = {};
    if (token && url.startsWith(DJANGO_BASE_URL)) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    if (method !== 'GET' && method !== 'HEAD') {
      headers['Content-Type'] = 'application/json';
    }

    try {
      const options = { method, headers };
      if (body && (method === 'POST' || method === 'PUT')) {
        options.body = body;
      }

      const res = await fetch(url, options);
      setResponseStatus(res.status);

      const headerObj = {};
      res.headers.forEach((val, key) => {
        headerObj[key] = val;
      });
      setResponseHeaders(headerObj);

      const contentType = res.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const json = await res.json();
        setResponseData(JSON.stringify(json, null, 2));
      } else {
        const text = await res.text();
        setResponseData(text);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2 className="card-title">🩺 Live API Diagnostics & Inspector</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem', marginBottom: '1.25rem' }}>
          Inspect REST endpoints, verify CORS headers, and audit raw responses across Django and FastAPI.
        </p>

        {/* Quick Presets */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Presets:</span>
          {presets.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSelectPreset(p)}
              className="btn btn-secondary btn-sm"
            >
              {p.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleExecute}>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
            <select
              className="form-select"
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              style={{ width: '120px' }}
            >
              <option value="GET">GET</option>
              <option value="POST">POST</option>
              <option value="PUT">PUT</option>
              <option value="DELETE">DELETE</option>
            </select>
            <input
              type="text"
              className="form-input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              style={{ flex: 1 }}
            />
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Sending...' : 'Send Request'}
            </button>
          </div>

          {(method === 'POST' || method === 'PUT') && (
            <div className="form-group">
              <label className="form-label">Request Body (JSON)</label>
              <textarea
                className="form-textarea"
                rows={4}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                style={{ fontFamily: 'var(--font-mono)', fontSize: '0.825rem' }}
              />
            </div>
          )}
        </form>

        {error && (
          <div className="alert-banner danger" style={{ marginTop: '1rem' }}>
            <span>⚠️ Network Error: {error}</span>
          </div>
        )}

        {responseStatus !== null && (
          <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
              <span className="form-label" style={{ margin: 0 }}>Response Status:</span>
              <span
                className="brand-tag"
                style={{
                  background: responseStatus < 300 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                  borderColor: responseStatus < 300 ? '#10b981' : '#ef4444',
                  color: responseStatus < 300 ? '#6ee7b7' : '#fca5a5',
                  fontSize: '0.85rem',
                }}
              >
                {responseStatus}
              </span>
            </div>

            <div className="form-group">
              <label className="form-label">Response Payload</label>
              <pre className="code-block">{responseData || 'Empty response'}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
