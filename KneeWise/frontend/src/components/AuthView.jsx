import React, { useState } from 'react';
import { login, authStorage } from '../api';

export default function AuthView({ user, setUser, onLoginSuccess }) {
  const [username, setUsername] = useState('dr_sharma');
  const [password, setPassword] = useState('SecurePass123!');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const data = await login(username, password);
      setUser({ username });
      setSuccess(`Authenticated successfully as ${username}! JWT tokens stored.`);
      if (onLoginSuccess) onLoginSuccess();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    authStorage.clear();
    setUser(null);
    setSuccess('Signed out. Tokens cleared from storage.');
  };

  const token = authStorage.getToken();

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto' }}>
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🔐 Clinician Authentication</h2>
          <span className="brand-tag">JWT Bearer</span>
        </div>

        {error && (
          <div className="alert-banner danger">
            <span>⚠️ {error}</span>
          </div>
        )}

        {success && (
          <div className="alert-banner success">
            <span>✅ {success}</span>
          </div>
        )}

        {user ? (
          <div>
            <div className="alert-banner info" style={{ marginBottom: '1.5rem' }}>
              <div>
                <strong>Active Session:</strong> Logged in as <code>{user.username}</code>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Active JWT Access Token</label>
              <div className="code-block" style={{ maxHeight: '120px' }}>
                {token || 'No token found'}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
              <button onClick={handleLogout} className="btn btn-danger">
                Sign Out
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1.25rem' }}>
              Sign in with your clinician account to access patient medical records, submit radiographs for AI analysis, and manage rehabilitation sessions.
            </p>

            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                type="text"
                className="form-input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <input
                type="password"
                className="form-input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Demo Credentials: <code>dr_sharma</code> / <code>SecurePass123!</code> or create a new user via Django admin.
              </span>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
              {loading ? 'Authenticating...' : 'Sign In as Clinician'}
            </button>
          </form>
        )}
      </div>

      <div className="card">
        <h3 className="card-title" style={{ fontSize: '0.95rem', marginBottom: '0.75rem' }}>
          🛡️ Security & Access Control Features
        </h3>
        <ul style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', paddingLeft: '1.2rem', lineHeight: '1.7' }}>
          <li>Role-based patient isolation: Clinicians can only access radiographs and sessions of patients assigned to them.</li>
          <li>SEC-01 compliance: Public media access is forbidden; medical images are served through authenticated streams.</li>
          <li>Short-lived JWT tokens with secure token refresh rotation.</li>
        </ul>
      </div>
    </div>
  );
}
