import React from 'react';

export default function Navbar({ activeTab, setActiveTab, user, onLogout, backendOnline, fastapiOnline }) {
  const tabs = [
    { id: 'screening', label: '🩻 X-ray AI Screening' },
    { id: 'rehab', label: '🏃 Rehabilitation Studio' },
    { id: 'patients', label: '👥 Patients Directory' },
    { id: 'history', label: '📊 Rehab Session History' },
    { id: 'inspector', label: '🩺 API Inspector' },
    { id: 'auth', label: user ? '👤 Clinician Profile' : '🔐 Login' },
  ];

  return (
    <header>
      <div className="app-header">
        <div className="brand">
          <div className="brand-icon">K</div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="brand-title">KneeWise</span>
              <span className="brand-tag">Clinical AI</span>
            </div>
          </div>
        </div>

        <div className="header-status">
          <div className="status-pill" title="Django REST Backend: http://127.0.0.1:8080">
            <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`} />
            <span>Django (8080): {backendOnline ? 'Online' : 'Offline'}</span>
          </div>

          <div className="status-pill" title="FastAPI ML Microservice: http://127.0.0.1:8000">
            <span className={`status-dot ${fastapiOnline ? 'online' : 'offline'}`} />
            <span>FastAPI (8000): {fastapiOnline ? 'Online' : 'Offline'}</span>
          </div>

          {user ? (
            <div className="status-pill" style={{ borderColor: 'rgba(6, 182, 212, 0.4)' }}>
              <span>Dr. {user.username}</span>
              <button
                onClick={onLogout}
                className="btn btn-sm"
                style={{ padding: '0.1rem 0.4rem', background: 'transparent', color: '#f87171' }}
              >
                Logout
              </button>
            </div>
          ) : (
            <button onClick={() => setActiveTab('auth')} className="btn btn-primary btn-sm">
              Sign In
            </button>
          )}
        </div>
      </div>

      <nav className="nav-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </header>
  );
}
