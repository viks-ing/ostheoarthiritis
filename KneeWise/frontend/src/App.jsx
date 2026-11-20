import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import AuthView from './components/AuthView';
import PatientsView from './components/PatientsView';
import ScreeningView from './components/ScreeningView';
import RehabStudioView from './components/RehabStudioView';
import RehabHistoryView from './components/RehabHistoryView';
import ApiInspectorView from './components/ApiInspectorView';
import { authStorage, checkDjangoHealth, checkFastApiHealth } from './api';
import './styles/app.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('screening');
  const [user, setUser] = useState(authStorage.getUser());
  const [backendOnline, setBackendOnline] = useState(false);
  const [fastapiOnline, setFastapiOnline] = useState(false);
  const [selectedPatientForAction, setSelectedPatientForAction] = useState(null);

  // Poll backend health status
  const checkHealth = async () => {
    const djangoRes = await checkDjangoHealth();
    setBackendOnline(djangoRes.online);

    const fastRes = await checkFastApiHealth();
    setFastapiOnline(fastRes.online);
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    authStorage.clear();
    setUser(null);
    setActiveTab('auth');
  };

  const handleSelectPatientForScreening = (patient) => {
    setSelectedPatientForAction(patient);
    setActiveTab('screening');
  };

  const handleSelectPatientForRehab = (patient) => {
    setSelectedPatientForAction(patient);
    setActiveTab('rehab');
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        user={user}
        onLogout={handleLogout}
        backendOnline={backendOnline}
        fastapiOnline={fastapiOnline}
      />

      <main className="main-content">
        {activeTab === 'screening' && (
          <ScreeningView initialPatient={selectedPatientForAction} />
        )}

        {activeTab === 'rehab' && (
          <RehabStudioView
            initialPatient={selectedPatientForAction}
            onSessionCompleted={() => setActiveTab('history')}
          />
        )}

        {activeTab === 'patients' && (
          <PatientsView
            onSelectPatientForScreening={handleSelectPatientForScreening}
            onSelectPatientForRehab={handleSelectPatientForRehab}
          />
        )}

        {activeTab === 'history' && <RehabHistoryView />}

        {activeTab === 'inspector' && <ApiInspectorView />}

        {activeTab === 'auth' && (
          <AuthView
            user={user}
            setUser={setUser}
            onLoginSuccess={() => setActiveTab('screening')}
          />
        )}
      </main>

      <footer style={{ background: 'var(--bg-secondary)', borderTop: '1px solid var(--border-color)', padding: '1rem', textAlign: 'center', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        KneeWise Clinical AI & Rehabilitation Platform • EfficientNet-B0 Frozen Champion • MediaPipe 2D Planar Kinematics
      </footer>
    </div>
  );
}
