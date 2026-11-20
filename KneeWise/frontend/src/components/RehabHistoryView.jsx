import React, { useState, useEffect } from 'react';
import { getRehabSessions, getPatients } from '../api';

export default function RehabHistoryView() {
  const [sessions, setSessions] = useState([]);
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchSessions = async (patientId = null) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getRehabSessions(patientId);
      setSessions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    getPatients().then(setPatients).catch(console.warn);
    fetchSessions();
  }, []);

  const handlePatientFilterChange = (e) => {
    const val = e.target.value;
    setSelectedPatientId(val);
    fetchSessions(val || null);
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">📊 Rehabilitation Session History</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem', marginTop: '0.2rem' }}>
            Historical exercise adherence records, angular excursion metrics, and multi-set logs.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <select
            className="form-select"
            value={selectedPatientId}
            onChange={handlePatientFilterChange}
            style={{ width: '220px' }}
          >
            <option value="">All Authorized Patients</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.patient_id} - {p.first_name} {p.last_name}
              </option>
            ))}
          </select>
          <button onClick={() => fetchSessions(selectedPatientId || null)} className="btn btn-secondary btn-sm">
            🔄 Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-banner danger">
          <span>⚠️ {error}</span>
        </div>
      )}

      {loading ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
          Loading session logs...
        </div>
      ) : sessions.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
          No rehabilitation sessions recorded yet. Start a session in the <strong>Rehabilitation Studio</strong>.
        </div>
      ) : (
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient</th>
                <th>Exercise</th>
                <th>Leg</th>
                <th>Status</th>
                <th>Reps</th>
                <th>Sets</th>
                <th>Avg ROM</th>
                <th>Best ROM</th>
                <th>Availability</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>
                    <strong>{s.patient_name || 'Patient'}</strong>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{s.patient_mrn}</div>
                  </td>
                  <td>{s.exercise_name?.replace(/_/g, ' ')}</td>
                  <td>
                    <span style={{ textTransform: 'uppercase', fontWeight: 600 }}>{s.selected_leg}</span>
                  </td>
                  <td>
                    <span
                      className="brand-tag"
                      style={{
                        background: s.status === 'COMPLETED' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                        borderColor: s.status === 'COMPLETED' ? '#10b981' : '#f59e0b',
                        color: s.status === 'COMPLETED' ? '#6ee7b7' : '#fcd34d',
                      }}
                    >
                      {s.status}
                    </span>
                  </td>
                  <td>
                    <strong style={{ color: 'var(--success-green)' }}>{s.total_reps}</strong>
                    {s.rejected_reps > 0 && (
                      <span style={{ fontSize: '0.75rem', color: 'var(--danger-red)', marginLeft: '0.25rem' }}>
                        (-{s.rejected_reps})
                      </span>
                    )}
                  </td>
                  <td>{s.set_count || s.sets?.length || 1}</td>
                  <td>{s.avg_excursion_deg ? `${s.avg_excursion_deg.toFixed(1)}°` : '0°'}</td>
                  <td>{s.best_excursion_deg ? `${s.best_excursion_deg.toFixed(1)}°` : '0°'}</td>
                  <td>
                    <span style={{ color: s.tracking_availability_pct >= 80 ? 'var(--success-green)' : 'var(--warning-amber)' }}>
                      {s.tracking_availability_pct ? `${s.tracking_availability_pct.toFixed(1)}%` : '0%'}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {new Date(s.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
