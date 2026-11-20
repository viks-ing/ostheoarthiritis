import React, { useState, useEffect } from 'react';
import { getPatients, createPatient } from '../api';

export default function PatientsView({ onSelectPatientForScreening, onSelectPatientForRehab }) {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);

  // Form fields
  const [patientId, setPatientId] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dob, setDob] = useState('1975-06-15');
  const [gender, setGender] = useState('M');
  const [saving, setSaving] = useState(false);

  const fetchPatientList = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getPatients();
      setPatients(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatientList();
  }, []);

  const handleCreatePatient = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await createPatient({
        patient_id: patientId || `MRN-${Math.floor(1000 + Math.random() * 9000)}`,
        first_name: firstName,
        last_name: lastName,
        date_of_birth: dob,
        gender: gender,
      });
      setShowAddModal(false);
      setPatientId('');
      setFirstName('');
      setLastName('');
      fetchPatientList();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">👥 Clinical Patient Directory</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem', marginTop: '0.2rem' }}>
              Patients assigned to your clinician account. Select a patient to begin knee screening or start rehabilitation.
            </p>
          </div>
          <button onClick={() => setShowAddModal(!showAddModal)} className="btn btn-primary btn-sm">
            {showAddModal ? 'Cancel' : '+ New Patient'}
          </button>
        </div>

        {error && (
          <div className="alert-banner danger">
            <span>⚠️ {error}</span>
          </div>
        )}

        {showAddModal && (
          <form onSubmit={handleCreatePatient} className="card" style={{ background: 'var(--bg-secondary)', marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '0.95rem', marginBottom: '1rem', color: 'var(--accent-cyan)' }}>
              Register New Patient
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div className="form-group">
                <label className="form-label">Medical Record Number (MRN)</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. MRN-501"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">First Name</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="John"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Last Name</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="Doe"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Date of Birth</label>
                <input
                  type="date"
                  className="form-input"
                  value={dob}
                  onChange={(e) => setDob(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Gender</label>
                <select className="form-select" value={gender} onChange={(e) => setGender(e.target.value)}>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="O">Other</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
              <button type="button" onClick={() => setShowAddModal(false)} className="btn btn-secondary btn-sm">
                Cancel
              </button>
              <button type="submit" className="btn btn-success btn-sm" disabled={saving}>
                {saving ? 'Saving...' : 'Save Patient'}
              </button>
            </div>
          </form>
        )}

        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
            Loading patients from Django backend...
          </div>
        ) : patients.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
            No patients registered for your account yet. Click "+ New Patient" to create one.
          </div>
        ) : (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>MRN</th>
                  <th>Patient Name</th>
                  <th>Gender</th>
                  <th>Date of Birth</th>
                  <th>Created At</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p) => (
                  <tr key={p.id}>
                    <td>
                      <code style={{ color: 'var(--accent-cyan)' }}>{p.patient_id}</code>
                    </td>
                    <td>
                      <strong>{p.first_name} {p.last_name}</strong>
                    </td>
                    <td>{p.gender === 'M' ? 'Male' : p.gender === 'F' ? 'Female' : p.gender}</td>
                    <td>{p.date_of_birth || 'N/A'}</td>
                    <td>{new Date(p.created_at).toLocaleDateString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        onClick={() => onSelectPatientForScreening(p)}
                        className="btn btn-primary btn-sm"
                        style={{ marginRight: '0.5rem' }}
                      >
                        🩻 Screen X-ray
                      </button>
                      <button
                        onClick={() => onSelectPatientForRehab(p)}
                        className="btn btn-secondary btn-sm"
                      >
                        🏃 Start Rehab
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
