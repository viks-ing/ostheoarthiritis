import React, { useState, useEffect } from 'react';
import { getPatients, uploadXray, triggerPrediction, fetchProtectedXrayBlob } from '../api';

const KL_LABELS = {
  0: { title: 'KL0: Normal', desc: 'No radiographic features of osteoarthritis', color: '#10b981' },
  1: { title: 'KL1: Doubtful', desc: 'Possible osteophytic lipping and doubtful narrowing', color: '#38bdf8' },
  2: { title: 'KL2: Minimal', desc: 'Definite osteophytes, definite unimpaired joint space', color: '#f59e0b' },
  3: { title: 'KL3: Moderate', desc: 'Moderate joint space reduction, multiple osteophytes', color: '#f97316' },
  4: { title: 'KL4: Severe', desc: 'Severe joint space loss, marked sclerosis, bone deformity', color: '#ef4444' },
};

export default function ScreeningView({ initialPatient }) {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(initialPatient ? initialPatient.id : '');
  const [laterality, setLaterality] = useState('RIGHT');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [threshold, setThreshold] = useState(0.30);

  // Status states
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [protectedImageUrl, setProtectedImageUrl] = useState(null);

  useEffect(() => {
    getPatients()
      .then((data) => {
        setPatients(data);
        if (!selectedPatientId && data.length > 0) {
          setSelectedPatientId(data[0].id);
        }
      })
      .catch((err) => setError('Could not load patients: ' + err.message));
  }, []);

  useEffect(() => {
    if (initialPatient) {
      setSelectedPatientId(initialPatient.id);
    }
  }, [initialPatient]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictionResult(null);
      setProtectedImageUrl(null);
    }
  };

  const handleUploadAndPredict = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select a knee radiograph image file (PNG/JPEG).');
      return;
    }
    if (!selectedPatientId) {
      setError('Please select a patient.');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      // 1. Upload to Django backend
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('image', selectedFile);
      formData.append('patient_id', selectedPatientId);
      formData.append('patient', selectedPatientId);
      formData.append('knee_laterality', laterality);
      formData.append('laterality', laterality);

      const uploadRes = await uploadXray(formData);
      const xrayId = uploadRes.id || uploadRes.uploaded_xray?.id;

      // 2. Trigger AI Inference pipeline
      const prediction = await triggerPrediction(xrayId, threshold);
      setPredictionResult(prediction);

      // 3. Test SEC-01 protected image fetch
      try {
        const blob = await fetchProtectedXrayBlob(xrayId);
        setProtectedImageUrl(URL.createObjectURL(blob));
      } catch (imgErr) {
        console.warn('Protected image fetch warning:', imgErr);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 420px) 1fr', gap: '1.5rem' }}>
      {/* Upload and Configuration Column */}
      <div className="card">
        <h2 className="card-title">🩻 Knee Radiograph Upload</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.825rem', marginBottom: '1.25rem' }}>
          Upload high-resolution weight-bearing bilateral or unilateral knee X-ray for Kellgren-Lawrence grading.
        </p>

        {error && (
          <div className="alert-banner danger">
            <span>⚠️ {error}</span>
          </div>
        )}

        <form onSubmit={handleUploadAndPredict}>
          <div className="form-group">
            <label className="form-label">Patient</label>
            <select
              className="form-select"
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              required
            >
              <option value="">-- Select Patient --</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.patient_id} - {p.first_name} {p.last_name}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Knee Laterality</label>
            <select
              className="form-select"
              value={laterality}
              onChange={(e) => setLaterality(e.target.value)}
            >
              <option value="RIGHT">Right Knee</option>
              <option value="LEFT">Left Knee</option>
              <option value="BILATERAL">Bilateral</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Select Radiograph File (PNG/JPEG)</label>
            <input
              type="file"
              accept=".png,.jpg,.jpeg"
              onChange={handleFileChange}
              className="form-input"
            />
          </div>

          {previewUrl && (
            <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
              <img
                src={previewUrl}
                alt="Selected Radiograph"
                style={{
                  maxHeight: '180px',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  maxWidth: '100%',
                }}
              />
            </div>
          )}

          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <label className="form-label">Binary Screening Threshold</label>
              <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Provisional operating point: 0.30 (Sens: 80.86%, Spec: 88.66%)
            </span>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', marginTop: '0.5rem' }}
            disabled={uploading || !selectedFile}
          >
            {uploading ? 'Processing AI Inference...' : '🚀 Submit & Run AI Grading'}
          </button>
        </form>
      </div>

      {/* Results and Provenance Column */}
      <div>
        {predictionResult ? (
          <div>
            <div className="card">
              <div className="card-header">
                <div>
                  <h2 className="card-title">Analysis Results</h2>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Processed in {predictionResult.latency_ms ? `${predictionResult.latency_ms.toFixed(1)} ms` : 'N/A'}
                  </span>
                </div>
                <div
                  className="brand-tag"
                  style={{
                    background: predictionResult.binary_referral_recommended
                      ? 'rgba(239, 68, 68, 0.2)'
                      : 'rgba(16, 185, 129, 0.2)',
                    borderColor: predictionResult.binary_referral_recommended ? '#ef4444' : '#10b981',
                    color: predictionResult.binary_referral_recommended ? '#fca5a5' : '#6ee7b7',
                    fontSize: '0.8rem',
                  }}
                >
                  {predictionResult.binary_referral_recommended
                    ? '⚠️ REFERRAL RECOMMENDED'
                    : '✅ ROUTINE MONITORING'}
                </div>
              </div>

              {/* Primary Metric Tiles */}
              <div className="metrics-grid">
                <div className="metric-tile">
                  <span className="metric-label">Predicted Severity</span>
                  <span
                    className="metric-value"
                    style={{ color: KL_LABELS[predictionResult.predicted_grade]?.color || '#fff' }}
                  >
                    KL {predictionResult.predicted_grade}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    {KL_LABELS[predictionResult.predicted_grade]?.desc}
                  </span>
                </div>

                <div className="metric-tile">
                  <span className="metric-label">Definite OA Risk P(KL≥2)</span>
                  <span
                    className="metric-value"
                    style={{
                      color: predictionResult.binary_oa_risk >= threshold ? '#ef4444' : '#10b981',
                    }}
                  >
                    {(predictionResult.binary_oa_risk * 100).toFixed(1)}%
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    Operating threshold: {threshold.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* 5-Class Probability Distribution */}
              <div style={{ marginTop: '1.25rem' }}>
                <h3 style={{ fontSize: '0.85rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Kellgren-Lawrence Probability Spectrum
                </h3>

                {[0, 1, 2, 3, 4].map((grade) => {
                  const prob = predictionResult.kl_probabilities
                    ? predictionResult.kl_probabilities[`KL${grade}`] || 0
                    : 0;
                  const pct = (prob * 100).toFixed(1);
                  const isTop = predictionResult.predicted_grade === grade;
                  const info = KL_LABELS[grade];

                  return (
                    <div key={grade} className="prob-bar-container">
                      <div className="prob-bar-header">
                        <span style={{ fontWeight: isTop ? 700 : 400, color: isTop ? info.color : 'var(--text-primary)' }}>
                          {info.title}
                        </span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                          {pct}%
                        </span>
                      </div>
                      <div className="prob-bar-bg">
                        <div
                          className="prob-bar-fill"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: info.color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Protected Image Access View */}
              {protectedImageUrl && (
                <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                  <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                    🔒 SEC-01 Verified Protected Image Stream
                  </h4>
                  <img
                    src={protectedImageUrl}
                    alt="Protected X-ray View"
                    style={{
                      maxHeight: '260px',
                      borderRadius: '8px',
                      border: '1px solid var(--border-color)',
                      boxShadow: '0 4px 10px rgba(0,0,0,0.5)',
                    }}
                  />
                </div>
              )}
            </div>

            {/* Governance & Provenance Card */}
            <div className="card">
              <h3 className="card-title" style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>
                🛡️ Checkpoint Integrity & Governance
              </h3>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <div>Architecture: <strong>EfficientNet-B0</strong> (Experiment D Champion)</div>
                <div>Checksum SHA-256: <code style={{ color: 'var(--accent-cyan)' }}>d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b</code></div>
                <div>Training state: <strong>FROZEN</strong> (No retraining in production)</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '380px', textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🩻</div>
            <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>No Radiograph Analyzed Yet</h3>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '420px', fontSize: '0.875rem' }}>
              Select a patient, attach a knee radiograph image on the left, and click <strong>"Submit & Run AI Grading"</strong> to trigger the microservice pipeline.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
