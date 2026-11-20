/**
 * RehabStudioView.jsx - Real-Time Webcam Rehabilitation CV Studio
 * Completely webcam-driven with real MediaPipe Pose detection and vector angle biomechanics.
 * Zero simulations, zero animated SVG placeholders, zero fake numbers.
 */

import React, { useState, useEffect, useRef } from 'react';
import { getPatients, createRehabSession, completeRehabSession, authStorage, DJANGO_BASE_URL } from '../api';

export default function RehabStudioView({ initialPatient, onSessionCompleted }) {
  // Session & Biometric States
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(initialPatient ? initialPatient.id : '');
  const [selectedLeg, setSelectedLeg] = useState('right');
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const [kneeAngle, setKneeAngle] = useState(null);
  const [smoothedAngle, setSmoothedAngle] = useState(null);
  const [repCount, setRepCount] = useState(0);
  const [exerciseState, setExerciseState] = useState('IDLE');
  const [feedback, setFeedback] = useState('Enable camera to begin seated knee extension tracking.');
  const [feedbackSeverity, setFeedbackSeverity] = useState('info');
  const [trackingStatus, setTrackingStatus] = useState('OFF');
  const [landmarkConfidence, setLandmarkConfidence] = useState(0);

  // Camera & Device States
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [transportMode, setTransportMode] = useState('WS'); // 'WS' or 'HTTP'

  // Persistent Summary
  const [summaryReport, setSummaryReport] = useState(null);
  const [savingSession, setSavingSession] = useState(false);
  const [error, setError] = useState(null);

  // DOM & Pipeline Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const wsRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const isProcessingFrameRef = useRef(false);

  // Metric Accumulators
  const sessionStartRef = useRef(null);
  const totalFramesRef = useRef(0);
  const trackedFramesRef = useRef(0);
  const excursionsRef = useRef([]);
  const maxAnglesRef = useRef([]);

  // 1. Load Patients and Cameras
  useEffect(() => {
    getPatients()
      .then((data) => {
        setPatients(data);
        if (!selectedPatientId && data.length > 0) {
          setSelectedPatientId(data[0].id);
        }
      })
      .catch((err) => setError('Could not load patients: ' + err.message));

    loadVideoDevices();
  }, []);

  const loadVideoDevices = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      setCameraError('MediaDevices API not supported in this browser.');
      return;
    }
    try {
      const allDevs = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = allDevs.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevs[0].deviceId);
      }
    } catch (e) {
      console.warn('Could not enumerate video devices:', e);
    }
  };

  // 2. Explicit Camera Start
  const startCamera = async () => {
    setCameraLoading(true);
    setCameraError(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Camera access requires localhost or HTTPS origin.');
      setCameraLoading(false);
      return;
    }

    try {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }

      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          facingMode: selectedDeviceId ? undefined : { ideal: 'user' },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.onended = () => {
          setCameraError('Camera device was disconnected.');
          stopCamera();
        };
      }

      setCameraActive(true);
      setTrackingStatus('CAMERA_READY');
      setFeedback('Camera active. Position yourself sideways with hip, knee, and ankle in frame.');
      await loadVideoDevices();
    } catch (err) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError('Camera permission was denied. Please allow access in browser site settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraError('No webcam detected on this computer.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setCameraError('Webcam is currently in use by another program.');
      } else {
        setCameraError(`Camera error: ${err.message || err.name}`);
      }
      setCameraActive(false);
    } finally {
      setCameraLoading(false);
    }
  };

  // 3. Stop Camera
  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setTrackingStatus('OFF');
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    };
  }, []);

  // 4. Draw Genuine Detected Skeleton Overlay
  const drawPoseOverlay = (landmarks) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !video.videoWidth) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || !landmarks.hip || !landmarks.knee || !landmarks.ankle) {
      return;
    }

    const w = canvas.width;
    const h = canvas.height;

    const hipPx = { x: landmarks.hip.x * w, y: landmarks.hip.y * h };
    const kneePx = { x: landmarks.knee.x * w, y: landmarks.knee.y * h };
    const anklePx = { x: landmarks.ankle.x * w, y: landmarks.ankle.y * h };

    // Thigh Segment
    ctx.beginPath();
    ctx.moveTo(hipPx.x, hipPx.y);
    ctx.lineTo(kneePx.x, kneePx.y);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Shin Segment
    ctx.beginPath();
    ctx.moveTo(kneePx.x, kneePx.y);
    ctx.lineTo(anklePx.x, anklePx.y);
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Joint Nodes
    ctx.beginPath();
    ctx.arc(hipPx.x, hipPx.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(kneePx.x, kneePx.y, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#06b6d4';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(anklePx.x, anklePx.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Angle HUD at Knee
    if (smoothedAngle !== null) {
      ctx.font = 'bold 22px Inter, sans-serif';
      const text = `${Math.round(smoothedAngle)}°`;
      const textWidth = ctx.measureText(text).width;

      ctx.fillStyle = 'rgba(11, 15, 25, 0.85)';
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(kneePx.x + 12, kneePx.y - 28, textWidth + 16, 32, 6);
        ctx.fill();
      } else {
        ctx.fillRect(kneePx.x + 12, kneePx.y - 28, textWidth + 16, 32);
      }

      ctx.fillStyle = '#38bdf8';
      ctx.fillText(text, kneePx.x + 20, kneePx.y - 6);
    }
  };

  // 5. Handle response from Python CV
  const handleCvResponse = (data) => {
    if (!data || data.error) {
      if (data && data.error) console.warn('CV error:', data.error);
      return;
    }

    totalFramesRef.current += 1;
    if (data.is_tracking_reliable) {
      trackedFramesRef.current += 1;
    }

    setTrackingStatus(data.tracking_status || 'UNKNOWN');
    setKneeAngle(data.knee_angle_deg);
    setSmoothedAngle(data.smoothed_angle_deg);
    setRepCount(data.repetitions || 0);
    setExerciseState(data.exercise_state || 'WAITING');
    setFeedback(data.feedback || '');
    setFeedbackSeverity(data.feedback_severity || 'info');
    setLandmarkConfidence(data.landmark_confidence || 0);

    if (data.metrics && data.metrics.current_excursion > 0) {
      excursionsRef.current.push(data.metrics.current_excursion);
    }
    if (data.metrics && data.metrics.max_extension_in_rep > 0) {
      maxAnglesRef.current.push(data.metrics.max_extension_in_rep);
    }

    drawPoseOverlay(data.landmarks);
  };

  // 6. Send live frame to Python CV
  const captureAndSendFrame = async () => {
    if (isProcessingFrameRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return;

    isProcessingFrameRef.current = true;

    try {
      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement('canvas');
      }
      const offCanvas = offscreenCanvasRef.current;
      offCanvas.width = 640;
      offCanvas.height = 480;
      const ctx = offCanvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 640, 480);

      const b64Image = offCanvas.toDataURL('image/jpeg', 0.65);

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ image: b64Image }));
      } else {
        const token = authStorage.getToken();
        const res = await fetch(`${DJANGO_BASE_URL}/api/rehab/process-frame/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            session_id: sessionId,
            image: b64Image,
            selected_leg: selectedLeg,
          }),
        });
        if (res.ok) {
          const result = await res.json();
          handleCvResponse(result);
        }
      }
    } catch (e) {
      console.warn('Frame stream error:', e);
    } finally {
      isProcessingFrameRef.current = false;
    }
  };

  // 7. Start Session
  const handleStartSession = async () => {
    if (!selectedPatientId) {
      setError('Please select a patient first.');
      return;
    }
    setError(null);
    setSavingSession(true);

    if (!cameraActive) {
      await startCamera();
    }

    try {
      const res = await createRehabSession({
        patient: selectedPatientId,
        exercise_name: 'seated_knee_extension',
        selected_leg: selectedLeg,
      });

      const activeSessionId = res.id;
      setSessionId(activeSessionId);
      setSessionActive(true);
      setRepCount(0);
      setKneeAngle(null);
      setSmoothedAngle(null);
      setExerciseState('INITIALIZING');
      setFeedback('Connecting to Python CV engine...');
      setSummaryReport(null);

      sessionStartRef.current = Date.now() / 1000;
      totalFramesRef.current = 0;
      trackedFramesRef.current = 0;
      excursionsRef.current = [];
      maxAnglesRef.current = [];

      // Connect to WebSocket server on port 8002
      const token = authStorage.getToken();
      const wsUrl = `ws://127.0.0.1:8002`;
      try {
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          setTransportMode('WS');
          ws.send(
            JSON.stringify({
              action: 'auth',
              token: token,
              session_id: activeSessionId,
              selected_leg: selectedLeg,
            })
          );
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.status === 'authenticated') {
              setFeedback('Connected to Python CV! Position yourself sideways to camera.');
            } else {
              handleCvResponse(msg);
            }
          } catch (pe) {
            console.warn('WS parse error:', pe);
          }
        };

        ws.onerror = () => setTransportMode('HTTP');
        ws.onclose = () => setTransportMode('HTTP');
        wsRef.current = ws;
      } catch (wsErr) {
        setTransportMode('HTTP');
      }

      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = setInterval(captureAndSendFrame, 66);
    } catch (err) {
      setError('Failed to start session: ' + err.message);
      setSessionActive(false);
    } finally {
      setSavingSession(false);
    }
  };

  // 8. End Session
  const handleCompleteSession = async () => {
    if (!sessionId) return;
    setSavingSession(true);

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: 'finish' }));
        wsRef.current.close();
      } catch (e) {}
      wsRef.current = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setSessionActive(false);

    const nowSec = Date.now() / 1000;
    const sessionStartSec = sessionStartRef.current || nowSec - 1.0;
    const actualDuration = Math.max(0.1, nowSec - sessionStartSec);

    const totalFrames = totalFramesRef.current;
    const trackedFrames = trackedFramesRef.current;
    const availability = totalFrames > 0 ? (trackedFrames / totalFrames) * 100.0 : 0.0;

    const allExcursions = excursionsRef.current;
    const totalReps = repCount;
    const avgExcursion = allExcursions.length > 0 ? allExcursions.reduce((a, b) => a + b, 0) / allExcursions.length : 0.0;
    const bestExcursion = allExcursions.length > 0 ? Math.max(...allExcursions) : 0.0;
    const maxAngle = maxAnglesRef.current.length > 0 ? Math.max(...maxAnglesRef.current) : 0.0;

    const payload = {
      total_reps: totalReps,
      rejected_reps: 0,
      set_count: 1,
      avg_excursion_deg: Math.round(avgExcursion * 10) / 10,
      best_excursion_deg: Math.round(bestExcursion * 10) / 10,
      tracking_availability_pct: Math.round(availability * 10) / 10,
      duration_sec: Math.round(actualDuration * 10) / 10,
      notes:
        totalReps > 0
          ? `Completed ${totalReps} knee extensions with ${avgExcursion.toFixed(1)}° mean excursion.`
          : 'Session ended without completed repetitions.',
      sets: [
        {
          set_number: 1,
          reps_completed: totalReps,
          reps_rejected: 0,
          rest_duration_sec: 0.0,
          avg_excursion_deg: Math.round(avgExcursion * 10) / 10,
          max_extension_deg: Math.round(maxAngle * 10) / 10,
        },
      ],
    };

    try {
      const report = await completeRehabSession(sessionId, payload);
      setSummaryReport(report);
      setSessionId(null);
      if (onSessionCompleted) onSessionCompleted();
    } catch (err) {
      setError('Failed to persist session: ' + err.message);
    } finally {
      setSavingSession(false);
    }
  };

  return (
    <div>
      <div className="alert-banner warning">
        <span>⚠️ <strong>Clinical Notice:</strong> KneeWise Computer Vision exercise guidance provides 2D biomechanical motion assistance. It is not diagnostic clinical goniometry.</span>
      </div>

      {cameraError && (
        <div className="alert-banner danger">
          <span>🚫 {cameraError}</span>
        </div>
      )}

      {error && (
        <div className="alert-banner danger">
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Configuration & Controls Bar */}
      <div className="card" style={{ padding: '1rem 1.5rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr)) auto', gap: '1rem', alignItems: 'flex-end' }}>
          <div>
            <label className="form-label">Patient</label>
            <select
              className="form-select"
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              disabled={sessionActive}
            >
              <option value="">-- Select Patient --</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.patient_id} - {p.first_name} {p.last_name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="form-label">Tracked Leg</label>
            <select
              className="form-select"
              value={selectedLeg}
              onChange={(e) => setSelectedLeg(e.target.value)}
              disabled={sessionActive}
            >
              <option value="right">Right Knee Extension</option>
              <option value="left">Left Knee Extension</option>
            </select>
          </div>

          <div>
            <label className="form-label">Webcam Source</label>
            <select
              className="form-select"
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
              disabled={sessionActive}
            >
              {devices.length > 0 ? (
                devices.map((d, idx) => (
                  <option key={d.deviceId || idx} value={d.deviceId}>
                    {d.label || `Camera ${idx + 1}`}
                  </option>
                ))
              ) : (
                <option value="">Default System Webcam</option>
              )}
            </select>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {!cameraActive ? (
              <button onClick={startCamera} className="btn btn-secondary" disabled={cameraLoading}>
                {cameraLoading ? 'Connecting...' : '📷 Enable Camera'}
              </button>
            ) : (
              <button onClick={stopCamera} className="btn btn-secondary" disabled={sessionActive}>
                ⏹ Stop Camera
              </button>
            )}

            {!sessionActive ? (
              <button
                onClick={handleStartSession}
                className="btn btn-primary"
                disabled={savingSession || !selectedPatientId || cameraLoading}
              >
                ▶ Start Exercise Tracking
              </button>
            ) : (
              <button onClick={handleCompleteSession} className="btn btn-danger" disabled={savingSession}>
                ⏹ End & Save Session
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Studio Viewport */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(380px, 1.4fr) 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        {/* Live Camera Feed Card */}
        <div className="card" style={{ position: 'relative', overflow: 'hidden' }}>
          <div className="card-header">
            <h3 className="card-title">Live Camera Feed</h3>
            <span
              className={`brand-tag ${trackingStatus === 'TRACKING' ? 'green' : cameraActive ? 'cyan' : ''}`}
              style={{
                background: trackingStatus === 'TRACKING' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                borderColor: trackingStatus === 'TRACKING' ? '#10b981' : '#3b82f6',
                color: trackingStatus === 'TRACKING' ? '#6ee7b7' : '#93c5fd',
              }}
            >
              {trackingStatus === 'TRACKING' ? '🟢 POSE TRACKING ACTIVE' : cameraActive ? `📷 CAMERA READY (${transportMode})` : '⚪ CAMERA OFF'}
            </span>
          </div>

          {/* Viewport for video and overlay canvas */}
          <div
            style={{
              position: 'relative',
              width: '100%',
              aspectRatio: '4/3',
              backgroundColor: '#070a11',
              borderRadius: '8px',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                display: cameraActive ? 'block' : 'none',
              }}
            />

            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                pointerEvents: 'none',
                display: cameraActive ? 'block' : 'none',
              }}
            />

            {!cameraActive && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                <div style={{ fontSize: '3.5rem', marginBottom: '0.75rem' }}>📷</div>
                <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Camera Not Activated</p>
                <p style={{ fontSize: '0.82rem', maxWidth: '340px', margin: '0 auto' }}>
                  Click <strong>"Enable Camera"</strong> or <strong>"Start Exercise Tracking"</strong> to permit browser webcam streaming.
                </p>
              </div>
            )}
          </div>

          {/* Real-time Dynamic Feedback Banner from Python Engine */}
          <div
            className={`alert-banner ${feedbackSeverity === 'success' ? 'success' : feedbackSeverity === 'warning' ? 'warning' : 'info'}`}
            style={{ marginTop: '1rem', marginBottom: 0 }}
          >
            <strong>{exerciseState}:</strong> {feedback}
          </div>
        </div>

        {/* Live Biometrics Display Card */}
        <div className="card">
          <h3 className="card-title">Live Biometrics (Python CV)</h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
            Measurements computed by server-side MediaPipe Pose & vector trigonometry.
          </p>

          <div className="metrics-grid">
            <div className="metric-tile">
              <span className="metric-label">Knee Angle (Raw)</span>
              <span className="metric-value cyan">
                {kneeAngle !== null ? `${Math.round(kneeAngle)}°` : '—'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Vector Biomechanics</span>
            </div>

            <div className="metric-tile">
              <span className="metric-label">Filtered Angle</span>
              <span className="metric-value" style={{ color: '#38bdf8' }}>
                {smoothedAngle !== null ? `${Math.round(smoothedAngle)}°` : '—'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Temporal Smoothing</span>
            </div>

            <div className="metric-tile">
              <span className="metric-label">Completed Reps</span>
              <span className="metric-value green">{repCount}</span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>FSM Validated</span>
            </div>

            <div className="metric-tile">
              <span className="metric-label">Tracking Reliability</span>
              <span className="metric-value" style={{ color: trackingStatus === 'TRACKING' ? '#10b981' : '#f59e0b' }}>
                {trackingStatus === 'TRACKING' ? `${Math.round(landmarkConfidence * 100)}%` : 'LOST'}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Joint Visibility Mean</span>
            </div>
          </div>

          {/* State & Tracked Limb Progress */}
          <div style={{ marginTop: '1.5rem', background: 'var(--bg-secondary)', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Exercise Phase:</span>
              <strong style={{ color: exerciseState === 'EXTENDED' ? '#10b981' : exerciseState === 'EXTENDING' ? '#38bdf8' : 'var(--text-primary)' }}>
                {exerciseState}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Tracked Limb:</span>
              <strong>{selectedLeg.toUpperCase()} LEG</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Target Full Extension:</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>≥ 155°</span>
            </div>
          </div>

          {/* Technical Diagnostics */}
          <div style={{ marginTop: '1.25rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <p>Engine: <code>rehab_cv (MediaPipe Pose 0.10)</code></p>
            <p>Transport: <code>{transportMode} streaming ({totalFramesRef.current} frames processed)</code></p>
          </div>
        </div>
      </div>

      {/* Persistent Database Session Summary */}
      {summaryReport && (
        <div className="card" style={{ borderColor: 'var(--success-green)', marginTop: '1.5rem' }}>
          <div className="card-header">
            <div>
              <h2 className="card-title" style={{ color: 'var(--success-green)' }}>
                ✅ Rehabilitation Session Saved to Database
              </h2>
              <span style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                Session ID: <code>{summaryReport.id}</code> | Patient: <strong>{summaryReport.patient_mrn}</strong>
              </span>
            </div>
            <span className="brand-tag green">COMPLETED</span>
          </div>

          <div className="metrics-grid">
            <div className="metric-tile">
              <span className="metric-label">Total Repetitions</span>
              <span className="metric-value green">{summaryReport.total_reps}</span>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Mean ROM Excursion</span>
              <span className="metric-value cyan">{summaryReport.avg_excursion_deg}°</span>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Best Extension</span>
              <span className="metric-value">{summaryReport.best_excursion_deg}°</span>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Session Duration</span>
              <span className="metric-value">{summaryReport.duration_sec}s</span>
            </div>
            <div className="metric-tile">
              <span className="metric-label">Tracking Availability</span>
              <span className="metric-value">{summaryReport.tracking_availability_pct}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
