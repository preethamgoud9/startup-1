import React, { useState } from 'react';
import { Camera, CheckCircle, AlertCircle, Loader, User, BookOpen, Fingerprint, X } from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { startEnrollment, captureImage } from '../services/api';
import type { EnrollmentSessionResponse } from '../services/api';
import './EnrollStudent.css';

export const EnrollStudent: React.FC = () => {
  const { videoRef, isActive, error: webcamError, startWebcam, stopWebcam, captureFrame } = useWebcam();
  const [studentId, setStudentId] = useState('');
  const [name, setName] = useState('');
  const [className, setClassName] = useState('');
  const [session, setSession] = useState<EnrollmentSessionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const handleStartEnrollment = async () => {
    if (!studentId || !name || !className) {
      setError('Required fields are missing. Please provide complete student identity.');
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      const sessionData = await startEnrollment(studentId, name, className);
      setSession(sessionData);
      await startWebcam();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Identity enrollment failed to initialize.');
    }
  };

  const handleCapture = async () => {
    if (!session || !isActive) return;

    const frame = captureFrame();
    if (!frame) {
      setError('Image acquisition failed.');
      return;
    }

    setIsCapturing(true);
    try {
      const response = await captureImage(session.session_id, frame);
      setSession({
        ...session,
        current_count: response.current_count,
        next_pose: response.next_pose,
      });

      if (response.completed) {
        setSuccess('Identity localized. Enrollment success.');
        stopWebcam();
        setSession(null);
        setStudentId('');
        setName('');
        setClassName('');
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Biometric capture failed.');
    } finally {
      setIsCapturing(false);
    }
  };

  const handleCancel = () => {
    stopWebcam();
    setSession(null);
    setError(null);
    setSuccess(null);
  };

  const progress = session
    ? Math.round((session.current_count / session.required_images) * 100)
    : 0;

  return (
    <div className="enroll-container animate-fade-in">
      <h1 className="enroll-title">Student Enrollment</h1>

      {!session ? (
        <div className="enroll-wrapper">
          <div className="form-glass glass">
            <div className="form-group">
              <label className="form-label">
                <Fingerprint size={16} /> Student ID
              </label>
              <input
                type="text"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                className="form-input"
                placeholder="ID-8829"
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                <User size={16} /> Full Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="form-input"
                placeholder="Johnathan Doe"
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                <BookOpen size={16} /> Department / Class
              </label>
              <input
                type="text"
                value={className}
                onChange={(e) => setClassName(e.target.value)}
                className="form-input"
                placeholder="CS-101"
              />
            </div>

            {error && (
              <div className="error-toast" style={{ margin: 0 }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {success && (
              <div className="success-toast" style={{ margin: 0 }}>
                <CheckCircle size={20} />
                <span>{success}</span>
              </div>
            )}

            <button className="btn-primary" onClick={handleStartEnrollment} style={{ width: '100%', justifyContent: 'center' }}>
              Initialize Enrollment
            </button>
          </div>
        </div>
      ) : (
        <div className="capture-layout">
          <div>
            <div className="enroll-video-card">
              {webcamError && (
                <div className="error-toast" style={{ position: 'absolute', top: 20, left: 20, right: 20, zIndex: 10 }}>
                  <AlertCircle size={20} />
                  <span>{webcamError}</span>
                </div>
              )}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="enroll-video"
              />
              <div style={{ position: 'absolute', top: 20, right: 20 }}>
                <div style={{ padding: '8px 16px', background: 'rgba(0,0,0,0.5)', borderRadius: '20px', color: 'white', fontSize: '0.8rem', fontWeight: 600, backdropFilter: 'blur(4px)' }}>
                  Live Identity Acquisition
                </div>
              </div>
            </div>

            <div className="pose-card">
              {session.next_pose && (
                <>
                  <div className="pose-label">Position Request</div>
                  <div className="pose-name">{session.next_pose.toUpperCase()}</div>
                </>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <div className="progress-card glass">
              <div className="progress-header">
                <span className="sidebar-title" style={{ fontSize: '1.1rem' }}>Acquisition Progress</span>
                <span className="info-value">{progress}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-bar" style={{ width: `${progress}%` }} />
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                {session.current_count} of {session.required_images} biometrics secured
              </div>
            </div>

            <div className="info-card glass">
              <h3 className="sidebar-title" style={{ fontSize: '1.1rem', marginBottom: 12 }}>Metadata</h3>
              <div className="info-item">
                <span className="info-label">Identity ID</span>
                <span className="info-value">{session.student_id}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Legal Name</span>
                <span className="info-value">{name}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Assigned Class</span>
                <span className="info-value">{className}</span>
              </div>
            </div>

            {error && (
              <div className="error-toast" style={{ margin: 0 }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <button
                className="btn-primary"
                onClick={handleCapture}
                disabled={isCapturing}
                style={{ width: '100%', justifyContent: 'center' }}
              >
                {isCapturing ? (
                  <>
                    <Loader size={20} className="animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Camera size={20} fill="currentColor" />
                    <span>Secure Biometric</span>
                  </>
                )}
              </button>

              <button
                onClick={handleCancel}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                  borderRadius: '12px',
                  color: '#ef4444',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                <X size={18} /> Abort Session
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
