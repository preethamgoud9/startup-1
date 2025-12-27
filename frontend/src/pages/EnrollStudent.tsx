import React, { useState } from 'react';
import { Camera, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { startEnrollment, captureImage } from '../services/api';
import type { EnrollmentSessionResponse } from '../services/api';

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
      setError('Please fill in all fields');
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      const sessionData = await startEnrollment(studentId, name, className);
      setSession(sessionData);
      await startWebcam();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start enrollment');
    }
  };

  const handleCapture = async () => {
    if (!session || !isActive) return;

    const frame = captureFrame();
    if (!frame) {
      setError('Failed to capture frame');
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
        setSuccess('Enrollment completed successfully!');
        stopWebcam();
        setSession(null);
        setStudentId('');
        setName('');
        setClassName('');
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to capture image');
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
    <div style={styles.container}>
      <h1 style={styles.title}>Enroll New Student</h1>

      {!session ? (
        <div style={styles.formContainer}>
          <div style={styles.form}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Student ID</label>
              <input
                type="text"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                style={styles.input}
                placeholder="Enter student ID"
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Full Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                style={styles.input}
                placeholder="Enter student name"
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Class</label>
              <input
                type="text"
                value={className}
                onChange={(e) => setClassName(e.target.value)}
                style={styles.input}
                placeholder="Enter class (e.g., 10A)"
              />
            </div>

            {error && (
              <div style={styles.error}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {success && (
              <div style={styles.success}>
                <CheckCircle size={20} />
                <span>{success}</span>
              </div>
            )}

            <button style={styles.startButton} onClick={handleStartEnrollment}>
              Start Enrollment
            </button>
          </div>
        </div>
      ) : (
        <div style={styles.captureContainer}>
          <div style={styles.videoSection}>
            <div style={styles.videoWrapper}>
              {webcamError && (
                <div style={styles.error}>
                  <AlertCircle size={20} />
                  <span>{webcamError}</span>
                </div>
              )}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={styles.video}
              />
            </div>

            <div style={styles.poseInstruction}>
              {session.next_pose && (
                <>
                  <h3 style={styles.poseTitle}>Next Pose:</h3>
                  <p style={styles.poseText}>{session.next_pose.toUpperCase()}</p>
                </>
              )}
            </div>
          </div>

          <div style={styles.sidebar}>
            <div style={styles.progressSection}>
              <h3 style={styles.sidebarTitle}>Progress</h3>
              <div style={styles.progressBar}>
                <div style={{ ...styles.progressFill, width: `${progress}%` }} />
              </div>
              <p style={styles.progressText}>
                {session.current_count} / {session.required_images} images captured
              </p>
            </div>

            <div style={styles.studentInfo}>
              <h3 style={styles.sidebarTitle}>Student Information</h3>
              <p><strong>ID:</strong> {session.student_id}</p>
              <p><strong>Name:</strong> {name}</p>
              <p><strong>Class:</strong> {className}</p>
            </div>

            {error && (
              <div style={styles.error}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            <div style={styles.controls}>
              <button
                style={styles.captureButton}
                onClick={handleCapture}
                disabled={isCapturing}
              >
                {isCapturing ? (
                  <>
                    <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Capturing...</span>
                  </>
                ) : (
                  <>
                    <Camera size={20} />
                    <span>Capture Image</span>
                  </>
                )}
              </button>

              <button style={styles.cancelButton} onClick={handleCancel}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '2rem 2.5rem',
    width: '100%',
    minHeight: 'calc(100vh - 80px)',
  },
  title: {
    fontSize: '2.25rem',
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: '2.5rem',
    letterSpacing: '-0.025em',
  },
  formContainer: {
    display: 'flex',
    justifyContent: 'center',
    padding: '2rem 0',
  },
  form: {
    backgroundColor: 'white',
    padding: '2.5rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    width: '100%',
    maxWidth: '550px',
    border: '1px solid #e2e8f0',
  },
  formGroup: {
    marginBottom: '1.75rem',
  },
  label: {
    display: 'block',
    marginBottom: '0.625rem',
    fontWeight: '600',
    color: '#0f172a',
    fontSize: '0.9375rem',
  },
  input: {
    width: '100%',
    padding: '0.875rem 1rem',
    border: '2px solid #e2e8f0',
    borderRadius: '0.75rem',
    fontSize: '1rem',
    boxSizing: 'border-box',
    transition: 'all 0.2s',
  },
  startButton: {
    width: '100%',
    padding: '1rem',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 4px 6px -1px rgba(59, 130, 246, 0.3)',
    transition: 'all 0.2s',
  },
  captureContainer: {
    display: 'grid',
    gridTemplateColumns: '1fr 400px',
    gap: '2rem',
    alignItems: 'start',
    width: '100%',
  },
  videoSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  videoWrapper: {
    backgroundColor: '#000',
    borderRadius: '1rem',
    overflow: 'hidden',
    aspectRatio: '4/3',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    border: '2px solid #1e293b',
  },
  video: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  poseInstruction: {
    background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
    color: 'white',
    padding: '2rem',
    borderRadius: '1rem',
    textAlign: 'center',
    boxShadow: '0 10px 15px -3px rgba(59, 130, 246, 0.3)',
  },
  poseTitle: {
    fontSize: '1rem',
    marginBottom: '0.75rem',
    opacity: 0.9,
    fontWeight: '500',
  },
  poseText: {
    fontSize: '2.5rem',
    fontWeight: '700',
    margin: 0,
    letterSpacing: '-0.025em',
  },
  sidebar: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  progressSection: {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    border: '1px solid #e2e8f0',
  },
  sidebarTitle: {
    fontSize: '1.25rem',
    fontWeight: '700',
    marginBottom: '1.25rem',
    color: '#0f172a',
  },
  progressBar: {
    width: '100%',
    height: '1.25rem',
    backgroundColor: '#e2e8f0',
    borderRadius: '0.75rem',
    overflow: 'hidden',
    marginBottom: '0.75rem',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
    transition: 'width 0.3s ease',
    boxShadow: 'inset 0 2px 4px rgba(255, 255, 255, 0.3)',
  },
  progressText: {
    fontSize: '0.9375rem',
    color: '#64748b',
    textAlign: 'center',
    fontWeight: '500',
  },
  studentInfo: {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    border: '1px solid #e2e8f0',
  },
  controls: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  captureButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.625rem',
    padding: '1rem',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.3)',
    transition: 'all 0.2s',
  },
  cancelButton: {
    padding: '1rem',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 4px 6px -1px rgba(239, 68, 68, 0.3)',
    transition: 'all 0.2s',
  },
  error: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    padding: '1rem 1.25rem',
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    borderRadius: '0.75rem',
    marginBottom: '1.5rem',
    border: '1px solid #fecaca',
  },
  success: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    padding: '1rem 1.25rem',
    backgroundColor: '#d1fae5',
    color: '#065f46',
    borderRadius: '0.75rem',
    marginBottom: '1.5rem',
    border: '1px solid #a7f3d0',
  },
};
