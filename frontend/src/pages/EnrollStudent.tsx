import React, { useState, useRef } from 'react';
import { Camera, CheckCircle, AlertCircle, Loader, User, BookOpen, Fingerprint, X, Zap, Upload, Image } from 'lucide-react';
import { useWebcam } from '../hooks/useWebcam';
import { startEnrollment, captureImage, quickEnroll, uploadEnroll } from '../services/api';
import type { EnrollmentSessionResponse } from '../services/api';
import './EnrollStudent.css';

export const EnrollStudent: React.FC = () => {
  const { videoRef, isActive, error: webcamError, startWebcam, stopWebcam, captureFrame } = useWebcam();
  const [studentId, setStudentId] = useState('');
  const [name, setName] = useState('');
  const [className, setClassName] = useState('');
  const [enrollmentMode, setEnrollmentMode] = useState<'full' | 'quick' | 'upload'>('full');
  const [session, setSession] = useState<EnrollmentSessionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  // Upload mode state
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadPreviews, setUploadPreviews] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length === 0) return;

    const total = uploadFiles.length + selected.length;
    if (total > 15) {
      setError('Maximum 15 images allowed.');
      return;
    }

    const validFiles = selected.filter(f =>
      f.type === 'image/jpeg' || f.type === 'image/png' || f.type === 'image/webp'
    );

    if (validFiles.length !== selected.length) {
      setError('Only JPEG, PNG, and WebP images are accepted.');
    }

    if (validFiles.length === 0) return;

    setError(null);
    const newFiles = [...uploadFiles, ...validFiles];
    setUploadFiles(newFiles);

    // Generate previews
    const newPreviews = [...uploadPreviews];
    validFiles.forEach(file => {
      const url = URL.createObjectURL(file);
      newPreviews.push(url);
    });
    setUploadPreviews(newPreviews);

    // Reset input so same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeFile = (index: number) => {
    const newFiles = uploadFiles.filter((_, i) => i !== index);
    URL.revokeObjectURL(uploadPreviews[index]);
    const newPreviews = uploadPreviews.filter((_, i) => i !== index);
    setUploadFiles(newFiles);
    setUploadPreviews(newPreviews);
  };

  const handleUploadEnroll = async () => {
    if (!studentId || !name || !className) {
      setError('Required fields are missing. Please provide complete student identity.');
      return;
    }
    if (uploadFiles.length === 0) {
      setError('Please select at least one image to upload.');
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      setIsCapturing(true);

      const response = await uploadEnroll(studentId, name, className, uploadFiles);

      setSuccess(response.message);

      // Cleanup
      uploadPreviews.forEach(url => URL.revokeObjectURL(url));
      setUploadFiles([]);
      setUploadPreviews([]);
      setStudentId('');
      setName('');
      setClassName('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload enrollment failed.');
    } finally {
      setIsCapturing(false);
    }
  };

  const handleQuickEnroll = async () => {
    if (!studentId || !name || !className) {
      setError('Required fields are missing. Please provide complete student identity.');
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      setIsCapturing(true);

      await startWebcam();

      await new Promise(resolve => setTimeout(resolve, 1500));

      let frame = null;
      let attempts = 0;
      const maxAttempts = 3;

      while (!frame && attempts < maxAttempts) {
        frame = captureFrame();
        if (!frame) {
          await new Promise(resolve => setTimeout(resolve, 500));
          attempts++;
        }
      }

      if (!frame) {
        setError('Image acquisition failed. Please ensure camera is accessible and try again.');
        stopWebcam();
        setIsCapturing(false);
        return;
      }

      const response = await quickEnroll(studentId, name, className, frame);

      setSuccess(response.message);
      stopWebcam();

      setStudentId('');
      setName('');
      setClassName('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Quick enrollment failed.');
    } finally {
      setIsCapturing(false);
      stopWebcam();
    }
  };

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

  const modeButtonStyle = (mode: string) => ({
    flex: 1,
    padding: '10px 14px',
    borderRadius: '10px',
    border: 'none',
    background: enrollmentMode === mode ? 'white' : 'transparent',
    color: enrollmentMode === mode ? 'var(--primary)' : 'var(--text-muted)',
    fontWeight: 600 as const,
    fontSize: '0.85rem',
    cursor: 'pointer' as const,
    transition: 'all 0.3s ease',
    boxShadow: enrollmentMode === mode ? '0 4px 12px rgba(0, 0, 0, 0.1)' : 'none',
    display: 'flex' as const,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    gap: '6px',
  });

  return (
    <div className="enroll-container animate-fade-in">
      <h1 className="enroll-title">Student Enrollment</h1>

      {!session ? (
        <div className="enroll-wrapper">
          <div className="form-glass glass">
            {/* Mode Toggle - 3 modes */}
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', gap: '8px', padding: '6px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '12px', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
                <button onClick={() => setEnrollmentMode('full')} style={modeButtonStyle('full')}>
                  <Camera size={15} />
                  Full (15 images)
                </button>
                <button onClick={() => setEnrollmentMode('quick')} style={modeButtonStyle('quick')}>
                  <Zap size={15} />
                  Quick (1 image)
                </button>
                <button onClick={() => setEnrollmentMode('upload')} style={modeButtonStyle('upload')}>
                  <Upload size={15} />
                  Upload Photos
                </button>
              </div>
              {enrollmentMode === 'quick' && (
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(236, 72, 153, 0.1)', border: '1px solid rgba(236, 72, 153, 0.2)', borderRadius: '10px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  <AlertCircle size={14} style={{ display: 'inline', marginRight: '6px', color: 'var(--secondary)' }} />
                  Quick mode uses 1 image. Less accurate than full enrollment.
                </div>
              )}
              {enrollmentMode === 'upload' && (
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(34, 197, 94, 0.1)', border: '1px solid rgba(34, 197, 94, 0.2)', borderRadius: '10px', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  <Image size={14} style={{ display: 'inline', marginRight: '6px', color: '#22c55e' }} />
                  Upload 1-15 high-quality photos from your phone, camera, or laptop for best accuracy.
                </div>
              )}
            </div>

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

            {/* Upload area - only shown in upload mode */}
            {enrollmentMode === 'upload' && (
              <div style={{ marginBottom: '20px' }}>
                <label className="form-label">
                  <Image size={16} /> Photos ({uploadFiles.length}/15)
                </label>

                {/* Drop zone / file picker */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: '2px dashed rgba(99, 102, 241, 0.3)',
                    borderRadius: '12px',
                    padding: '24px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: 'rgba(99, 102, 241, 0.03)',
                    transition: 'all 0.2s ease',
                    marginBottom: uploadFiles.length > 0 ? '16px' : '0',
                  }}
                  onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = 'var(--primary)'; e.currentTarget.style.background = 'rgba(99, 102, 241, 0.08)'; }}
                  onDragLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.3)'; e.currentTarget.style.background = 'rgba(99, 102, 241, 0.03)'; }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.3)';
                    e.currentTarget.style.background = 'rgba(99, 102, 241, 0.03)';
                    const droppedFiles = Array.from(e.dataTransfer.files);
                    const fakeEvent = { target: { files: droppedFiles } } as any;
                    handleFilesSelected(fakeEvent);
                  }}
                >
                  <Upload size={28} style={{ color: 'var(--primary)', marginBottom: '8px' }} />
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Click to browse or drag photos here
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    JPEG, PNG, WebP — up to 15 images
                  </div>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  onChange={handleFilesSelected}
                  style={{ display: 'none' }}
                />

                {/* Image previews */}
                {uploadFiles.length > 0 && (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
                    gap: '8px',
                  }}>
                    {uploadPreviews.map((preview, index) => (
                      <div
                        key={index}
                        style={{
                          position: 'relative',
                          borderRadius: '8px',
                          overflow: 'hidden',
                          aspectRatio: '1',
                          border: '1px solid rgba(0,0,0,0.1)',
                        }}
                      >
                        <img
                          src={preview}
                          alt={`Photo ${index + 1}`}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                        <button
                          onClick={(e) => { e.stopPropagation(); removeFile(index); }}
                          style={{
                            position: 'absolute',
                            top: '4px',
                            right: '4px',
                            width: '22px',
                            height: '22px',
                            borderRadius: '50%',
                            border: 'none',
                            background: 'rgba(239, 68, 68, 0.9)',
                            color: 'white',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: 0,
                          }}
                        >
                          <X size={12} />
                        </button>
                        <div style={{
                          position: 'absolute',
                          bottom: 0,
                          left: 0,
                          right: 0,
                          padding: '2px',
                          background: 'rgba(0,0,0,0.5)',
                          color: 'white',
                          fontSize: '0.65rem',
                          textAlign: 'center',
                        }}>
                          {index + 1}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

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

            <button
              className="btn-primary"
              onClick={
                enrollmentMode === 'upload'
                  ? handleUploadEnroll
                  : enrollmentMode === 'quick'
                  ? handleQuickEnroll
                  : handleStartEnrollment
              }
              disabled={isCapturing}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {isCapturing ? (
                <>
                  <Loader size={20} className="animate-spin" />
                  <span>Processing...</span>
                </>
              ) : enrollmentMode === 'upload' ? (
                <>
                  <Upload size={20} />
                  <span>Upload & Enroll{uploadFiles.length > 0 ? ` (${uploadFiles.length} photos)` : ''}</span>
                </>
              ) : enrollmentMode === 'quick' ? (
                <>
                  <Zap size={20} />
                  <span>Quick Enroll</span>
                </>
              ) : (
                <>
                  <Camera size={20} />
                  <span>Initialize Enrollment</span>
                </>
              )}
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
