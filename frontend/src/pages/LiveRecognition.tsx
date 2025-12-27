import React, { useEffect, useState, useRef } from 'react';
import { Play, Square, AlertCircle } from 'lucide-react';
import { startCamera, stopCamera, getLiveRecognition, markAttendance } from '../services/api';
import type { RecognitionFrameResponse } from '../services/api';

export const LiveRecognition: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [frameData, setFrameData] = useState<RecognitionFrameResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [markedStudents, setMarkedStudents] = useState<Set<string>>(new Set());
  const intervalRef = useRef<number | null>(null);

  const handleStart = async () => {
    try {
      setError(null);
      await startCamera();
      setIsRunning(true);
      startPolling();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start camera');
    }
  };

  const handleStop = async () => {
    try {
      await stopCamera();
      setIsRunning(false);
      stopPolling();
      setFrameData(null);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to stop camera');
    }
  };

  const startPolling = () => {
    if (intervalRef.current) return;

    intervalRef.current = window.setInterval(async () => {
      try {
        const data = await getLiveRecognition();
        setFrameData(data);

        for (const result of data.results) {
          if (result.is_known && result.student_id && !markedStudents.has(result.student_id)) {
            await markAttendance(
              result.student_id,
              result.name || result.student_id,
              result.class || 'Unknown',
              result.confidence
            );
            setMarkedStudents((prev) => new Set(prev).add(result.student_id!));
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 500);
  };

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Live Face Recognition</h1>
        <div style={styles.controls}>
          {!isRunning ? (
            <button style={styles.startButton} onClick={handleStart}>
              <Play size={20} />
              <span>Start Camera</span>
            </button>
          ) : (
            <button style={styles.stopButton} onClick={handleStop}>
              <Square size={20} />
              <span>Stop Camera</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div style={styles.error}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div style={styles.content}>
        <div style={styles.videoContainer}>
          {frameData?.frame ? (
            <img src={frameData.frame} alt="Live feed" style={styles.video} />
          ) : (
            <div style={styles.placeholder}>
              <Camera size={64} style={{ opacity: 0.3 }} />
              <p>Camera feed will appear here</p>
            </div>
          )}
        </div>

        <div style={styles.sidebar}>
          <h2 style={styles.sidebarTitle}>Detected Faces</h2>
          <div style={styles.resultsList}>
            {frameData?.results && frameData.results.length > 0 ? (
              frameData.results.map((result, idx) => (
                <div
                  key={idx}
                  style={{
                    ...styles.resultCard,
                    borderLeft: result.is_known ? '4px solid #10b981' : '4px solid #ef4444',
                  }}
                >
                  <div style={styles.resultHeader}>
                    <span style={styles.resultName}>
                      {result.name || result.student_id || 'Unknown'}
                    </span>
                    <span style={styles.resultConfidence}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {result.class && <div style={styles.resultClass}>Class: {result.class}</div>}
                  <div style={styles.resultStatus}>
                    {result.is_known ? (
                      <span style={{ color: '#10b981' }}>✓ Recognized</span>
                    ) : (
                      <span style={{ color: '#ef4444' }}>✗ Unknown</span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <p style={styles.noResults}>No faces detected</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const Camera: React.FC<{ size: number; style?: React.CSSProperties }> = ({ size, style }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" style={style}>
    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
    <circle cx="12" cy="13" r="4" />
  </svg>
);

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '2rem 2.5rem',
    width: '100%',
    minHeight: 'calc(100vh - 80px)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '2.5rem',
  },
  title: {
    fontSize: '2.25rem',
    fontWeight: '700',
    color: '#0f172a',
    letterSpacing: '-0.025em',
  },
  controls: {
    display: 'flex',
    gap: '1rem',
  },
  startButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.875rem 2rem',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.3)',
    transition: 'all 0.2s',
  },
  stopButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.875rem 2rem',
    backgroundColor: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
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
  content: {
    display: 'grid',
    gridTemplateColumns: '1fr 400px',
    gap: '2rem',
    alignItems: 'start',
    width: '100%',
  },
  videoContainer: {
    backgroundColor: '#000',
    borderRadius: '1rem',
    overflow: 'hidden',
    aspectRatio: '16/9',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    border: '2px solid #1e293b',
  },
  video: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#64748b',
    gap: '1.5rem',
  },
  sidebar: {
    backgroundColor: 'white',
    borderRadius: '1rem',
    padding: '2rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    border: '1px solid #e2e8f0',
    minHeight: '400px',
  },
  sidebarTitle: {
    fontSize: '1.5rem',
    fontWeight: '700',
    marginBottom: '1.5rem',
    color: '#0f172a',
    borderBottom: '2px solid #e2e8f0',
    paddingBottom: '0.75rem',
  },
  resultsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },
  resultCard: {
    backgroundColor: '#f8fafc',
    padding: '1.25rem',
    borderRadius: '0.75rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
    transition: 'all 0.2s',
    border: '1px solid #e2e8f0',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.75rem',
  },
  resultName: {
    fontWeight: '700',
    fontSize: '1.125rem',
    color: '#0f172a',
  },
  resultConfidence: {
    fontSize: '0.875rem',
    fontWeight: '600',
    color: '#64748b',
    backgroundColor: '#e2e8f0',
    padding: '0.25rem 0.75rem',
    borderRadius: '0.5rem',
  },
  resultClass: {
    fontSize: '0.875rem',
    color: '#64748b',
    marginBottom: '0.75rem',
    fontWeight: '500',
  },
  resultStatus: {
    fontSize: '0.875rem',
    fontWeight: '600',
    padding: '0.5rem 0.75rem',
    borderRadius: '0.5rem',
    display: 'inline-block',
  },
  noResults: {
    textAlign: 'center',
    color: '#94a3b8',
    padding: '3rem 1rem',
    fontSize: '1rem',
  },
};
