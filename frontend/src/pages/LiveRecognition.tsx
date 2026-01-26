import React, { useEffect, useState, useRef } from 'react';
import { Play, Square, AlertCircle, Camera as CameraIcon, Scan, Clock } from 'lucide-react';
import { startCamera, stopCamera, getLiveRecognition, markAttendance } from '../services/api';
import type { RecognitionFrameResponse } from '../services/api';
import './LiveRecognition.css';

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
    <div className="live-container animate-fade-in">
      <div className="live-header">
        <h1 className="live-title">Live Recognition</h1>
        <div className="live-controls">
          {!isRunning ? (
            <button className="btn-primary" onClick={handleStart}>
              <Play size={20} fill="currentColor" />
              <span>Start Intelligence</span>
            </button>
          ) : (
            <button className="btn-primary" onClick={handleStop} style={{ background: 'linear-gradient(135deg, #ef4444, #f43f5e)' }}>
              <Square size={20} fill="currentColor" />
              <span>Terminate Feed</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="error-toast">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="live-content">
        <div className="video-wrapper">
          {frameData?.frame ? (
            <img src={frameData.frame} alt="Live analytical feed" className="video-feed" />
          ) : (
            <div className="video-placeholder">
              <CameraIcon size={64} />
              <p>Initialize vision engine to begin...</p>
            </div>
          )}
          {isRunning && (
            <div style={{ position: 'absolute', top: 20, left: 20, display: 'flex', alignItems: 'center', gap: 8, color: '#10b981' }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px #10b981' }}></div>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' }}>Feed Active</span>
            </div>
          )}
        </div>

        <div className="sidebar-glass glass">
          <h2 className="sidebar-title">
            <Scan size={20} color="var(--primary)" />
            Real-time Detections
          </h2>
          <div className="results-list">
            {frameData?.results && frameData.results.length > 0 ? (
              frameData.results.map((result, idx) => (
                <div
                  key={idx}
                  className={`result-card ${result.is_known ? 'known' : 'unknown'}`}
                >
                  <div className="result-info">
                    <span className="student-name">
                      {result.name || result.student_id || 'Subject Unknown'}
                    </span>
                    <span className="confidence-badge">
                      {(result.confidence * 100).toFixed(1)}% match
                    </span>
                  </div>
                  <div className="student-meta">
                    {result.class && <span>Class: {result.class}</span>}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Clock size={12} />
                      <span>{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </div>
                  <div className={`status-indicator ${result.is_known ? 'status-recognized' : 'status-unknown'}`}>
                    {result.is_known ? 'Authenticated' : 'Unidentified'}
                  </div>
                </div>
              ))
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 0', opacity: 0.5 }}>
                <p>Waiting for identity detection...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
