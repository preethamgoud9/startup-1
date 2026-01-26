import React, { useEffect, useState } from 'react';
import { Download, RefreshCw, Calendar, Users, FileText, CheckCircle } from 'lucide-react';
import { getTodayAttendance, getAttendanceByDate, exportAttendance } from '../services/api';
import type { AttendanceRecord } from '../services/api';
import './AttendanceDashboard.css';

export const AttendanceDashboard: React.FC = () => {
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAttendance = async (date?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = date
        ? await getAttendanceByDate(date)
        : await getTodayAttendance();
      setRecords(response.records);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Identity ledger failed to load.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const blob = await exportAttendance(selectedDate);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `attendance_report_${selectedDate}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Report synthesis failed.');
    }
  };

  useEffect(() => {
    loadAttendance(selectedDate);
  }, [selectedDate]);

  return (
    <div className="dashboard-container animate-fade-in">
      <div className="dashboard-header">
        <h1 className="live-title">Attendance Analytics</h1>
        <div className="dashboard-controls">
          <div className="date-picker-glass">
            <Calendar size={18} color="var(--primary)" />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="date-input-hidden"
            />
          </div>
          <button className="btn-primary" onClick={() => loadAttendance(selectedDate)} style={{ padding: '10px 16px' }}>
            <RefreshCw size={18} className={isLoading ? "animate-spin" : ""} />
            <span>Sync</span>
          </button>
          <button
            className="btn-primary"
            onClick={handleExport}
            disabled={records.length === 0}
            style={{ padding: '10px 16px', background: 'linear-gradient(135deg, var(--accent), var(--primary))' }}
          >
            <Download size={18} />
            <span>Export</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="error-toast">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="stats-grid">
        <div className="stat-glass-card glass">
          <div className="stat-label">Total Authenticated</div>
          <div className="stat-value">{records.length}</div>
          <div style={{ position: 'absolute', bottom: 20, right: 20, opacity: 0.2 }}>
            <Users size={48} color="var(--primary)" />
          </div>
        </div>
        <div className="stat-glass-card glass">
          <div className="stat-label">Reference Period</div>
          <div className="stat-value" style={{ fontSize: '1.5rem', marginTop: 10 }}>{new Date(selectedDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</div>
          <div style={{ position: 'absolute', bottom: 20, right: 20, opacity: 0.2 }}>
            <FileText size={48} color="var(--primary)" />
          </div>
        </div>
      </div>

      <div className="table-glass-wrapper glass">
        {isLoading ? (
          <div className="loader-container">
            <RefreshCw size={48} className="animate-spin" opacity={0.2} />
            <p>Accessing identity ledger...</p>
          </div>
        ) : records.length === 0 ? (
          <div className="loader-container">
            <CheckCircle size={48} opacity={0.2} />
            <p>No identities localized for this period.</p>
          </div>
        ) : (
          <table className="attendance-table">
            <thead>
              <tr>
                <th>Identity ID</th>
                <th>Full Name</th>
                <th>Classification</th>
                <th>Localized Time</th>
                <th>Reliability</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{record.student_id}</td>
                  <td>{record.name}</td>
                  <td>{record.class}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{record.time}</td>
                  <td>
                    <span className="confidence-indicator">
                      {record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

const AlertCircle: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);
