import React, { useEffect, useState } from 'react';
import { Download, RefreshCw, Calendar } from 'lucide-react';
import { getTodayAttendance, getAttendanceByDate, exportAttendance } from '../services/api';
import type { AttendanceRecord } from '../services/api';

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
      setError(err.response?.data?.detail || 'Failed to load attendance');
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
      a.download = `attendance_${selectedDate}.xlsx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to export attendance');
    }
  };

  useEffect(() => {
    loadAttendance(selectedDate);
  }, [selectedDate]);

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Attendance Dashboard</h1>
        <div style={styles.controls}>
          <div style={styles.dateControl}>
            <Calendar size={20} />
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={styles.dateInput}
            />
          </div>
          <button style={styles.refreshButton} onClick={() => loadAttendance(selectedDate)}>
            <RefreshCw size={20} />
            <span>Refresh</span>
          </button>
          <button style={styles.exportButton} onClick={handleExport} disabled={records.length === 0}>
            <Download size={20} />
            <span>Export Excel</span>
          </button>
        </div>
      </div>

      {error && (
        <div style={styles.error}>
          <span>{error}</span>
        </div>
      )}

      <div style={styles.statsContainer}>
        <div style={styles.statCard}>
          <div style={styles.statValue}>{records.length}</div>
          <div style={styles.statLabel}>Students Present</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statValue}>{selectedDate}</div>
          <div style={styles.statLabel}>Date</div>
        </div>
      </div>

      <div style={styles.tableContainer}>
        {isLoading ? (
          <div style={styles.loading}>Loading...</div>
        ) : records.length === 0 ? (
          <div style={styles.empty}>No attendance records for this date</div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr style={styles.tableHeaderRow}>
                <th style={styles.tableHeader}>Student ID</th>
                <th style={styles.tableHeader}>Name</th>
                <th style={styles.tableHeader}>Class</th>
                <th style={styles.tableHeader}>Time</th>
                <th style={styles.tableHeader}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {records.map((record, idx) => (
                <tr key={idx} style={styles.tableRow}>
                  <td style={styles.tableCell}>{record.student_id}</td>
                  <td style={styles.tableCell}>{record.name}</td>
                  <td style={styles.tableCell}>{record.class}</td>
                  <td style={styles.tableCell}>{record.time}</td>
                  <td style={styles.tableCell}>
                    {record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : 'N/A'}
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
    flexWrap: 'wrap',
    gap: '1.5rem',
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
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  dateControl: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.75rem 1.25rem',
    backgroundColor: 'white',
    border: '2px solid #e2e8f0',
    borderRadius: '0.75rem',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
  },
  dateInput: {
    border: 'none',
    outline: 'none',
    fontSize: '0.9375rem',
    fontWeight: '500',
    color: '#0f172a',
  },
  refreshButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.75rem 1.5rem',
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    cursor: 'pointer',
    fontSize: '0.9375rem',
    fontWeight: '600',
    boxShadow: '0 4px 6px -1px rgba(59, 130, 246, 0.3)',
    transition: 'all 0.2s',
  },
  exportButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.75rem 1.5rem',
    backgroundColor: '#10b981',
    color: 'white',
    border: 'none',
    borderRadius: '0.75rem',
    cursor: 'pointer',
    fontSize: '0.9375rem',
    fontWeight: '600',
    boxShadow: '0 4px 6px -1px rgba(16, 185, 129, 0.3)',
    transition: 'all 0.2s',
  },
  error: {
    padding: '1rem 1.25rem',
    backgroundColor: '#fee2e2',
    color: '#991b1b',
    borderRadius: '0.75rem',
    marginBottom: '1.5rem',
    border: '1px solid #fecaca',
    fontWeight: '500',
  },
  statsContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
    marginBottom: '2.5rem',
  },
  statCard: {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    textAlign: 'center',
    border: '1px solid #e2e8f0',
    transition: 'all 0.2s',
  },
  statValue: {
    fontSize: '2.5rem',
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: '0.75rem',
    letterSpacing: '-0.025em',
  },
  statLabel: {
    fontSize: '0.9375rem',
    color: '#64748b',
    fontWeight: '500',
  },
  tableContainer: {
    backgroundColor: 'white',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    overflow: 'hidden',
    border: '1px solid #e2e8f0',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  tableHeaderRow: {
    background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
  },
  tableHeader: {
    padding: '1.25rem 1.5rem',
    textAlign: 'left',
    fontWeight: '700',
    color: '#0f172a',
    borderBottom: '2px solid #e2e8f0',
    fontSize: '0.9375rem',
  },
  tableRow: {
    borderBottom: '1px solid #f1f5f9',
    transition: 'background-color 0.15s',
  },
  tableCell: {
    padding: '1.25rem 1.5rem',
    color: '#475569',
    fontSize: '0.9375rem',
  },
  loading: {
    padding: '4rem',
    textAlign: 'center',
    color: '#64748b',
    fontSize: '1.125rem',
    fontWeight: '500',
  },
  empty: {
    padding: '4rem',
    textAlign: 'center',
    color: '#94a3b8',
    fontSize: '1.125rem',
  },
};
