import React, { useEffect, useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Users, Calendar, Clock, Award, Activity } from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const Analytics: React.FC = () => {
  const [weeklyTrends, setWeeklyTrends] = useState<any>(null);
  const [classPerformance, setClassPerformance] = useState<any>(null);
  const [studentRates, setStudentRates] = useState<any[]>([]);
  const [peakHours, setPeakHours] = useState<any>(null);
  const [overallStats, setOverallStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const [trends, classPerf, rates, peak, overall] = await Promise.all([
        axios.get(`${API_BASE}/analytics/weekly-trends`),
        axios.get(`${API_BASE}/analytics/class-performance`),
        axios.get(`${API_BASE}/analytics/student-attendance-rate`),
        axios.get(`${API_BASE}/analytics/peak-hours`),
        axios.get(`${API_BASE}/analytics/overall-statistics`),
      ]);

      setWeeklyTrends(trends.data);
      setClassPerformance(classPerf.data);
      setStudentRates(rates.data);
      setPeakHours(peak.data);
      setOverallStats(overall.data);
    } catch (err: any) {
      console.error('Failed to load analytics:', err);
      setError(err.message || 'Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading analytics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.container}>
        <div style={styles.error}>
          <p>Error: {error}</p>
          <button style={styles.refreshButton} onClick={loadAnalytics}>
            <Activity size={20} />
            <span>Retry</span>
          </button>
        </div>
      </div>
    );
  }

  const weeklyChartData = weeklyTrends?.dates.map((date: string, idx: number) => ({
    date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    attendance: weeklyTrends.counts[idx],
  })) || [];

  const classChartData = Object.entries(classPerformance || {}).map(([className, data]: [string, any]) => ({
    class: className,
    total: data.total_attendance,
    avgPerDay: data.average_per_day,
  }));

  const peakHoursData = Object.entries(peakHours?.distribution || {}).map(([hour, count]) => ({
    hour: `${hour}:00`,
    count,
  }));


  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Analytics Dashboard</h1>
        <button style={styles.refreshButton} onClick={loadAnalytics}>
          <Activity size={20} />
          <span>Refresh Data</span>
        </button>
      </div>

      <div style={styles.statsGrid}>
        <div style={styles.statCard}>
          <div style={styles.statIcon} className="stat-icon-blue">
            <Users size={32} />
          </div>
          <div style={styles.statContent}>
            <div style={styles.statValue}>{overallStats?.total_records || 0}</div>
            <div style={styles.statLabel}>Total Attendance Records</div>
          </div>
        </div>

        <div style={styles.statCard}>
          <div style={styles.statIcon} className="stat-icon-green">
            <Award size={32} />
          </div>
          <div style={styles.statContent}>
            <div style={styles.statValue}>{overallStats?.unique_students || 0}</div>
            <div style={styles.statLabel}>Unique Students</div>
          </div>
        </div>

        <div style={styles.statCard}>
          <div style={styles.statIcon} className="stat-icon-orange">
            <Calendar size={32} />
          </div>
          <div style={styles.statContent}>
            <div style={styles.statValue}>{overallStats?.total_days || 0}</div>
            <div style={styles.statLabel}>Days Tracked</div>
          </div>
        </div>

        <div style={styles.statCard}>
          <div style={styles.statIcon} className="stat-icon-purple">
            <TrendingUp size={32} />
          </div>
          <div style={styles.statContent}>
            <div style={styles.statValue}>{overallStats?.average_daily_attendance || 0}</div>
            <div style={styles.statLabel}>Avg Daily Attendance</div>
          </div>
        </div>
      </div>

      <div style={styles.chartsGrid}>
        <div style={styles.chartCard}>
          <h2 style={styles.chartTitle}>
            <TrendingUp size={24} />
            Weekly Attendance Trend
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={weeklyChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '0.5rem' }} />
              <Legend />
              <Line type="monotone" dataKey="attendance" stroke="#3b82f6" strokeWidth={3} dot={{ fill: '#3b82f6', r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
          <div style={styles.chartFooter}>
            <span>Average: {weeklyTrends?.average || 0}</span>
            <span>Trend: {weeklyTrends?.trend || 'stable'}</span>
          </div>
        </div>

        <div style={styles.chartCard}>
          <h2 style={styles.chartTitle}>
            <Users size={24} />
            Class Performance
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={classChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="class" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '0.5rem' }} />
              <Legend />
              <Bar dataKey="total" fill="#10b981" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={styles.chartCard}>
          <h2 style={styles.chartTitle}>
            <Clock size={24} />
            Peak Attendance Hours
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={peakHoursData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="hour" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '0.5rem' }} />
              <Bar dataKey="count" fill="#f59e0b" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          {peakHours?.peak_hour && (
            <div style={styles.chartFooter}>
              <span>Peak Hour: {peakHours.peak_hour}:00 ({peakHours.peak_count} students)</span>
            </div>
          )}
        </div>

        <div style={styles.chartCard}>
          <h2 style={styles.chartTitle}>
            <Award size={24} />
            Top Students (30 Days)
          </h2>
          <div style={styles.studentList}>
            {studentRates.slice(0, 10).map((student, idx) => (
              <div key={idx} style={styles.studentItem}>
                <div style={styles.studentRank}>{idx + 1}</div>
                <div style={styles.studentInfo}>
                  <div style={styles.studentName}>{student.name}</div>
                  <div style={styles.studentClass}>Class: {student.class}</div>
                </div>
                <div style={styles.studentRate}>
                  <div style={styles.rateValue}>{student.attendance_rate}%</div>
                  <div style={styles.rateDays}>{student.days_present}/{student.total_days} days</div>
                </div>
              </div>
            ))}
          </div>
        </div>
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
  },
  title: {
    fontSize: '2.25rem',
    fontWeight: '700',
    color: '#0f172a',
    letterSpacing: '-0.025em',
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
  loading: {
    padding: '4rem',
    textAlign: 'center',
    fontSize: '1.25rem',
    color: '#64748b',
  },
  error: {
    padding: '4rem',
    textAlign: 'center',
    fontSize: '1.125rem',
    color: '#ef4444',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
    marginBottom: '2.5rem',
  },
  statCard: {
    backgroundColor: 'white',
    padding: '1.5rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    border: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    gap: '1.5rem',
  },
  statIcon: {
    width: '64px',
    height: '64px',
    borderRadius: '1rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
  },
  statContent: {
    flex: 1,
  },
  statValue: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: '0.25rem',
  },
  statLabel: {
    fontSize: '0.875rem',
    color: '#64748b',
    fontWeight: '500',
  },
  chartsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
    gap: '2rem',
  },
  chartCard: {
    backgroundColor: 'white',
    padding: '2rem',
    borderRadius: '1rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    border: '1px solid #e2e8f0',
  },
  chartTitle: {
    fontSize: '1.25rem',
    fontWeight: '700',
    color: '#0f172a',
    marginBottom: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
  },
  chartFooter: {
    marginTop: '1rem',
    padding: '1rem',
    backgroundColor: '#f8fafc',
    borderRadius: '0.5rem',
    display: 'flex',
    justifyContent: 'space-around',
    fontSize: '0.875rem',
    fontWeight: '500',
    color: '#64748b',
  },
  studentList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    maxHeight: '400px',
    overflowY: 'auto',
  },
  studentItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem',
    backgroundColor: '#f8fafc',
    borderRadius: '0.75rem',
    border: '1px solid #e2e8f0',
  },
  studentRank: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '700',
    fontSize: '0.875rem',
  },
  studentInfo: {
    flex: 1,
  },
  studentName: {
    fontWeight: '600',
    color: '#0f172a',
    marginBottom: '0.25rem',
  },
  studentClass: {
    fontSize: '0.75rem',
    color: '#64748b',
  },
  studentRate: {
    textAlign: 'right',
  },
  rateValue: {
    fontSize: '1.25rem',
    fontWeight: '700',
    color: '#10b981',
    marginBottom: '0.25rem',
  },
  rateDays: {
    fontSize: '0.75rem',
    color: '#64748b',
  },
};
