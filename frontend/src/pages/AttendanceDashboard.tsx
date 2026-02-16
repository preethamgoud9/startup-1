import React, { useEffect, useState, useMemo } from 'react';
import {
  BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import {
  Download, RefreshCw, Calendar, Users, FileText, CheckCircle,
  TrendingUp, Clock, Shield, UserPlus, Search,
  BarChart3, Activity, Eye,
} from 'lucide-react';
import {
  getTodayAttendance, getAttendanceByDate, exportAttendance,
  getAnalyticsOverview, getAnalyticsTrends, getClassBreakdown,
  getStudentsAnalytics, getHourlyDistribution, getConfidenceStats,
} from '../services/api';
import type {
  AttendanceRecord, AnalyticsOverview, AnalyticsTrends,
  ClassBreakdownItem, StudentsAnalytics, HourlyDistribution, ConfidenceStats,
} from '../services/api';
import './AttendanceDashboard.css';

type Tab = 'overview' | 'records' | 'students';
type DaysRange = 7 | 30 | 90 | 365;

const CHART_COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#7c3aed'];
const BAR_COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8'];

const tooltipStyle = {
  background: 'rgba(255,255,255,0.95)',
  border: '1px solid rgba(99,102,241,0.2)',
  borderRadius: '12px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
  fontSize: '0.85rem',
};

export const AttendanceDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [days, setDays] = useState<DaysRange>(30);

  // Records tab state
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Analytics state
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null);
  const [trends, setTrends] = useState<AnalyticsTrends | null>(null);
  const [classData, setClassData] = useState<ClassBreakdownItem[]>([]);
  const [hourly, setHourly] = useState<HourlyDistribution | null>(null);
  const [confidence, setConfidence] = useState<ConfidenceStats | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  // Students tab state
  const [studentsData, setStudentsData] = useState<StudentsAnalytics | null>(null);
  const [studentSearch, setStudentSearch] = useState('');
  const [studentSort, setStudentSort] = useState<'rate_desc' | 'rate_asc' | 'name'>('rate_desc');
  const [studentsLoading, setStudentsLoading] = useState(false);

  // Load analytics when overview tab is active or days changes
  useEffect(() => {
    if (activeTab === 'overview') loadAnalytics();
  }, [activeTab, days]);

  // Load records when records tab is active
  useEffect(() => {
    if (activeTab === 'records') loadAttendance(selectedDate);
  }, [activeTab, selectedDate]);

  // Load students when students tab is active or days changes
  useEffect(() => {
    if (activeTab === 'students') loadStudents();
  }, [activeTab, days]);

  const loadAnalytics = async () => {
    setAnalyticsLoading(true);
    try {
      const [ov, tr, cl, hr, co] = await Promise.all([
        getAnalyticsOverview(days),
        getAnalyticsTrends(days),
        getClassBreakdown(days),
        getHourlyDistribution(days),
        getConfidenceStats(days),
      ]);
      setOverview(ov);
      setTrends(tr);
      setClassData(cl.classes);
      setHourly(hr);
      setConfidence(co);
    } catch (err: any) {
      console.error('Failed to load analytics:', err);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const loadAttendance = async (date?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = date
        ? await getAttendanceByDate(date)
        : await getTodayAttendance();
      setRecords(response.records);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load attendance data.');
    } finally {
      setIsLoading(false);
    }
  };

  const loadStudents = async () => {
    setStudentsLoading(true);
    try {
      const data = await getStudentsAnalytics(days);
      setStudentsData(data);
    } catch (err: any) {
      console.error('Failed to load students:', err);
    } finally {
      setStudentsLoading(false);
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
      setError(err.response?.data?.detail || 'Export failed.');
    }
  };

  // Chart data transformations
  const trendChartData = useMemo(() => {
    if (!trends) return [];
    return trends.dates.map((d, i) => ({
      date: new Date(d).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      count: trends.counts[i],
    }));
  }, [trends]);

  const hourlyChartData = useMemo(() => {
    if (!hourly) return [];
    return Object.entries(hourly.distribution)
      .map(([hour, count]) => ({
        hour: `${hour.padStart(2, '0')}:00`,
        count,
        hourNum: parseInt(hour),
      }))
      .sort((a, b) => a.hourNum - b.hourNum);
  }, [hourly]);

  const filteredStudents = useMemo(() => {
    if (!studentsData) return [];
    const q = studentSearch.toLowerCase();
    return studentsData.students
      .filter(s =>
        s.name.toLowerCase().includes(q) ||
        s.student_id.includes(q) ||
        s.class.toLowerCase().includes(q)
      )
      .sort((a, b) => {
        if (studentSort === 'rate_desc') return b.attendance_rate - a.attendance_rate;
        if (studentSort === 'rate_asc') return a.attendance_rate - b.attendance_rate;
        return a.name.localeCompare(b.name);
      });
  }, [studentsData, studentSearch, studentSort]);

  const avgStudentRate = useMemo(() => {
    if (!studentsData?.students.length) return 0;
    const sum = studentsData.students.reduce((a, s) => a + s.attendance_rate, 0);
    return Math.round(sum / studentsData.students.length);
  }, [studentsData]);

  const getRateColor = (rate: number) => {
    if (rate >= 80) return '#10b981';
    if (rate >= 50) return '#f59e0b';
    return '#ef4444';
  };

  const getTrendBadge = (trend: string) => {
    if (trend === 'increasing') return { text: 'Increasing', color: '#10b981' };
    if (trend === 'decreasing') return { text: 'Decreasing', color: '#ef4444' };
    return { text: 'Stable', color: '#6366f1' };
  };

  return (
    <div className="dashboard-container animate-fade-in">
      {/* Header */}
      <div className="dashboard-header">
        <h1 className="live-title">Attendance Analytics</h1>
        <div className="tab-bar glass">
          <button
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <BarChart3 size={16} />
            <span>Overview</span>
          </button>
          <button
            className={`tab ${activeTab === 'records' ? 'active' : ''}`}
            onClick={() => setActiveTab('records')}
          >
            <FileText size={16} />
            <span>Records</span>
          </button>
          <button
            className={`tab ${activeTab === 'students' ? 'active' : ''}`}
            onClick={() => setActiveTab('students')}
          >
            <Users size={16} />
            <span>Students</span>
          </button>
        </div>
      </div>

      {/* ==================== OVERVIEW TAB ==================== */}
      {activeTab === 'overview' && (
        <>
          {/* Range selector */}
          <div className="analytics-toolbar">
            <div className="range-selector">
              {([7, 30, 90, 365] as DaysRange[]).map(d => (
                <button
                  key={d}
                  className={`range-btn ${days === d ? 'active' : ''}`}
                  onClick={() => setDays(d)}
                >
                  {d === 365 ? 'All' : `${d}d`}
                </button>
              ))}
            </div>
            <button
              className="btn-primary"
              onClick={loadAnalytics}
              disabled={analyticsLoading}
              style={{ padding: '8px 16px' }}
            >
              <RefreshCw size={16} className={analyticsLoading ? 'animate-spin' : ''} />
              <span>Refresh</span>
            </button>
          </div>

          {analyticsLoading && !overview ? (
            <div className="loader-container">
              <RefreshCw size={48} className="animate-spin" opacity={0.2} />
              <p>Loading analytics...</p>
            </div>
          ) : overview ? (
            <>
              {/* Stat Cards */}
              <div className="stats-grid-6">
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}>
                    <Activity size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{overview.total_records}</div>
                    <div className="stat-card-label">Total Records</div>
                  </div>
                </div>
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #10b981, #34d399)' }}>
                    <Users size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{overview.unique_students}</div>
                    <div className="stat-card-label">Active Students</div>
                  </div>
                </div>
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #f59e0b, #fbbf24)' }}>
                    <Calendar size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{overview.total_days}</div>
                    <div className="stat-card-label">Days Tracked</div>
                  </div>
                </div>
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #ec4899, #f472b6)' }}>
                    <TrendingUp size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{overview.avg_daily_attendance}</div>
                    <div className="stat-card-label">Avg Daily</div>
                  </div>
                </div>
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #06b6d4, #22d3ee)' }}>
                    <Shield size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{(overview.avg_confidence * 100).toFixed(0)}%</div>
                    <div className="stat-card-label">Avg Confidence</div>
                  </div>
                </div>
                <div className="stat-card-mini glass">
                  <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #8b5cf6, #a78bfa)' }}>
                    <UserPlus size={20} />
                  </div>
                  <div className="stat-card-body">
                    <div className="stat-card-value">{overview.enrolled_count}</div>
                    <div className="stat-card-label">Enrolled</div>
                  </div>
                </div>
              </div>

              {/* Attendance Trend - Full Width */}
              <div className="chart-card glass">
                <div className="chart-header">
                  <h3><TrendingUp size={20} /> Attendance Trend</h3>
                  {trends && (
                    <span
                      className="trend-badge"
                      style={{ background: getTrendBadge(trends.trend).color + '20', color: getTrendBadge(trends.trend).color }}
                    >
                      {getTrendBadge(trends.trend).text}
                    </span>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={trendChartData}>
                    <defs>
                      <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                    <XAxis
                      dataKey="date"
                      stroke="#94a3b8"
                      fontSize={12}
                      tickLine={false}
                      interval={Math.max(0, Math.floor(trendChartData.length / 8) - 1)}
                    />
                    <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area
                      type="monotone"
                      dataKey="count"
                      stroke="#6366f1"
                      strokeWidth={2.5}
                      fill="url(#trendGradient)"
                      dot={{ fill: '#6366f1', r: 3, strokeWidth: 0 }}
                      activeDot={{ r: 6, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
                {trends && (
                  <div className="chart-footer-row">
                    <span>Average: <strong>{trends.average}</strong> / day</span>
                    <span>Period: <strong>{days === 365 ? 'All time' : `Last ${days} days`}</strong></span>
                  </div>
                )}
              </div>

              {/* Two column charts */}
              <div className="charts-grid-2">
                {/* Class Breakdown */}
                <div className="chart-card glass">
                  <div className="chart-header">
                    <h3><Users size={20} /> Class Breakdown</h3>
                  </div>
                  {classData.length > 0 ? (
                    <>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={classData} barSize={40}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                          <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} />
                          <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Bar dataKey="total_records" radius={[8, 8, 0, 0]} name="Total Records">
                            {classData.map((_, i) => (
                              <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="class-detail-grid">
                        {classData.map((c, i) => (
                          <div key={c.name} className="class-detail-item">
                            <div className="class-dot" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                            <span className="class-detail-name">{c.name}</span>
                            <span className="class-detail-stat">{c.unique_students} students</span>
                            <span className="class-detail-stat">{c.avg_daily}/day</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty-chart">No class data available</div>
                  )}
                </div>

                {/* Peak Hours */}
                <div className="chart-card glass">
                  <div className="chart-header">
                    <h3><Clock size={20} /> Peak Hours</h3>
                    {hourly && hourly.peak_count > 0 && (
                      <span className="trend-badge" style={{ background: '#f59e0b20', color: '#f59e0b' }}>
                        Peak: {String(hourly.peak_hour).padStart(2, '0')}:00
                      </span>
                    )}
                  </div>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={hourlyChartData} barSize={20}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                      <XAxis dataKey="hour" stroke="#94a3b8" fontSize={11} tickLine={false} interval={1} />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Detections">
                        {hourlyChartData.map((entry, i) => (
                          <Cell
                            key={i}
                            fill={entry.hourNum === hourly?.peak_hour ? '#f59e0b' : '#c4b5fd'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  {hourly && hourly.peak_count > 0 && (
                    <div className="chart-footer-row">
                      <span>Peak: <strong>{hourly.peak_count}</strong> detections at <strong>{String(hourly.peak_hour).padStart(2, '0')}:00</strong></span>
                    </div>
                  )}
                </div>
              </div>

              {/* Confidence Distribution - Full Width */}
              <div className="chart-card glass">
                <div className="chart-header">
                  <h3><Shield size={20} /> Recognition Confidence Distribution</h3>
                </div>
                {confidence && confidence.mean > 0 ? (
                  <>
                    <div className="confidence-summary">
                      <div className="conf-stat">
                        <span className="conf-stat-value">{(confidence.min * 100).toFixed(1)}%</span>
                        <span className="conf-stat-label">Min</span>
                      </div>
                      <div className="conf-stat">
                        <span className="conf-stat-value">{(confidence.mean * 100).toFixed(1)}%</span>
                        <span className="conf-stat-label">Mean</span>
                      </div>
                      <div className="conf-stat">
                        <span className="conf-stat-value">{(confidence.median * 100).toFixed(1)}%</span>
                        <span className="conf-stat-label">Median</span>
                      </div>
                      <div className="conf-stat">
                        <span className="conf-stat-value">{(confidence.max * 100).toFixed(1)}%</span>
                        <span className="conf-stat-label">Max</span>
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={confidence.distribution} barSize={60}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                        <XAxis dataKey="range" stroke="#94a3b8" fontSize={12} tickLine={false} />
                        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Bar dataKey="count" radius={[8, 8, 0, 0]} name="Records">
                          {confidence.distribution.map((_, i) => (
                            <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </>
                ) : (
                  <div className="empty-chart">No confidence data available</div>
                )}
              </div>
            </>
          ) : null}
        </>
      )}

      {/* ==================== RECORDS TAB ==================== */}
      {activeTab === 'records' && (
        <>
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

          {error && (
            <div className="error-toast">
              <span>{error}</span>
            </div>
          )}

          <div className="stats-grid">
            <div className="stat-glass-card glass">
              <div className="stat-label">TOTAL AUTHENTICATED</div>
              <div className="stat-value">{records.length}</div>
              <div style={{ position: 'absolute', bottom: 20, right: 20, opacity: 0.2 }}>
                <Users size={48} color="var(--primary)" />
              </div>
            </div>
            <div className="stat-glass-card glass">
              <div className="stat-label">REFERENCE PERIOD</div>
              <div className="stat-value" style={{ fontSize: '1.5rem', marginTop: 10 }}>
                {new Date(selectedDate).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
              </div>
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
        </>
      )}

      {/* ==================== STUDENTS TAB ==================== */}
      {activeTab === 'students' && (
        <>
          <div className="students-toolbar">
            <div className="range-selector">
              {([7, 30, 90, 365] as DaysRange[]).map(d => (
                <button
                  key={d}
                  className={`range-btn ${days === d ? 'active' : ''}`}
                  onClick={() => setDays(d)}
                >
                  {d === 365 ? 'All' : `${d}d`}
                </button>
              ))}
            </div>
            <div className="search-glass glass">
              <Search size={18} color="#94a3b8" />
              <input
                placeholder="Search by name, ID, or class..."
                value={studentSearch}
                onChange={(e) => setStudentSearch(e.target.value)}
              />
            </div>
            <select
              className="sort-select glass"
              value={studentSort}
              onChange={(e) => setStudentSort(e.target.value as any)}
            >
              <option value="rate_desc">Highest Attendance</option>
              <option value="rate_asc">Lowest Attendance</option>
              <option value="name">Name A-Z</option>
            </select>
          </div>

          {/* Student summary cards */}
          <div className="stats-grid-3">
            <div className="stat-card-mini glass">
              <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}>
                <UserPlus size={20} />
              </div>
              <div className="stat-card-body">
                <div className="stat-card-value">{studentsData?.total_enrolled || 0}</div>
                <div className="stat-card-label">Total Enrolled</div>
              </div>
            </div>
            <div className="stat-card-mini glass">
              <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #10b981, #34d399)' }}>
                <TrendingUp size={20} />
              </div>
              <div className="stat-card-body">
                <div className="stat-card-value">{avgStudentRate}%</div>
                <div className="stat-card-label">Avg Attendance Rate</div>
              </div>
            </div>
            <div className="stat-card-mini glass">
              <div className="stat-card-icon" style={{ background: 'linear-gradient(135deg, #f59e0b, #fbbf24)' }}>
                <Eye size={20} />
              </div>
              <div className="stat-card-body">
                <div className="stat-card-value">{filteredStudents.length}</div>
                <div className="stat-card-label">Showing</div>
              </div>
            </div>
          </div>

          <div className="table-glass-wrapper glass">
            {studentsLoading ? (
              <div className="loader-container">
                <RefreshCw size={48} className="animate-spin" opacity={0.2} />
                <p>Loading student data...</p>
              </div>
            ) : filteredStudents.length === 0 ? (
              <div className="loader-container">
                <Users size={48} opacity={0.2} />
                <p>No students found.</p>
              </div>
            ) : (
              <table className="attendance-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Class</th>
                    <th>Attendance</th>
                    <th>Rate</th>
                    <th>Avg Confidence</th>
                    <th>Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStudents.map((student) => (
                    <tr key={student.student_id}>
                      <td style={{ fontWeight: 600 }}>{student.student_id}</td>
                      <td style={{ fontWeight: 500 }}>{student.name}</td>
                      <td>{student.class}</td>
                      <td style={{ color: 'var(--text-muted)' }}>
                        {student.days_present} / {student.total_days} days
                      </td>
                      <td>
                        <div className="rate-cell">
                          <div className="rate-bar-bg">
                            <div
                              className="rate-bar-fill"
                              style={{
                                width: `${Math.min(student.attendance_rate, 100)}%`,
                                background: getRateColor(student.attendance_rate),
                              }}
                            />
                          </div>
                          <span className="rate-text" style={{ color: getRateColor(student.attendance_rate) }}>
                            {student.attendance_rate.toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      <td>
                        <span className="confidence-indicator">
                          {student.avg_confidence > 0
                            ? `${(student.avg_confidence * 100).toFixed(1)}%`
                            : 'N/A'}
                        </span>
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                        {student.last_seen === 'Never'
                          ? 'Never'
                          : new Date(student.last_seen).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
};
