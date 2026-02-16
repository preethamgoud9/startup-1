import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
}

export interface EnrollmentSessionResponse {
  session_id: string;
  student_id: string;
  required_images: number;
  current_count: number;
  next_pose: string | null;
}

export interface CaptureImageResponse {
  session_id: string;
  current_count: number;
  required_images: number;
  next_pose: string | null;
  completed: boolean;
  message: string;
}

export interface RecognitionResult {
  student_id: string | null;
  name?: string;
  class?: string;
  confidence: number;
  is_known: boolean;
  bbox?: number[];
  timestamp: string;
}

export interface RecognitionFrameResponse {
  frame: string | null;
  results: RecognitionResult[];
}

export interface AttendanceRecord {
  student_id: string;
  name: string;
  class: string;
  date: string;
  time: string;
  timestamp: string;
  confidence?: number;
}

export interface AttendanceResponse {
  records: AttendanceRecord[];
  total: number;
  date: string;
}

export interface User {
  username: string;
  is_admin: boolean;
}

export interface Token {
  access_token: string;
  token_type: string;
}

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const startEnrollment = async (
  studentId: string,
  name: string,
  className: string
): Promise<EnrollmentSessionResponse> => {
  const response = await api.post<EnrollmentSessionResponse>('/enroll/start', {
    student_id: studentId,
    name,
    class: className,
  });
  return response.data;
};

export const captureImage = async (
  sessionId: string,
  imageData: string
): Promise<CaptureImageResponse> => {
  const response = await api.post<CaptureImageResponse>('/enroll/capture', {
    session_id: sessionId,
    image_data: imageData,
  });
  return response.data;
};

export const quickEnroll = async (
  studentId: string,
  name: string,
  className: string,
  imageData: string
): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/enroll/quick', {
    student_id: studentId,
    name,
    class: className,
    image_data: imageData,
  });
  return response.data;
};


export const uploadEnroll = async (
  studentId: string,
  name: string,
  className: string,
  files: File[]
): Promise<{ success: boolean; message: string; images_processed: number; images_failed: number }> => {
  const formData = new FormData();
  formData.append('student_id', studentId);
  formData.append('name', name);
  formData.append('class_name', className);
  files.forEach((file) => {
    formData.append('files', file);
  });
  const response = await api.post('/enroll/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const startCamera = async (source?: string): Promise<{ running: boolean; message: string }> => {
  const response = await api.post('/recognition/camera/start', { source });
  return response.data;
};

export const stopCamera = async (): Promise<{ running: boolean; message: string }> => {
  const response = await api.post('/recognition/camera/stop');
  return response.data;
};

export const getLiveRecognition = async (): Promise<RecognitionFrameResponse> => {
  const response = await api.get<RecognitionFrameResponse>('/recognition/live');
  return response.data;
};

export const getTodayAttendance = async (): Promise<AttendanceResponse> => {
  const response = await api.get<AttendanceResponse>('/attendance/today');
  return response.data;
};

export const getAttendanceByDate = async (date: string): Promise<AttendanceResponse> => {
  const response = await api.get<AttendanceResponse>(`/attendance/date/${date}`);
  return response.data;
};

export const markAttendance = async (
  studentId: string,
  name: string,
  className: string,
  confidence: number
): Promise<{ success: boolean; message: string }> => {
  const response = await api.post('/attendance/mark', {
    student_id: studentId,
    name,
    class_name: className,
    confidence,
  });
  return response.data;
};

export const exportAttendance = async (date?: string, classFilter?: string): Promise<Blob> => {
  const response = await api.post(
    '/attendance/export',
    { date, class_filter: classFilter },
    { responseType: 'blob' }
  );
  return response.data;
};

export const login = async (username: string, password: string): Promise<Token> => {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);

  const response = await api.post<Token>('/auth/login', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const getCurrentUser = async (): Promise<User> => {
  const response = await api.get<User>('/auth/me');
  return response.data;
};

// Production recognition types
export interface ProductionRecognitionResult {
  student_id: string | null;
  name?: string;
  class?: string;
  confidence: number;
  is_known: boolean;
  bbox?: number[];
  camera_id: number;
  timestamp: string;
}

export interface AttendanceFeedEntry {
  student_id: string;
  name: string;
  class: string;
  confidence: number;
  camera_id: number;
  camera_name: string;
  timestamp: string;
  message: string;
}

// Production recognition API functions
export const getCameraRecognitions = async (cameraId: number): Promise<ProductionRecognitionResult[]> => {
  const response = await api.get(`/production/cameras/${cameraId}/recognitions`);
  return response.data;
};

export const getAttendanceFeed = async (): Promise<AttendanceFeedEntry[]> => {
  const response = await api.get('/production/attendance-feed');
  return response.data;
};

export const addCameraFromSetup = async (data: {
  name: string;
  stream_url: string;
  fps_limit?: number;
  detection_enabled?: boolean;
}): Promise<any> => {
  const response = await api.post('/production/cameras/from-cctv-setup', data);
  return response.data;
};

// Analytics types
export interface AnalyticsOverview {
  total_records: number;
  unique_students: number;
  total_days: number;
  avg_daily_attendance: number;
  avg_confidence: number;
  classes: string[];
  enrolled_count: number;
  enrolled_classes: Record<string, number>;
  days: number;
}

export interface AnalyticsTrends {
  dates: string[];
  counts: number[];
  average: number;
  trend: string;
}

export interface ClassBreakdownItem {
  name: string;
  total_records: number;
  unique_students: number;
  avg_daily: number;
  days_active: number;
}

export interface StudentInsight {
  student_id: string;
  name: string;
  class: string;
  days_present: number;
  total_days: number;
  attendance_rate: number;
  avg_confidence: number;
  last_seen: string;
}

export interface StudentsAnalytics {
  students: StudentInsight[];
  total_enrolled: number;
}

export interface HourlyDistribution {
  distribution: Record<string, number>;
  peak_hour: number;
  peak_count: number;
}

export interface ConfidenceBucket {
  range: string;
  count: number;
}

export interface ConfidenceStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  distribution: ConfidenceBucket[];
}

export const getAnalyticsOverview = async (days: number = 30): Promise<AnalyticsOverview> => {
  const response = await api.get<AnalyticsOverview>(`/analytics/overview?days=${days}`);
  return response.data;
};

export const getAnalyticsTrends = async (days: number = 30): Promise<AnalyticsTrends> => {
  const response = await api.get<AnalyticsTrends>(`/analytics/trends?days=${days}`);
  return response.data;
};

export const getClassBreakdown = async (days: number = 30): Promise<{ classes: ClassBreakdownItem[] }> => {
  const response = await api.get(`/analytics/class-breakdown?days=${days}`);
  return response.data;
};

export const getStudentsAnalytics = async (days: number = 30): Promise<StudentsAnalytics> => {
  const response = await api.get<StudentsAnalytics>(`/analytics/students?days=${days}`);
  return response.data;
};

export const getHourlyDistribution = async (days: number = 30): Promise<HourlyDistribution> => {
  const response = await api.get<HourlyDistribution>(`/analytics/hourly?days=${days}`);
  return response.data;
};

export const getConfidenceStats = async (days: number = 30): Promise<ConfidenceStats> => {
  const response = await api.get<ConfidenceStats>(`/analytics/confidence?days=${days}`);
  return response.data;
};

export default api;
