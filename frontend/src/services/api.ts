import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

export default api;
