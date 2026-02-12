import React, { useState, useEffect, useCallback } from 'react';
import {
    Video,
    Loader2,
    Play,
    Square,
    Plus,
    Trash2,
    Settings,
    Cpu,
    Monitor,
    AlertCircle,
    CheckCircle2,
    RefreshCw,
    Grid3X3,
    Wifi,
    WifiOff,
    Zap,
    Users,
    Clock,
    Camera,
} from 'lucide-react';
import api from '../services/api';
import type { AttendanceFeedEntry, ProductionRecognitionResult } from '../services/api';
import './Production.css';

interface Camera {
    id: number;
    name: string;
    stream_url: string;
    enabled: boolean;
    fps_limit: number;
    resolution_scale: number;
    detection_enabled: boolean;
    recording_enabled: boolean;
    status: string;
    fps: number;
    error_message?: string;
    detection_count: number;
    frame_count: number;
    has_frame: boolean;
    health: 'green' | 'yellow' | 'red';
    reconnect_attempts: number;
}

interface SystemStats {
    running: boolean;
    total_cameras: number;
    connected_cameras: number;
    reconnecting_cameras: number;
    errored_cameras: number;
    total_fps: number;
    total_frames_processed: number;
    total_detections: number;
    processing_mode: string;
    worker_threads: number;
    recent_attendance_count: number;
}

interface GPUInfo {
    cuda_available: boolean;
    cuda_devices: { id: number; name: string }[];
    opencl_available: boolean;
    opencl_devices: { id: number; name: string }[];
    current_mode: string;
    gpu_device_id: number;
    detection_quality: string;
}

interface DetectionQuality {
    id: string;
    name: string;
    description: string;
}

interface ProductionConfig {
    max_cameras: number;
    processing_mode: string;
    gpu_device_id: number;
    worker_threads: number;
    frame_buffer_size: number;
    detection_interval_ms: number;
    enable_recording: boolean;
    recording_path: string;
    enable_alerts: boolean;
    alert_cooldown_seconds: number;
}

interface CameraFrame {
    camera_id: number;
    name: string;
    status: string;
    fps: number;
    health: string;
    reconnect_attempts: number;
    recognitions: ProductionRecognitionResult[];
    frame: string | null;
}

export const Production: React.FC = () => {
    const [cameras, setCameras] = useState<Camera[]>([]);
    const [stats, setStats] = useState<SystemStats | null>(null);
    const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
    const [config, setConfig] = useState<ProductionConfig | null>(null);
    const [cameraFrames, setCameraFrames] = useState<CameraFrame[]>([]);
    const [detectionQualities, setDetectionQualities] = useState<DetectionQuality[]>([]);
    const [attendanceFeed, setAttendanceFeed] = useState<AttendanceFeedEntry[]>([]);

    const [isLoading, setIsLoading] = useState(true);
    const [showAddCamera, setShowAddCamera] = useState(false);
    const [showSettings, setShowSettings] = useState(false);

    // New camera form
    const [newCamera, setNewCamera] = useState({
        name: '',
        stream_url: '',
        fps_limit: 5,
        detection_enabled: true,
    });

    const fetchData = useCallback(async () => {
        try {
            const [camerasRes, statsRes, gpuRes, configRes, qualitiesRes] = await Promise.all([
                api.get('/production/cameras'),
                api.get('/production/status'),
                api.get('/production/gpu'),
                api.get('/production/config'),
                api.get('/production/detection-qualities'),
            ]);
            setCameras(camerasRes.data);
            setStats(statsRes.data);
            setGpuInfo(gpuRes.data);
            setConfig(configRes.data);
            setDetectionQualities(qualitiesRes.data);
        } catch (err) {
            console.error('Failed to fetch production data', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const fetchFrames = useCallback(async () => {
        if (!stats?.running) return;
        try {
            const [framesRes, feedRes] = await Promise.all([
                api.get('/production/grid?columns=4&max_cameras=32'),
                api.get('/production/attendance-feed'),
            ]);
            setCameraFrames(framesRes.data.cameras);
            setAttendanceFeed(feedRes.data);
        } catch (err) {
            console.error('Failed to fetch frames', err);
        }
    }, [stats?.running]);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 3000);
        return () => clearInterval(interval);
    }, [fetchData]);

    useEffect(() => {
        if (stats?.running) {
            // Faster polling for smoother streaming (200ms vs 500ms)
            const frameInterval = setInterval(fetchFrames, 200);
            return () => clearInterval(frameInterval);
        }
    }, [stats?.running, fetchFrames]);

    const handleStartAll = async () => {
        try {
            await api.post('/production/start');
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to start');
        }
    };

    const handleStopAll = async () => {
        try {
            await api.post('/production/stop');
            setCameraFrames([]);
            setAttendanceFeed([]);
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to stop');
        }
    };

    const handleAddCamera = async () => {
        try {
            await api.post('/production/cameras', newCamera);
            setNewCamera({ name: '', stream_url: '', fps_limit: 5, detection_enabled: true });
            setShowAddCamera(false);
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to add camera');
        }
    };

    const handleRemoveCamera = async (cameraId: number) => {
        if (!confirm('Remove this camera?')) return;
        try {
            await api.delete(`/production/cameras/${cameraId}`);
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to remove camera');
        }
    };

    const handleSetProcessingMode = async (mode: string, deviceId: number = 0) => {
        try {
            await api.post('/production/gpu', { mode, device_id: deviceId });
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to set processing mode');
        }
    };

    const handleSetDetectionQuality = async (quality: string) => {
        try {
            await api.post('/production/detection-quality', { quality });
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to set detection quality');
        }
    };

    const handleUpdateConfig = async (updates: Partial<ProductionConfig>) => {
        try {
            await api.post('/production/config', updates);
            fetchData();
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to update config');
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'connected': return <Wifi size={14} />;
            case 'connecting': return <Loader2 size={14} className="animate-spin" />;
            case 'reconnecting': return <RefreshCw size={14} className="animate-spin" />;
            case 'error': return <AlertCircle size={14} />;
            default: return <WifiOff size={14} />;
        }
    };

    const getHealthColor = (health: string) => {
        switch (health) {
            case 'green': return '#22c55e';
            case 'yellow': return '#f59e0b';
            case 'red': return '#ef4444';
            default: return '#6b7280';
        }
    };

    if (isLoading) {
        return (
            <div className="production-container">
                <div className="loader-center">
                    <Loader2 className="animate-spin" size={48} />
                    <p>Loading Production System...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="production-container">
            <div className="production-header">
                <div className="header-left">
                    <h1 className="production-title">
                        <Monitor size={32} />
                        Production Control Center
                    </h1>
                    <p className="production-subtitle">Multi-camera recognition with auto-attendance</p>
                </div>
                <div className="header-actions">
                    <button
                        className="btn-icon"
                        onClick={() => setShowSettings(true)}
                        title="Settings"
                    >
                        <Settings size={20} />
                    </button>
                    <button
                        className="btn-icon"
                        onClick={fetchData}
                        title="Refresh"
                    >
                        <RefreshCw size={20} />
                    </button>
                    {stats?.running ? (
                        <button className="btn-danger" onClick={handleStopAll}>
                            <Square size={18} />
                            Stop All
                        </button>
                    ) : (
                        <button className="btn-success" onClick={handleStartAll}>
                            <Play size={18} />
                            Start All
                        </button>
                    )}
                </div>
            </div>

            {/* Stats Bar */}
            <div className="stats-bar glass">
                <div className="stat-item">
                    <span className="stat-value">{stats?.total_cameras || 0}</span>
                    <span className="stat-label">Total Cameras</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value" style={{ color: '#22c55e' }}>{stats?.connected_cameras || 0}</span>
                    <span className="stat-label">Connected</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value" style={{ color: '#f59e0b' }}>{stats?.reconnecting_cameras || 0}</span>
                    <span className="stat-label">Reconnecting</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value" style={{ color: '#ef4444' }}>{stats?.errored_cameras || 0}</span>
                    <span className="stat-label">Errors</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value">{stats?.total_fps || 0}</span>
                    <span className="stat-label">Total FPS</span>
                </div>
                <div className="stat-item">
                    <span className="stat-value">{stats?.total_detections || 0}</span>
                    <span className="stat-label">Recognitions</span>
                </div>
                <div className="stat-item gpu-status">
                    <Cpu size={16} />
                    <span className="stat-value">{gpuInfo?.current_mode?.toUpperCase() || 'CPU'}</span>
                </div>
            </div>

            {/* GPU Selection */}
            <div className="gpu-section glass">
                <h3><Zap size={20} /> Processing Mode</h3>
                <div className="gpu-options">
                    <button
                        className={`gpu-btn ${gpuInfo?.current_mode === 'cpu' ? 'active' : ''}`}
                        onClick={() => handleSetProcessingMode('cpu')}
                    >
                        <Cpu size={20} />
                        <span>CPU</span>
                    </button>
                    <button
                        className={`gpu-btn ${gpuInfo?.current_mode === 'cuda' ? 'active' : ''} ${!gpuInfo?.cuda_available ? 'disabled' : ''}`}
                        onClick={() => gpuInfo?.cuda_available && handleSetProcessingMode('cuda')}
                        disabled={!gpuInfo?.cuda_available}
                    >
                        <Zap size={20} />
                        <span>CUDA</span>
                        {!gpuInfo?.cuda_available && <small>Not Available</small>}
                    </button>
                    <button
                        className={`gpu-btn ${gpuInfo?.current_mode === 'opencl' ? 'active' : ''} ${!gpuInfo?.opencl_available ? 'disabled' : ''}`}
                        onClick={() => gpuInfo?.opencl_available && handleSetProcessingMode('opencl')}
                        disabled={!gpuInfo?.opencl_available}
                    >
                        <Monitor size={20} />
                        <span>OpenCL</span>
                        {!gpuInfo?.opencl_available && <small>Not Available</small>}
                    </button>
                </div>
            </div>

            {/* Detection Quality */}
            <div className="gpu-section glass">
                <h3><Video size={20} /> Detection Quality (Long-Range Accuracy)</h3>
                <div className="gpu-options">
                    {detectionQualities.map((q) => (
                        <button
                            key={q.id}
                            className={`gpu-btn ${gpuInfo?.detection_quality === q.id ? 'active' : ''}`}
                            onClick={() => handleSetDetectionQuality(q.id)}
                        >
                            <span>{q.name}</span>
                            <small>{q.description}</small>
                        </button>
                    ))}
                </div>
            </div>

            {/* Camera Grid */}
            <div className="camera-section">
                <div className="section-header">
                    <h2><Grid3X3 size={24} /> Camera Grid ({cameras.length}/32)</h2>
                    <button className="btn-primary" onClick={() => setShowAddCamera(true)}>
                        <Plus size={18} />
                        Add Camera
                    </button>
                </div>

                {cameras.length === 0 ? (
                    <div className="empty-state glass">
                        <Video size={48} />
                        <h3>No Cameras Configured</h3>
                        <p>Add cameras to start monitoring</p>
                        <button className="btn-primary" onClick={() => setShowAddCamera(true)}>
                            <Plus size={18} />
                            Add First Camera
                        </button>
                    </div>
                ) : (
                    <div className="camera-grid">
                        {cameras.map((camera) => {
                            const frameData = cameraFrames.find(f => f.camera_id === camera.id);
                            const recognitions = frameData?.recognitions || [];
                            const knownFaces = recognitions.filter(r => r.is_known);
                            const health = camera.health || frameData?.health || 'red';
                            return (
                                <div key={camera.id} className="camera-card glass">
                                    <div className="camera-preview">
                                        {frameData?.frame ? (
                                            <img src={frameData.frame} alt={camera.name} />
                                        ) : (
                                            <div className="no-preview">
                                                <Video size={32} />
                                                <span>
                                                    {camera.status === 'connected' ? 'Loading...' :
                                                        camera.status === 'reconnecting' ? `Reconnecting (${camera.reconnect_attempts})...` :
                                                            'No Signal'}
                                                </span>
                                            </div>
                                        )}
                                        <div className="camera-overlay">
                                            <span className="camera-status">
                                                <span
                                                    className="health-dot"
                                                    style={{ background: getHealthColor(health) }}
                                                />
                                                {getStatusIcon(camera.status)}
                                                {camera.status}
                                            </span>
                                            <span className="camera-fps">{camera.fps} FPS</span>
                                        </div>
                                        {/* Recognition badges overlay */}
                                        {knownFaces.length > 0 && (
                                            <div className="recognition-overlay">
                                                {knownFaces.slice(0, 3).map((r, idx) => (
                                                    <div key={idx} className="recognition-badge">
                                                        <Users size={12} />
                                                        <span>{r.name || r.student_id}</span>
                                                        <span className="badge-confidence">
                                                            {(r.confidence * 100).toFixed(0)}%
                                                        </span>
                                                    </div>
                                                ))}
                                                {knownFaces.length > 3 && (
                                                    <div className="recognition-badge more">
                                                        +{knownFaces.length - 3} more
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    <div className="camera-info">
                                        <h4>{camera.name}</h4>
                                        <div className="camera-meta">
                                            <span>ID: {camera.id}</span>
                                            <span>Detections: {camera.detection_count}</span>
                                        </div>
                                    </div>
                                    <div className="camera-actions">
                                        <button
                                            className="btn-icon-small danger"
                                            onClick={() => handleRemoveCamera(camera.id)}
                                            title="Remove"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Live Attendance Feed */}
            {stats?.running && (
                <div className="attendance-feed-section glass">
                    <div className="feed-header">
                        <h3><Users size={20} /> Live Attendance Feed</h3>
                        <span className="feed-count">{attendanceFeed.length} recent</span>
                    </div>
                    {attendanceFeed.length === 0 ? (
                        <div className="feed-empty">
                            <Clock size={24} />
                            <p>Waiting for face recognitions...</p>
                        </div>
                    ) : (
                        <div className="feed-list">
                            {attendanceFeed.slice(0, 20).map((entry, idx) => (
                                <div key={idx} className="feed-item">
                                    <div className="feed-avatar">
                                        <Users size={18} />
                                    </div>
                                    <div className="feed-info">
                                        <span className="feed-name">{entry.name}</span>
                                        <span className="feed-id">{entry.student_id} - {entry.class}</span>
                                    </div>
                                    <div className="feed-meta">
                                        <span className="feed-camera">
                                            <Camera size={12} />
                                            {entry.camera_name}
                                        </span>
                                        <span className="feed-time">
                                            <Clock size={12} />
                                            {new Date(entry.timestamp).toLocaleTimeString()}
                                        </span>
                                        <span className="feed-confidence">
                                            {(entry.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Add Camera Modal */}
            {showAddCamera && (
                <div className="modal-overlay" onClick={() => setShowAddCamera(false)}>
                    <div className="modal glass" onClick={e => e.stopPropagation()}>
                        <h2>Add Camera</h2>
                        <div className="form-group">
                            <label>Camera Name</label>
                            <input
                                type="text"
                                value={newCamera.name}
                                onChange={e => setNewCamera({ ...newCamera, name: e.target.value })}
                                placeholder="e.g., Entrance Camera"
                            />
                        </div>
                        <div className="form-group">
                            <label>Stream URL (RTSP/HTTP)</label>
                            <input
                                type="text"
                                value={newCamera.stream_url}
                                onChange={e => setNewCamera({ ...newCamera, stream_url: e.target.value })}
                                placeholder="rtsp://admin:password@192.168.1.100:554/stream"
                            />
                        </div>
                        <div className="form-row">
                            <div className="form-group">
                                <label>FPS Limit</label>
                                <select
                                    value={newCamera.fps_limit}
                                    onChange={e => setNewCamera({ ...newCamera, fps_limit: parseInt(e.target.value) })}
                                >
                                    {[1, 2, 3, 5, 10, 15, 20, 25, 30].map(fps => (
                                        <option key={fps} value={fps}>{fps} FPS</option>
                                    ))}
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="checkbox-label">
                                    <input
                                        type="checkbox"
                                        checked={newCamera.detection_enabled}
                                        onChange={e => setNewCamera({ ...newCamera, detection_enabled: e.target.checked })}
                                    />
                                    Enable Detection
                                </label>
                            </div>
                        </div>
                        <div className="modal-actions">
                            <button className="btn-secondary" onClick={() => setShowAddCamera(false)}>
                                Cancel
                            </button>
                            <button
                                className="btn-primary"
                                onClick={handleAddCamera}
                                disabled={!newCamera.name || !newCamera.stream_url}
                            >
                                <Plus size={18} />
                                Add Camera
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Settings Modal */}
            {showSettings && config && (
                <div className="modal-overlay" onClick={() => setShowSettings(false)}>
                    <div className="modal glass" onClick={e => e.stopPropagation()}>
                        <h2>Production Settings</h2>
                        <div className="form-group">
                            <label>Max Cameras (up to 64)</label>
                            <input
                                type="number"
                                value={config.max_cameras}
                                onChange={e => handleUpdateConfig({ max_cameras: parseInt(e.target.value) })}
                                min={1}
                                max={64}
                            />
                        </div>
                        <div className="form-group">
                            <label>Worker Threads</label>
                            <input
                                type="number"
                                value={config.worker_threads}
                                onChange={e => handleUpdateConfig({ worker_threads: parseInt(e.target.value) })}
                                min={1}
                                max={16}
                            />
                        </div>
                        <div className="form-group">
                            <label>Detection Interval (ms)</label>
                            <input
                                type="number"
                                value={config.detection_interval_ms}
                                onChange={e => handleUpdateConfig({ detection_interval_ms: parseInt(e.target.value) })}
                                min={50}
                                max={5000}
                            />
                        </div>
                        <div className="form-group">
                            <label className="checkbox-label">
                                <input
                                    type="checkbox"
                                    checked={config.enable_alerts}
                                    onChange={e => handleUpdateConfig({ enable_alerts: e.target.checked })}
                                />
                                Enable Alerts
                            </label>
                        </div>
                        <div className="modal-actions">
                            <button className="btn-primary" onClick={() => setShowSettings(false)}>
                                <CheckCircle2 size={18} />
                                Done
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
