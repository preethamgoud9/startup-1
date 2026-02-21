import React, { useState, useEffect, useCallback } from 'react';
import {
    Globe,
    Loader2,
    CheckCircle2,
    AlertCircle,
    ArrowRight,
    ArrowLeft,
    Wifi,
    WifiOff,
    Monitor,
    Copy,
    Network,
    Plug,
    Download,
    RefreshCw,
    CheckCircle,
    XCircle,
    Plus,
} from 'lucide-react';
import {
    getTailscaleStatus,
    getNetworkInfo,
    testRemoteConnection,
    testRemoteRTSP,
    addCameraFromSetup,
} from '../services/api';
import type {
    TailscaleStatus,
    TailscalePeer,
    NetworkInfo,
    TCPTestResult,
    RTSPTestResult,
} from '../services/api';
import './RemoteAccess.css';

const WIZARD_STEPS = [
    { label: 'Install', icon: Download },
    { label: 'Status', icon: Wifi },
    { label: 'Network', icon: Network },
    { label: 'Test', icon: Plug },
    { label: 'Deploy', icon: Monitor },
];

interface TestedCamera {
    name: string;
    url: string;
    width?: number | null;
    height?: number | null;
    fps?: number | null;
    added: boolean;
}

export const RemoteAccess: React.FC = () => {
    const [step, setStep] = useState(1);
    const [loading, setLoading] = useState(false);

    // Step 1-2: Tailscale status
    const [tsStatus, setTsStatus] = useState<TailscaleStatus | null>(null);
    const [statusError, setStatusError] = useState<string | null>(null);

    // Step 3: Network info
    const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);

    // Step 4: Connectivity tests
    const [testHost, setTestHost] = useState('');
    const [testPort, setTestPort] = useState(554);
    const [tcpResults, setTcpResults] = useState<TCPTestResult[]>([]);
    const [rtspUrl, setRtspUrl] = useState('');
    const [rtspResults, setRtspResults] = useState<RTSPTestResult[]>([]);
    const [isTesting, setIsTesting] = useState(false);

    // Step 5: Cameras to add
    const [testedCameras, setTestedCameras] = useState<TestedCamera[]>([]);

    // Fetch tailscale status
    const fetchStatus = useCallback(async () => {
        setLoading(true);
        setStatusError(null);
        try {
            const status = await getTailscaleStatus();
            setTsStatus(status);
            if (status.error) {
                setStatusError(status.error);
            }
        } catch (err: any) {
            setStatusError(err.response?.data?.detail || 'Failed to check Tailscale status');
        } finally {
            setLoading(false);
        }
    }, []);

    // Fetch network info
    const fetchNetwork = useCallback(async () => {
        setLoading(true);
        try {
            const info = await getNetworkInfo();
            setNetworkInfo(info);
        } catch {
            // silently fail - status page will show the issue
        } finally {
            setLoading(false);
        }
    }, []);

    // Auto-fetch on step change
    useEffect(() => {
        if (step === 1 || step === 2) {
            fetchStatus();
        } else if (step === 3) {
            fetchNetwork();
        }
    }, [step, fetchStatus, fetchNetwork]);

    // TCP test
    const runTcpTest = async () => {
        if (!testHost) return;
        setIsTesting(true);
        try {
            const result = await testRemoteConnection(testHost, testPort);
            setTcpResults(prev => [result, ...prev]);
        } catch (err: any) {
            setTcpResults(prev => [{
                success: false,
                host: testHost,
                port: testPort,
                latency_ms: null,
                error: err.response?.data?.detail || 'Test failed',
            }, ...prev]);
        } finally {
            setIsTesting(false);
        }
    };

    // RTSP test
    const runRtspTest = async () => {
        if (!rtspUrl) return;
        setIsTesting(true);
        try {
            const result = await testRemoteRTSP(rtspUrl);
            setRtspResults(prev => [result, ...prev]);
            if (result.success) {
                // Auto-add to tested cameras
                const existing = testedCameras.find(c => c.url === result.url);
                if (!existing) {
                    setTestedCameras(prev => [...prev, {
                        name: `Remote Camera ${prev.length + 1}`,
                        url: result.url,
                        width: result.frame_width,
                        height: result.frame_height,
                        fps: result.fps,
                        added: false,
                    }]);
                }
            }
        } catch (err: any) {
            setRtspResults(prev => [{
                success: false,
                url: rtspUrl,
                message: err.response?.data?.detail || 'RTSP test failed',
                frame_width: null,
                frame_height: null,
                fps: null,
                codec: null,
            }, ...prev]);
        } finally {
            setIsTesting(false);
        }
    };

    // Add camera to production
    const addToProduction = async (camera: TestedCamera, index: number) => {
        try {
            await addCameraFromSetup({
                name: camera.name,
                stream_url: camera.url,
                fps_limit: 5,
                detection_enabled: true,
            });
            setTestedCameras(prev => prev.map((c, i) => i === index ? { ...c, added: true } : c));
        } catch (err: any) {
            alert(err.response?.data?.detail || 'Failed to add camera');
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
    };

    const canProceed = (): boolean => {
        switch (step) {
            case 1:
                return tsStatus?.installed === true;
            case 2:
                return tsStatus?.running === true;
            default:
                return true;
        }
    };

    // --- Step renderers ---

    const renderStep1 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Download size={40} className="step-icon" />
                <h2>Tailscale Installation</h2>
                <p>Checking if Tailscale is installed on this machine</p>
            </div>

            {loading ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <Loader2 size={32} className="step-icon" style={{ animation: 'spin 1s linear infinite' }} />
                    <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>Checking installation...</p>
                </div>
            ) : tsStatus?.installed ? (
                <div className="status-toast status-success">
                    <CheckCircle2 size={20} />
                    <span>Tailscale is installed{tsStatus.version ? ` (v${tsStatus.version})` : ''}</span>
                </div>
            ) : (
                <>
                    <div className="status-toast status-error">
                        <AlertCircle size={20} />
                        <span>Tailscale is not installed on this machine</span>
                    </div>
                    <div className="install-instructions">
                        <h3>Install Tailscale</h3>
                        <ol>
                            <li>
                                Visit <strong>tailscale.com/download</strong> and install for your platform
                            </li>
                            <li>
                                <strong>macOS:</strong> Install from the Mac App Store or run:<br />
                                <code>brew install tailscale</code>
                            </li>
                            <li>
                                <strong>Linux:</strong><br />
                                <code>curl -fsSL https://tailscale.com/install.sh | sh</code>
                            </li>
                            <li>
                                <strong>Windows:</strong> Download the installer from tailscale.com/download
                            </li>
                            <li>
                                After installing, click <strong>Refresh</strong> below to re-check
                            </li>
                        </ol>
                    </div>
                </>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={fetchStatus} disabled={loading}>
                    <RefreshCw size={18} />
                    <span>Refresh</span>
                </button>
                <button className="btn-primary" onClick={() => setStep(2)} disabled={!canProceed()}>
                    <span>Next</span>
                    <ArrowRight size={18} />
                </button>
            </div>
        </div>
    );

    const renderStep2 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Wifi size={40} className="step-icon" />
                <h2>Connection Status</h2>
                <p>Tailscale VPN connection details</p>
            </div>

            {loading ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <Loader2 size={32} className="step-icon" style={{ animation: 'spin 1s linear infinite' }} />
                    <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>Checking connection...</p>
                </div>
            ) : tsStatus?.running ? (
                <>
                    <div className="status-toast status-success" style={{ marginBottom: '20px' }}>
                        <CheckCircle2 size={20} />
                        <span>Connected to Tailscale</span>
                    </div>
                    <div className="status-grid">
                        <div className="status-card">
                            <div className="label">Tailscale IP</div>
                            <div className="value">{tsStatus.self_ip || '-'}</div>
                        </div>
                        <div className="status-card">
                            <div className="label">Tailnet</div>
                            <div className="value">{tsStatus.tailnet_name || '-'}</div>
                        </div>
                        <div className="status-card">
                            <div className="label">Hostname</div>
                            <div className="value">{tsStatus.hostname || '-'}</div>
                        </div>
                        <div className="status-card">
                            <div className="label">Version</div>
                            <div className="value">{tsStatus.version || '-'}</div>
                        </div>
                    </div>
                </>
            ) : (
                <>
                    <div className="status-toast status-error" style={{ marginBottom: '20px' }}>
                        <WifiOff size={20} />
                        <span>Tailscale is not connected</span>
                    </div>
                    {statusError && (
                        <div className="install-instructions">
                            <h3>How to Connect</h3>
                            <ol>
                                <li>Open the Tailscale app or run <code>tailscale up</code> in your terminal</li>
                                <li>Sign in with your Tailscale account</li>
                                <li>Make sure the school's device is also on the same Tailnet</li>
                                <li>Click <strong>Refresh</strong> after connecting</li>
                            </ol>
                        </div>
                    )}
                </>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(1)}>
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </button>
                <button className="btn-secondary" onClick={fetchStatus} disabled={loading}>
                    <RefreshCw size={18} />
                    <span>Refresh</span>
                </button>
                <button className="btn-primary" onClick={() => setStep(3)} disabled={!canProceed()}>
                    <span>Next</span>
                    <ArrowRight size={18} />
                </button>
            </div>
        </div>
    );

    const renderStep3 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Network size={40} className="step-icon" />
                <h2>Network Peers</h2>
                <p>Devices on your Tailnet and advertised subnet routes</p>
            </div>

            {loading ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <Loader2 size={32} className="step-icon" style={{ animation: 'spin 1s linear infinite' }} />
                    <p style={{ color: 'var(--text-muted)', marginTop: '12px' }}>Loading network info...</p>
                </div>
            ) : networkInfo ? (
                <>
                    <div className="status-grid" style={{ marginBottom: '20px' }}>
                        <div className="status-card">
                            <div className="label">Online Peers</div>
                            <div className="value">{networkInfo.online_count} / {networkInfo.total_count}</div>
                        </div>
                        <div className="status-card">
                            <div className="label">Subnet Routes</div>
                            <div className="value">{networkInfo.subnet_routes.length}</div>
                        </div>
                    </div>

                    {networkInfo.subnet_routes.length > 0 && (
                        <div className="subnet-summary">
                            <h4>Advertised Subnet Routes</h4>
                            <div className="routes">
                                {networkInfo.subnet_routes.map((r, i) => (
                                    <span key={i} className="subnet-badge">{r}</span>
                                ))}
                            </div>
                        </div>
                    )}

                    {networkInfo.peers.length > 0 ? (
                        <div style={{ overflowX: 'auto' }}>
                            <table className="peer-table">
                                <thead>
                                    <tr>
                                        <th>Status</th>
                                        <th>Hostname</th>
                                        <th>IP</th>
                                        <th>OS</th>
                                        <th>Subnets</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {networkInfo.peers.map((peer, i) => (
                                        <tr key={i}>
                                            <td>
                                                <span className={`online-dot ${peer.online ? 'online' : 'offline'}`} />
                                                {peer.online ? 'Online' : 'Offline'}
                                            </td>
                                            <td>{peer.hostname}</td>
                                            <td style={{ fontFamily: 'monospace' }}>{peer.ip}</td>
                                            <td>{peer.os || '-'}</td>
                                            <td>
                                                {peer.subnet_routes.length > 0
                                                    ? peer.subnet_routes.map((r, j) => (
                                                        <span key={j} className="subnet-badge">{r}</span>
                                                    ))
                                                    : '-'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="status-toast status-warning">
                            <AlertCircle size={20} />
                            <span>No peers found. Make sure other devices are connected to your Tailnet.</span>
                        </div>
                    )}

                    <div className="install-instructions" style={{ marginTop: '16px' }}>
                        <h3>Setting up the School Device</h3>
                        <p style={{ color: 'var(--text-muted)', marginBottom: '12px', fontSize: '13px' }}>
                            Run this on the school device to advertise camera subnets:
                        </p>
                        <div className="copy-command">
                            <code>sudo tailscale up --advertise-routes=192.168.1.0/24</code>
                            <button onClick={() => copyToClipboard('sudo tailscale up --advertise-routes=192.168.1.0/24')}>
                                <Copy size={14} /> Copy
                            </button>
                        </div>
                        <p style={{ color: 'var(--text-muted)', fontSize: '12px', margin: 0 }}>
                            Replace <code>192.168.1.0/24</code> with the actual camera subnet at the school.
                            Then approve the route in the Tailscale admin console.
                        </p>
                    </div>
                </>
            ) : (
                <div className="status-toast status-error">
                    <AlertCircle size={20} />
                    <span>Could not load network info. Is Tailscale running?</span>
                </div>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(2)}>
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </button>
                <button className="btn-secondary" onClick={fetchNetwork} disabled={loading}>
                    <RefreshCw size={18} />
                    <span>Refresh</span>
                </button>
                <button className="btn-primary" onClick={() => setStep(4)}>
                    <span>Next</span>
                    <ArrowRight size={18} />
                </button>
            </div>
        </div>
    );

    const renderStep4 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Plug size={40} className="step-icon" />
                <h2>Connectivity Test</h2>
                <p>Test TCP and RTSP connectivity to remote cameras</p>
            </div>

            {/* TCP Test */}
            <h3 style={{ marginBottom: '12px', fontSize: '16px' }}>TCP Port Test</h3>
            <div className="test-form">
                <div className="form-group">
                    <label>Remote IP / Hostname</label>
                    <input
                        className="form-input"
                        type="text"
                        placeholder="192.168.1.100 or 100.x.y.z"
                        value={testHost}
                        onChange={e => setTestHost(e.target.value)}
                    />
                </div>
                <div className="form-group" style={{ maxWidth: '100px' }}>
                    <label>Port</label>
                    <input
                        className="form-input"
                        type="number"
                        value={testPort}
                        onChange={e => setTestPort(Number(e.target.value))}
                    />
                </div>
                <button
                    className="btn-primary"
                    onClick={runTcpTest}
                    disabled={isTesting || !testHost}
                    style={{ padding: '10px 20px', height: 'fit-content' }}
                >
                    {isTesting ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Plug size={16} />}
                    <span>Test</span>
                </button>
            </div>

            {tcpResults.length > 0 && (
                <div className="test-results">
                    {tcpResults.map((r, i) => (
                        <div key={i} className={`test-result-item ${r.success ? 'success' : 'failure'}`}>
                            <span className="result-icon">
                                {r.success ? <CheckCircle size={18} color="#22c55e" /> : <XCircle size={18} color="#ef4444" />}
                            </span>
                            <span className="result-text">
                                {r.host}:{r.port} — {r.success ? 'Reachable' : r.error}
                            </span>
                            {r.latency_ms !== null && (
                                <span className="result-latency">{r.latency_ms}ms</span>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* RTSP Test */}
            <h3 style={{ marginBottom: '12px', marginTop: '24px', fontSize: '16px' }}>RTSP Stream Test</h3>
            <div className="rtsp-form">
                <div className="form-group">
                    <label>RTSP URL</label>
                    <input
                        className="form-input"
                        type="text"
                        placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
                        value={rtspUrl}
                        onChange={e => setRtspUrl(e.target.value)}
                    />
                </div>
                <button
                    className="btn-primary"
                    onClick={runRtspTest}
                    disabled={isTesting || !rtspUrl}
                    style={{ padding: '10px 20px', height: 'fit-content', flexShrink: 0 }}
                >
                    {isTesting ? <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> : <Wifi size={16} />}
                    <span>Test Stream</span>
                </button>
            </div>

            {rtspResults.length > 0 && (
                <div className="test-results">
                    {rtspResults.map((r, i) => (
                        <div key={i} className={`test-result-item ${r.success ? 'success' : 'failure'}`}>
                            <span className="result-icon">
                                {r.success ? <CheckCircle size={18} color="#22c55e" /> : <XCircle size={18} color="#ef4444" />}
                            </span>
                            <span className="result-text">
                                {r.message}
                                {r.frame_width && r.frame_height && ` (${r.frame_width}x${r.frame_height})`}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(3)}>
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </button>
                <button className="btn-primary" onClick={() => setStep(5)} disabled={testedCameras.length === 0}>
                    <span>Add to Production</span>
                    <ArrowRight size={18} />
                </button>
            </div>
        </div>
    );

    const renderStep5 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Monitor size={40} className="step-icon" />
                <h2>Add to Production</h2>
                <p>Add tested remote cameras to the production pipeline</p>
            </div>

            {testedCameras.length === 0 ? (
                <div className="status-toast status-warning">
                    <AlertCircle size={20} />
                    <span>No cameras tested yet. Go back and test an RTSP stream first.</span>
                </div>
            ) : (
                <div className="camera-add-list">
                    {testedCameras.map((cam, i) => (
                        <div key={i} className="camera-add-item">
                            <div className="cam-info">
                                <input
                                    className="cam-name"
                                    style={{
                                        border: 'none',
                                        background: 'transparent',
                                        fontWeight: 600,
                                        fontSize: '15px',
                                        color: 'var(--text-main)',
                                        padding: 0,
                                        outline: 'none',
                                        width: '100%',
                                    }}
                                    value={cam.name}
                                    onChange={e => {
                                        setTestedCameras(prev =>
                                            prev.map((c, j) => j === i ? { ...c, name: e.target.value } : c)
                                        );
                                    }}
                                />
                                <span className="cam-url">{cam.url}</span>
                                <span className="cam-stats">
                                    {cam.width && cam.height ? `${cam.width}x${cam.height}` : ''}
                                    {cam.fps ? ` @ ${cam.fps.toFixed(1)} FPS` : ''}
                                </span>
                            </div>
                            {cam.added ? (
                                <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#22c55e', fontWeight: 600, fontSize: '14px' }}>
                                    <CheckCircle size={18} />
                                    Added
                                </span>
                            ) : (
                                <button
                                    className="btn-primary"
                                    style={{ padding: '8px 16px', fontSize: '13px' }}
                                    onClick={() => addToProduction(cam, i)}
                                >
                                    <Plus size={16} />
                                    <span>Add</span>
                                </button>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {testedCameras.some(c => c.added) && (
                <div className="status-toast status-success">
                    <CheckCircle2 size={20} />
                    <span>Cameras added! Go to Production to start monitoring.</span>
                </div>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(4)}>
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </button>
            </div>
        </div>
    );

    return (
        <div className="remote-container">
            <div className="live-title">
                <Globe size={28} />
                <h1>Remote Access</h1>
            </div>

            {/* Progress bar */}
            <div className="wizard-progress">
                {WIZARD_STEPS.map((s, i) => (
                    <div
                        key={i}
                        className={`progress-step ${i + 1 <= step ? 'active' : ''} ${i + 1 === step ? 'current' : ''}`}
                        onClick={() => setStep(i + 1)}
                        style={{ cursor: 'pointer' }}
                    >
                        <div className="step-number">{i + 1}</div>
                        <span className="step-label">{s.label}</span>
                    </div>
                ))}
            </div>

            {/* Wizard content */}
            <div className="wizard-content glass">
                {step === 1 && renderStep1()}
                {step === 2 && renderStep2()}
                {step === 3 && renderStep3()}
                {step === 4 && renderStep4()}
                {step === 5 && renderStep5()}
            </div>
        </div>
    );
};
