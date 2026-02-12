import React, { useState, useEffect } from 'react';
import {
    Video,
    Loader2,
    CheckCircle2,
    AlertCircle,
    Wifi,
    Search,
    ArrowRight,
    ArrowLeft,
    Camera,
    Shield,
    Zap,
    HelpCircle,
    RefreshCw,
    CheckCircle,
    XCircle,
    Monitor,
} from 'lucide-react';
import api from '../services/api';
import './CCTVSetup.css';

interface Brand {
    name: string;
    description: string;
    default_port: number;
    default_username: string;
    notes?: string;
    has_http?: boolean;
    has_ddns?: boolean;
}

interface Protocol {
    name: string;
    description: string;
    default_port: number;
    prefix: string;
}

interface ConnectionResult {
    success: boolean;
    message: string;
    stream_url?: string;
    frame_width?: number;
    frame_height?: number;
    fps?: number;
    codec?: string;
    protocol?: string;
    resolved_ip?: string;
    troubleshooting_tips?: string[];
}

interface DDNSValidation {
    hostname: string;
    is_ddns: boolean;
    provider?: string;
    resolved_ip?: string;
    port_open: boolean;
    error?: string;
}

interface ScannedDevice {
    ip: string;
    port: number;
}

export const CCTVSetup: React.FC = () => {
    // Wizard state
    const [step, setStep] = useState(1);
    const [brands, setBrands] = useState<Record<string, Brand>>({});
    const [protocols, setProtocols] = useState<Record<string, Protocol>>({});

    // Form state
    const [selectedBrand, setSelectedBrand] = useState('');
    const [ip, setIp] = useState('');
    const [port, setPort] = useState(554);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [channel, setChannel] = useState(1);
    const [protocol, setProtocol] = useState('rtsp');
    const [useTcp, setUseTcp] = useState(false);

    // DDNS state
    const [ddnsValidation, setDdnsValidation] = useState<DDNSValidation | null>(null);
    const [isValidatingDdns, setIsValidatingDdns] = useState(false);

    // Connection state
    const [isConnecting, setIsConnecting] = useState(false);
    const [isScanning, setIsScanning] = useState(false);
    const [connectionResult, setConnectionResult] = useState<ConnectionResult | null>(null);
    const [scannedDevices, setScannedDevices] = useState<ScannedDevice[]>([]);

    // Load supported brands and protocols
    useEffect(() => {
        const fetchData = async () => {
            try {
                const [brandsRes, protocolsRes] = await Promise.all([
                    api.get('/cctv/brands'),
                    api.get('/cctv/protocols'),
                ]);
                setBrands(brandsRes.data);
                setProtocols(protocolsRes.data);
            } catch (err) {
                console.error('Failed to fetch data', err);
            }
        };
        fetchData();
    }, []);

    // Update defaults when brand changes
    useEffect(() => {
        if (selectedBrand && brands[selectedBrand]) {
            const brand = brands[selectedBrand];
            setPort(brand.default_port);
            setUsername(brand.default_username);
        }
    }, [selectedBrand, brands]);

    // Validate DDNS when IP changes (debounced)
    useEffect(() => {
        if (!ip || ip.length < 4) {
            setDdnsValidation(null);
            return;
        }

        const timer = setTimeout(async () => {
            // Check if it looks like a hostname (not just IP)
            const isHostname = ip.includes('.') && !ip.match(/^\d+\.\d+\.\d+\.\d+$/);
            if (isHostname) {
                setIsValidatingDdns(true);
                try {
                    const response = await api.post('/cctv/validate-ddns', {
                        hostname: ip,
                        port: port,
                    });
                    setDdnsValidation(response.data);
                } catch (err) {
                    console.error('DDNS validation failed', err);
                }
                setIsValidatingDdns(false);
            } else {
                setDdnsValidation(null);
            }
        }, 500);

        return () => clearTimeout(timer);
    }, [ip, port]);

    const handleScanNetwork = async () => {
        setIsScanning(true);
        setScannedDevices([]);
        try {
            const response = await api.get('/cctv/scan-network');
            setScannedDevices(response.data.devices || []);
        } catch (err) {
            console.error('Network scan failed', err);
        } finally {
            setIsScanning(false);
        }
    };

    const handleConnect = async () => {
        setIsConnecting(true);
        setConnectionResult(null);
        try {
            const response = await api.post('/cctv/connect-and-save', {
                brand: selectedBrand,
                ip,
                username,
                password,
                port,
                channel,
                protocol,
                use_tcp: useTcp,
            });
            setConnectionResult(response.data);
            if (response.data.success) {
                setStep(4); // Success step
            }
        } catch (err: any) {
            setConnectionResult({
                success: false,
                message: err.response?.data?.detail || 'Connection failed',
                troubleshooting_tips: [
                    'Check if the camera is powered on',
                    'Verify network connectivity',
                    'Ensure RTSP is enabled on the camera',
                ],
            });
        } finally {
            setIsConnecting(false);
        }
    };

    const handleSelectDevice = (device: ScannedDevice) => {
        setIp(device.ip);
        setPort(device.port);
    };

    const renderStep1 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Camera size={32} className="step-icon" />
                <h2>Select Your Camera Brand</h2>
                <p>Choose your CCTV/DVR/NVR brand for automatic configuration</p>
            </div>

            <div className="brand-grid">
                {Object.entries(brands).map(([brandId, brand]) => (
                    <button
                        key={brandId}
                        className={`brand-card ${selectedBrand === brandId ? 'selected' : ''}`}
                        onClick={() => setSelectedBrand(brandId)}
                    >
                        <div className="brand-name">{brand.name}</div>
                        <div className="brand-desc">{brand.description}</div>
                    </button>
                ))}
            </div>

            <div className="step-actions">
                <button
                    className="btn-primary"
                    onClick={() => setStep(2)}
                    disabled={!selectedBrand}
                >
                    <span>Continue</span>
                    <ArrowRight size={20} />
                </button>
            </div>
        </div>
    );

    const renderStep2 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Wifi size={32} className="step-icon" />
                <h2>Network Configuration</h2>
                <p>Enter your camera's IP address, DDNS hostname, or scan to find devices</p>
            </div>

            <div className="scan-section">
                <button
                    className="btn-secondary scan-btn"
                    onClick={handleScanNetwork}
                    disabled={isScanning}
                >
                    {isScanning ? (
                        <Loader2 className="animate-spin" size={20} />
                    ) : (
                        <Search size={20} />
                    )}
                    <span>{isScanning ? 'Scanning...' : 'Scan Network'}</span>
                </button>

                {scannedDevices.length > 0 && (
                    <div className="scanned-devices">
                        <h4>Found Devices:</h4>
                        <div className="device-list">
                            {scannedDevices.map((device, idx) => (
                                <button
                                    key={idx}
                                    className={`device-item ${ip === device.ip ? 'selected' : ''}`}
                                    onClick={() => handleSelectDevice(device)}
                                >
                                    <Wifi size={16} />
                                    <span>{device.ip}:{device.port}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            <div className="form-section">
                <div className="form-row">
                    <div className="form-group">
                        <label className="form-label">Camera IP / DDNS Hostname</label>
                        <input
                            type="text"
                            className="form-input"
                            value={ip}
                            onChange={(e) => setIp(e.target.value)}
                            placeholder="192.168.1.100 or mycamera.ddns.net"
                        />
                        {/* DDNS Validation Status */}
                        {isValidatingDdns && (
                            <p className="input-hint" style={{ color: 'var(--primary)' }}>
                                <Loader2 className="animate-spin" size={12} style={{ display: 'inline', marginRight: '4px' }} />
                                Resolving hostname...
                            </p>
                        )}
                        {ddnsValidation && !isValidatingDdns && (
                            <div className={`ddns-status ${ddnsValidation.resolved_ip ? 'success' : 'error'}`}>
                                {ddnsValidation.resolved_ip ? (
                                    <>
                                        <CheckCircle size={14} />
                                        <span>
                                            {ddnsValidation.is_ddns ? `DDNS resolved to ${ddnsValidation.resolved_ip}` : 'IP valid'}
                                            {ddnsValidation.provider && ` (${ddnsValidation.provider})`}
                                            {ddnsValidation.port_open ? ' - Port open' : ' - Port closed'}
                                        </span>
                                    </>
                                ) : (
                                    <>
                                        <XCircle size={14} />
                                        <span>{ddnsValidation.error || 'Could not resolve hostname'}</span>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="form-group" style={{ maxWidth: '120px' }}>
                        <label className="form-label">Port</label>
                        <input
                            type="number"
                            className="form-input"
                            value={port}
                            onChange={(e) => setPort(parseInt(e.target.value) || 554)}
                        />
                    </div>
                </div>

                {/* Protocol Selection */}
                <div className="form-group">
                    <label className="form-label">Streaming Protocol</label>
                    <div className="protocol-grid">
                        {Object.entries(protocols).map(([protoId, proto]) => (
                            <button
                                key={protoId}
                                type="button"
                                className={`protocol-btn ${protocol === protoId ? 'selected' : ''}`}
                                onClick={() => {
                                    setProtocol(protoId);
                                    setPort(proto.default_port);
                                }}
                            >
                                <span className="proto-name">{proto.name}</span>
                                <span className="proto-desc">{proto.description}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* TCP Transport Option */}
                {protocol === 'rtsp' && (
                    <div className="form-group">
                        <label className="checkbox-label">
                            <input
                                type="checkbox"
                                checked={useTcp}
                                onChange={(e) => setUseTcp(e.target.checked)}
                            />
                            <span>Use TCP transport (more reliable over unstable networks)</span>
                        </label>
                    </div>
                )}

                <div className="form-group">
                    <label className="form-label">Channel Number</label>
                    <select
                        className="form-input"
                        value={channel}
                        onChange={(e) => setChannel(parseInt(e.target.value))}
                    >
                        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((ch) => (
                            <option key={ch} value={ch}>
                                Channel {ch}
                            </option>
                        ))}
                    </select>
                    <p className="input-hint">Select the camera channel from your DVR/NVR</p>
                </div>
            </div>

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(1)}>
                    <ArrowLeft size={20} />
                    <span>Back</span>
                </button>
                <button
                    className="btn-primary"
                    onClick={() => setStep(3)}
                    disabled={!ip}
                >
                    <span>Continue</span>
                    <ArrowRight size={20} />
                </button>
            </div>
        </div>
    );

    const renderStep3 = () => (
        <div className="wizard-step animate-fade-in">
            <div className="step-header">
                <Shield size={32} className="step-icon" />
                <h2>Authentication</h2>
                <p>Enter your camera login credentials</p>
            </div>

            <div className="form-section">
                <div className="form-group">
                    <label className="form-label">Username</label>
                    <input
                        type="text"
                        className="form-input"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="admin"
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Password</label>
                    <input
                        type="password"
                        className="form-input"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter password"
                    />
                </div>

                <div className="info-box">
                    <HelpCircle size={20} />
                    <div>
                        <strong>Default Credentials</strong>
                        <p>
                            Most cameras use "admin" as username. Check your camera's
                            documentation or the sticker on the device for default password.
                        </p>
                    </div>
                </div>
            </div>

            {connectionResult && !connectionResult.success && (
                <div className="error-section">
                    <div className="status-toast status-error">
                        <AlertCircle size={24} />
                        <span>{connectionResult.message}</span>
                    </div>
                    {connectionResult.troubleshooting_tips && (
                        <div className="troubleshooting">
                            <h4>Troubleshooting Tips:</h4>
                            <ul>
                                {connectionResult.troubleshooting_tips.map((tip, idx) => (
                                    <li key={idx}>{tip}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            <div className="step-actions">
                <button className="btn-secondary" onClick={() => setStep(2)}>
                    <ArrowLeft size={20} />
                    <span>Back</span>
                </button>
                <button
                    className="btn-primary connect-btn"
                    onClick={handleConnect}
                    disabled={isConnecting || !username || !password}
                >
                    {isConnecting ? (
                        <>
                            <Loader2 className="animate-spin" size={20} />
                            <span>Connecting...</span>
                        </>
                    ) : (
                        <>
                            <Zap size={20} />
                            <span>Connect Camera</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );

    const renderStep4 = () => (
        <div className="wizard-step animate-fade-in success-step">
            <div className="success-icon">
                <CheckCircle2 size={64} />
            </div>
            <h2>Camera Connected Successfully!</h2>
            <p>Your CCTV camera is now configured and ready to use.</p>

            {connectionResult && (
                <div className="connection-details">
                    <div className="detail-item">
                        <span className="detail-label">Resolution</span>
                        <span className="detail-value">
                            {connectionResult.frame_width}x{connectionResult.frame_height}
                        </span>
                    </div>
                    {connectionResult.fps && (
                        <div className="detail-item">
                            <span className="detail-label">Frame Rate</span>
                            <span className="detail-value">{connectionResult.fps.toFixed(1)} FPS</span>
                        </div>
                    )}
                    {connectionResult.codec && (
                        <div className="detail-item">
                            <span className="detail-label">Codec</span>
                            <span className="detail-value">{connectionResult.codec}</span>
                        </div>
                    )}
                </div>
            )}

            <div className="step-actions">
                <button
                    className="btn-secondary"
                    onClick={() => {
                        setStep(1);
                        setConnectionResult(null);
                    }}
                >
                    <RefreshCw size={20} />
                    <span>Setup Another Camera</span>
                </button>
                <button
                    className="btn-primary"
                    onClick={async () => {
                        if (!connectionResult?.stream_url) return;
                        try {
                            const cameraName = `${selectedBrand ? (brands[selectedBrand]?.name || selectedBrand) : 'Camera'} - Ch${channel}`;
                            await api.post('/production/cameras/from-cctv-setup', {
                                name: cameraName,
                                stream_url: connectionResult.stream_url,
                                fps_limit: 5,
                                detection_enabled: true,
                            });
                            window.location.href = '/production';
                        } catch (err: any) {
                            alert(err.response?.data?.detail || 'Failed to add camera to production');
                        }
                    }}
                >
                    <Monitor size={20} />
                    <span>Add to Production</span>
                </button>
                <button
                    className="btn-secondary"
                    onClick={() => window.location.href = '/live'}
                >
                    <Video size={20} />
                    <span>Go to Live View</span>
                </button>
            </div>
        </div>
    );

    return (
        <div className="cctv-setup-container">
            <h1 className="live-title">
                <Video size={32} />
                CCTV Setup Wizard
            </h1>

            {/* Progress indicator */}
            <div className="wizard-progress">
                {[1, 2, 3, 4].map((s) => (
                    <div
                        key={s}
                        className={`progress-step ${step >= s ? 'active' : ''} ${step === s ? 'current' : ''}`}
                    >
                        <div className="step-number">{s === 4 ? '✓' : s}</div>
                        <div className="step-label">
                            {s === 1 && 'Brand'}
                            {s === 2 && 'Network'}
                            {s === 3 && 'Login'}
                            {s === 4 && 'Done'}
                        </div>
                    </div>
                ))}
            </div>

            <div className="wizard-content glass">
                {step === 1 && renderStep1()}
                {step === 2 && renderStep2()}
                {step === 3 && renderStep3()}
                {step === 4 && renderStep4()}
            </div>
        </div>
    );
};
