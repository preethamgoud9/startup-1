import React, { useState, useEffect } from 'react';
import { Video, Save, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import api from '../services/api';
import './Settings.css';

export const Settings: React.FC = () => {
    const [usbId, setUsbId] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

    useEffect(() => {
        const fetchSettings = async () => {
            try {
                const response = await api.get('/settings/camera');
                setUsbId(response.data.usb_device_id);
            } catch (err) {
                console.error('Failed to fetch settings', err);
            } finally {
                setIsLoading(false);
            }
        };
        fetchSettings();
    }, []);

    const handleSave = async () => {
        setIsSaving(true);
        setMessage(null);
        try {
            await api.post('/settings/camera', {
                usb_device_id: usbId,
            });
            setMessage({ type: 'success', text: 'Vision configuration synchronized successfully.' });
        } catch (err: any) {
            setMessage({
                type: 'error',
                text: err.response?.data?.detail || 'Failed to connect to the specified vision source.'
            });
        } finally {
            setIsSaving(false);
        }
    };

    if (isLoading) {
        return (
            <div className="loader-container" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Loader2 className="animate-spin" size={48} opacity={0.2} />
            </div>
        );
    }

    return (
        <div className="settings-container animate-fade-in">
            <h1 className="live-title">System Configuration</h1>

            <div className="settings-card glass">
                <div className="settings-section">
                    <div className="section-header">
                        <Video size={28} className="brand-icon" />
                        <h2 className="section-title">Vision Acquisition Engine</h2>
                    </div>

                    <div className="settings-grid" style={{ display: 'grid', gap: '32px' }}>
                        <div className="form-group animate-fade-in">
                            <label className="form-label">Hardware Interface Index</label>
                            <select
                                className="form-input"
                                value={usbId}
                                onChange={(e) => setUsbId(parseInt(e.target.value))}
                            >
                                <option value={0}>Primary Interface (Index 0)</option>
                                <option value={1}>Secondary Interface (Index 1)</option>
                                <option value={2}>Auxiliary Interface (Index 2)</option>
                            </select>
                            <p className="input-hint">Standard USB cameras usually mount on index 0.</p>
                        </div>
                    </div>

                    {message && (
                        <div className={`status-toast ${message.type === 'success' ? 'status-success' : 'status-error'}`}>
                            {message.type === 'success' ? <CheckCircle2 size={24} /> : <AlertCircle size={24} />}
                            <span>{message.text}</span>
                        </div>
                    )}

                    <button
                        className="btn-primary synchronize-btn"
                        onClick={handleSave}
                        disabled={isSaving}
                    >
                        {isSaving ? (
                            <Loader2 className="animate-spin" size={24} />
                        ) : (
                            <Save size={24} />
                        )}
                        <span>{isSaving ? 'Processing...' : 'Synchronize Parameters'}</span>
                    </button>
                </div>
            </div>
        </div>
    );
};
