import React, { useState } from 'react';
import { Lock, User, ShieldCheck, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { login as loginApi } from '../services/api';
import './Login.css';

export const Login: React.FC = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const { login } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);

        try {
            const response = await loginApi(username, password);
            // For simplicity, we assume the user is admin as per our backend logic
            login(response.access_token, { username, is_admin: true });
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Authentication failed. Please check credentials.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="login-container animate-fade-in">
            <div className="login-card glass">
                <div className="login-header">
                    <div className="login-icon-wrapper">
                        <ShieldCheck size={32} />
                    </div>
                    <h1 className="login-title">Secure Access</h1>
                    <p className="login-subtitle">Enter credentials to access the vision engine</p>
                </div>

                <form className="login-form" onSubmit={handleSubmit}>
                    <div className="form-group">
                        <div className="input-icon-wrapper">
                            <User size={20} className="input-icon" />
                            <input
                                type="text"
                                className="login-input"
                                placeholder="Username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                            />
                        </div>
                    </div>

                    <div className="form-group">
                        <div className="input-icon-wrapper">
                            <Lock size={20} className="input-icon" />
                            <input
                                type="password"
                                className="login-input"
                                placeholder="Password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>
                    </div>

                    {error && (
                        <div className="error-toast" style={{ margin: 0, fontSize: '0.9rem' }}>
                            <AlertCircle size={20} />
                            <span>{error}</span>
                        </div>
                    )}

                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={isLoading}
                        style={{ width: '100%', justifyContent: 'center', marginTop: 12 }}
                    >
                        {isLoading ? (
                            <>
                                <Loader2 size={20} className="animate-spin" />
                                <span>Authenticating...</span>
                            </>
                        ) : (
                            <span>Authorize Login</span>
                        )}
                    </button>
                </form>

                <div className="login-footer">
                    Videntity Analytical Suite v1.0
                </div>
            </div>
        </div>
    );
};
