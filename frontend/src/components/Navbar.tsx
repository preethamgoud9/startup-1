import React from 'react';
import { Camera, UserPlus, Users, LogOut, User as UserIcon, Settings as SettingsIcon, Radio, Monitor } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Navbar.css';

interface NavbarProps {
  currentPage: 'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production';
  onNavigate: (page: 'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate }) => {
  const { user, logout } = useAuth();

  return (
    <nav className="navbar-container glass animate-fade-in">
      <div className="brand">
        <Camera size={28} className="brand-icon" />
        <span className="brand-text">FaceRecog AI</span>
      </div>

      <div className="nav-links">
        <button
          className={`nav-item ${currentPage === 'live' ? 'active' : ''}`}
          onClick={() => onNavigate('live')}
        >
          <Camera size={18} />
          <span>Live View</span>
        </button>
        <button
          className={`nav-item ${currentPage === 'enroll' ? 'active' : ''}`}
          onClick={() => onNavigate('enroll')}
        >
          <UserPlus size={18} />
          <span>Enroll</span>
        </button>
        <button
          className={`nav-item ${currentPage === 'attendance' ? 'active' : ''}`}
          onClick={() => onNavigate('attendance')}
        >
          <Users size={18} />
          <span>Records</span>
        </button>
        <button
          className={`nav-item ${currentPage === 'cctv' ? 'active' : ''}`}
          onClick={() => onNavigate('cctv')}
        >
          <Radio size={18} />
          <span>CCTV Setup</span>
        </button>
        <button
          className={`nav-item ${currentPage === 'production' ? 'active' : ''}`}
          onClick={() => onNavigate('production')}
        >
          <Monitor size={18} />
          <span>Production</span>
        </button>
        <button
          className={`nav-item ${currentPage === 'settings' ? 'active' : ''}`}
          onClick={() => onNavigate('settings')}
        >
          <SettingsIcon size={18} />
          <span>Settings</span>
        </button>
      </div>

      <div className="nav-user-section">
        <div className="user-profile">
          <div className="user-avatar">
            <UserIcon size={16} />
          </div>
          <span className="user-name">{user?.username}</span>
        </div>
        <button className="logout-btn" onClick={logout} title="End Session">
          <LogOut size={18} />
        </button>
      </div>
    </nav>
  );
};
