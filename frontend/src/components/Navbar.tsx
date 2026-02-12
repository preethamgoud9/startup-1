import React, { useState } from 'react';
import { Camera, UserPlus, Users, LogOut, User as UserIcon, Settings as SettingsIcon, Radio, Monitor, Menu, X } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Navbar.css';

interface NavbarProps {
  currentPage: 'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production';
  onNavigate: (page: 'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate }) => {
  const { user, logout } = useAuth();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleNavigate = (page: 'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production') => {
    onNavigate(page);
    setMobileMenuOpen(false);
  };

  return (
    <>
      <nav className="navbar-container glass animate-fade-in">
        <div className="brand">
          <Camera size={28} className="brand-icon" />
          <span className="brand-text">FaceRecog AI</span>
        </div>

        {/* Desktop Navigation */}
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
          {/* Mobile Menu Button */}
          <button
            className="mobile-menu-btn glass"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>

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

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="mobile-menu animate-fade-in">
          <button
            className={`mobile-menu-item ${currentPage === 'live' ? 'active' : ''}`}
            onClick={() => handleNavigate('live')}
          >
            <Camera size={20} />
            <span>Live Recognition</span>
          </button>
          <button
            className={`mobile-menu-item ${currentPage === 'enroll' ? 'active' : ''}`}
            onClick={() => handleNavigate('enroll')}
          >
            <UserPlus size={20} />
            <span>Enroll Student</span>
          </button>
          <button
            className={`mobile-menu-item ${currentPage === 'attendance' ? 'active' : ''}`}
            onClick={() => handleNavigate('attendance')}
          >
            <Users size={20} />
            <span>Attendance Records</span>
          </button>
          <button
            className={`mobile-menu-item ${currentPage === 'cctv' ? 'active' : ''}`}
            onClick={() => handleNavigate('cctv')}
          >
            <Radio size={20} />
            <span>CCTV Setup</span>
          </button>
          <button
            className={`mobile-menu-item ${currentPage === 'production' ? 'active' : ''}`}
            onClick={() => handleNavigate('production')}
          >
            <Monitor size={20} />
            <span>Production Mode</span>
          </button>
          <button
            className={`mobile-menu-item ${currentPage === 'settings' ? 'active' : ''}`}
            onClick={() => handleNavigate('settings')}
          >
            <SettingsIcon size={20} />
            <span>Settings</span>
          </button>
        </div>
      )}
    </>
  );
};
