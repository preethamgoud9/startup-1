import React from 'react';
import { Camera, UserPlus, Users } from 'lucide-react';
import './Navbar.css';

interface NavbarProps {
  currentPage: 'live' | 'enroll' | 'attendance';
  onNavigate: (page: 'live' | 'enroll' | 'attendance') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate }) => {
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
      </div>
    </nav>
  );
};
