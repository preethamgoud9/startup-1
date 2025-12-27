import React from 'react';
import { Camera, UserPlus, Users } from 'lucide-react';

interface NavbarProps {
  currentPage: 'live' | 'enroll' | 'attendance';
  onNavigate: (page: 'live' | 'enroll' | 'attendance') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate }) => {
  return (
    <nav style={styles.nav}>
      <div style={styles.brand}>
        <Camera size={24} />
        <span style={styles.brandText}>Face Recognition Attendance</span>
      </div>
      <div style={styles.navLinks}>
        <button
          style={{
            ...styles.navButton,
            ...(currentPage === 'live' ? styles.navButtonActive : {}),
          }}
          onClick={() => onNavigate('live')}
        >
          <Camera size={18} />
          <span>Live Recognition</span>
        </button>
        <button
          style={{
            ...styles.navButton,
            ...(currentPage === 'enroll' ? styles.navButtonActive : {}),
          }}
          onClick={() => onNavigate('enroll')}
        >
          <UserPlus size={18} />
          <span>Enroll Student</span>
        </button>
        <button
          style={{
            ...styles.navButton,
            ...(currentPage === 'attendance' ? styles.navButtonActive : {}),
          }}
          onClick={() => onNavigate('attendance')}
        >
          <Users size={18} />
          <span>Attendance</span>
        </button>
      </div>
    </nav>
  );
};

const styles: Record<string, React.CSSProperties> = {
  nav: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1.25rem 2.5rem',
    backgroundColor: '#0f172a',
    color: 'white',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    borderBottom: '2px solid #1e293b',
    width: '100%',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.875rem',
    fontSize: '1.5rem',
    fontWeight: '700',
    letterSpacing: '-0.025em',
  },
  brandText: {
    fontSize: '1.5rem',
    background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  navLinks: {
    display: 'flex',
    gap: '0.5rem',
  },
  navButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.75rem 1.5rem',
    backgroundColor: 'transparent',
    color: '#cbd5e1',
    border: 'none',
    borderRadius: '0.75rem',
    cursor: 'pointer',
    fontSize: '0.9375rem',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
  navButtonActive: {
    backgroundColor: '#3b82f6',
    color: 'white',
    boxShadow: '0 4px 6px -1px rgba(59, 130, 246, 0.4)',
    fontWeight: '600',
  },
};
