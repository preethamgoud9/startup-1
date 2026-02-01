import { useState } from 'react'
import { Navbar } from './components/Navbar'
import { LiveRecognition } from './pages/LiveRecognition'
import { EnrollStudent } from './pages/EnrollStudent'
import { AttendanceDashboard } from './pages/AttendanceDashboard'
import { Login } from './pages/Login'
import { Settings } from './pages/Settings'
import { CCTVSetup } from './pages/CCTVSetup'
import { Production } from './pages/Production'
import { AuthProvider, useAuth } from './context/AuthContext'
import './App.css'

function AppContent() {
  const [currentPage, setCurrentPage] = useState<'live' | 'enroll' | 'attendance' | 'settings' | 'cctv' | 'production'>('live')
  const { isAuthenticated, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="loader-container" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Initializing Secure Session...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Login />
  }

  return (
    <div style={{ minHeight: '100vh', width: '100vw', margin: 0, padding: 0, overflowX: 'hidden' }}>
      <Navbar currentPage={currentPage} onNavigate={setCurrentPage} />
      {currentPage === 'live' && <LiveRecognition />}
      {currentPage === 'enroll' && <EnrollStudent />}
      {currentPage === 'attendance' && <AttendanceDashboard />}
      {currentPage === 'settings' && <Settings />}
      {currentPage === 'cctv' && <CCTVSetup />}
      {currentPage === 'production' && <Production />}
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}

export default App
