import { useState } from 'react'
import { Navbar } from './components/Navbar'
import { LiveRecognition } from './pages/LiveRecognition'
import { EnrollStudent } from './pages/EnrollStudent'
import { AttendanceDashboard } from './pages/AttendanceDashboard'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState<'live' | 'enroll' | 'attendance'>('live')

  return (
    <div style={{ minHeight: '100vh', width: '100vw', backgroundColor: '#f1f5f9', margin: 0, padding: 0, overflowX: 'hidden' }}>
      <Navbar currentPage={currentPage} onNavigate={setCurrentPage} />
      {currentPage === 'live' && <LiveRecognition />}
      {currentPage === 'enroll' && <EnrollStudent />}
      {currentPage === 'attendance' && <AttendanceDashboard />}
    </div>
  )
}

export default App
