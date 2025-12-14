import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import HealthFormPage from './pages/health_form'
import LandingPage from './pages/landing_page'

function AppRouter() {
  const [username, setUsername] = useState<string | null>(() => sessionStorage.getItem('username'))
  const [path, setPath] = useState(() => window.location.pathname || '/')

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname || '/')
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const go = (nextPath: string) => {
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, '', nextPath)
      setPath(nextPath)
    }
  }

  const handleContinue = (name: string) => {
    setUsername(name)
    sessionStorage.setItem('username', name)
    go('/health')
  }

  if (path === '/health') {
    if (!username) {
      go('/')
      return <LandingPage onContinue={handleContinue} />
    }
    return <HealthFormPage username={username} />
  }

  return <LandingPage onContinue={handleContinue} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppRouter />
  </StrictMode>,
)
