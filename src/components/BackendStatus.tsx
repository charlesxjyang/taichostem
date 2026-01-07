import { useState, useEffect } from 'react'

const BACKEND_URL = 'http://127.0.0.1:8000'
const POLL_INTERVAL_MS = 2000

interface BackendStatusProps {
  className?: string
}

/**
 * Status indicator component that polls the backend health endpoint
 * and displays connection status.
 */
export function BackendStatus({ className }: BackendStatusProps) {
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(1500),
        })
        const data = await response.json()
        setIsConnected(data.status === 'ok')
      } catch {
        setIsConnected(false)
      }
    }

    checkHealth()
    const intervalId = setInterval(checkHealth, POLL_INTERVAL_MS)

    return () => clearInterval(intervalId)
  }, [])

  return (
    <div className={`backend-status ${className ?? ''}`}>
      <span
        className={`backend-status-dot ${isConnected ? 'connected' : 'disconnected'}`}
      />
      <span className="backend-status-text">
        Backend: {isConnected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  )
}
