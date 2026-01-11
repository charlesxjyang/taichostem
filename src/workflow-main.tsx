import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { WorkflowWindow } from './components/WorkflowWindow'
import './index.css'

createRoot(document.getElementById('workflow-root')!).render(
  <StrictMode>
    <WorkflowWindow />
  </StrictMode>,
)
