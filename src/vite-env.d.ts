/// <reference types="vite/client" />

interface DetectorConfig {
  type: 'none' | 'bf' | 'abf' | 'adf'
  inner: number
  outer: number
}

interface SaveResult {
  success: boolean
  canceled?: boolean
  filePath?: string
  error?: string
}

interface LoadConfigResult {
  success: boolean
  canceled?: boolean
  config?: DetectorConfig
  filePath?: string
  error?: string
}

interface DatasetInfo {
  filePath: string | null
  datasetPath: string | null
  shape: number[] | null
}

interface ElectronAPI {
  platform: string
  onFileSelected: (callback: (filePath: string) => void) => void
  removeFileSelectedListener: () => void
  onShowFilterHotPixelsDialog: (callback: () => void) => void
  removeFilterHotPixelsDialogListener: () => void
  saveDetectorConfig: (config: DetectorConfig) => Promise<SaveResult>
  loadDetectorConfig: () => Promise<LoadConfigResult>
  saveCsv: (content: string, defaultFilename: string) => Promise<SaveResult>
  saveImage: (base64Data: string, defaultFilename: string) => Promise<SaveResult>

  // Workflow window API
  openWorkflowWindow: (workflowType: string) => Promise<{ success: boolean }>
  updateDatasetInfo: (info: DatasetInfo) => Promise<{ success: boolean }>
  getDatasetInfo: () => Promise<DatasetInfo>

  // Position communication between windows
  sendPositionClicked: (position: { x: number; y: number }) => void
  onPositionClicked: (callback: (position: { x: number; y: number }) => void) => void
  removePositionClickedListener: () => void
  highlightPosition: (position: { x: number; y: number }) => void
  onHighlightPosition: (callback: (position: { x: number; y: number }) => void) => void
  removeHighlightPositionListener: () => void

  // Dataset info updates (for workflow window)
  onDatasetInfoUpdated: (callback: (info: DatasetInfo) => void) => void
  removeDatasetInfoUpdatedListener: () => void

  // Clicked position management
  updateClickedPosition: (position: { x: number; y: number } | null) => Promise<{ success: boolean }>
  getClickedPosition: () => Promise<{ x: number; y: number } | null>

  // Selection geometry management
  updateSelection: (selection: { type: 'rectangle' | 'ellipse' | 'polygon'; points: [number, number][] } | null) => Promise<{ success: boolean }>
  getSelection: () => Promise<{ type: 'rectangle' | 'ellipse' | 'polygon'; points: [number, number][] } | null>

  // Workflow window state notifications
  onWorkflowWindowOpened: (callback: () => void) => void
  onWorkflowWindowClosed: (callback: () => void) => void
  removeWorkflowWindowListeners: () => void
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export {}
