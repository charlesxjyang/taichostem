import { contextBridge, ipcRenderer } from 'electron'

export interface DetectorConfig {
  type: 'none' | 'bf' | 'abf' | 'adf'
  inner: number
  outer: number
}

export interface SaveResult {
  success: boolean
  canceled?: boolean
  filePath?: string
  error?: string
}

export interface LoadConfigResult {
  success: boolean
  canceled?: boolean
  config?: DetectorConfig
  filePath?: string
  error?: string
}

export interface DatasetInfo {
  filePath: string | null
  datasetPath: string | null
  shape: number[] | null
}

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  onFileSelected: (callback: (filePath: string) => void) => {
    ipcRenderer.on('file-selected', (_event, filePath: string) => callback(filePath))
  },
  removeFileSelectedListener: () => {
    ipcRenderer.removeAllListeners('file-selected')
  },
  onShowFilterHotPixelsDialog: (callback: () => void) => {
    ipcRenderer.on('show-filter-hot-pixels-dialog', () => callback())
  },
  removeFilterHotPixelsDialogListener: () => {
    ipcRenderer.removeAllListeners('show-filter-hot-pixels-dialog')
  },
  saveDetectorConfig: (config: DetectorConfig): Promise<SaveResult> => {
    return ipcRenderer.invoke('save-detector-config', config)
  },
  loadDetectorConfig: (): Promise<LoadConfigResult> => {
    return ipcRenderer.invoke('load-detector-config')
  },
  saveCsv: (content: string, defaultFilename: string): Promise<SaveResult> => {
    return ipcRenderer.invoke('save-csv', content, defaultFilename)
  },
  saveImage: (base64Data: string, defaultFilename: string): Promise<SaveResult> => {
    return ipcRenderer.invoke('save-image', base64Data, defaultFilename)
  },

  // Workflow window API
  openWorkflowWindow: (workflowType: string): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('open-workflow-window', workflowType)
  },
  updateDatasetInfo: (info: DatasetInfo): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('update-dataset-info', info)
  },
  getDatasetInfo: (): Promise<DatasetInfo> => {
    return ipcRenderer.invoke('get-dataset-info')
  },

  // Position communication between windows
  sendPositionClicked: (position: { x: number; y: number }) => {
    ipcRenderer.send('position-clicked', position)
  },
  onPositionClicked: (callback: (position: { x: number; y: number }) => void) => {
    ipcRenderer.on('position-clicked', (_event, position) => callback(position))
  },
  removePositionClickedListener: () => {
    ipcRenderer.removeAllListeners('position-clicked')
  },
  highlightPosition: (position: { x: number; y: number }) => {
    ipcRenderer.send('highlight-position', position)
  },
  onHighlightPosition: (callback: (position: { x: number; y: number }) => void) => {
    ipcRenderer.on('highlight-position', (_event, position) => callback(position))
  },
  removeHighlightPositionListener: () => {
    ipcRenderer.removeAllListeners('highlight-position')
  },

  // Dataset info updates (for workflow window)
  onDatasetInfoUpdated: (callback: (info: DatasetInfo) => void) => {
    ipcRenderer.on('dataset-info-updated', (_event, info) => callback(info))
  },
  removeDatasetInfoUpdatedListener: () => {
    ipcRenderer.removeAllListeners('dataset-info-updated')
  },

  // Clicked position management
  updateClickedPosition: (position: { x: number; y: number } | null): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('update-clicked-position', position)
  },
  getClickedPosition: (): Promise<{ x: number; y: number } | null> => {
    return ipcRenderer.invoke('get-clicked-position')
  },

  // Selection geometry management
  updateSelection: (selection: { type: 'rectangle' | 'ellipse' | 'polygon'; points: [number, number][] } | null): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('update-selection', selection)
  },
  getSelection: (): Promise<{ type: 'rectangle' | 'ellipse' | 'polygon'; points: [number, number][] } | null> => {
    return ipcRenderer.invoke('get-selection')
  },

  // Workflow window state notifications
  onWorkflowWindowOpened: (callback: () => void) => {
    ipcRenderer.on('workflow-window-opened', () => callback())
  },
  onWorkflowWindowClosed: (callback: () => void) => {
    ipcRenderer.on('workflow-window-closed', () => callback())
  },
  removeWorkflowWindowListeners: () => {
    ipcRenderer.removeAllListeners('workflow-window-opened')
    ipcRenderer.removeAllListeners('workflow-window-closed')
  },
})
