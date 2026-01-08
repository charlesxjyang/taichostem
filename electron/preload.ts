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
})
