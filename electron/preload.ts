import { contextBridge, ipcRenderer } from 'electron'

export interface DetectorConfig {
  type: 'bf' | 'adf'
  inner: number
  outer: number
}

export interface SaveConfigResult {
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
  saveDetectorConfig: (config: DetectorConfig): Promise<SaveConfigResult> => {
    return ipcRenderer.invoke('save-detector-config', config)
  },
  loadDetectorConfig: (): Promise<LoadConfigResult> => {
    return ipcRenderer.invoke('load-detector-config')
  },
  saveCsv: (content: string, defaultFilename: string): Promise<SaveConfigResult> => {
    return ipcRenderer.invoke('save-csv', content, defaultFilename)
  },
})
