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
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export {}
