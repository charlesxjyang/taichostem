/// <reference types="vite/client" />

interface DetectorConfig {
  type: 'bf' | 'adf'
  inner: number
  outer: number
}

interface SaveConfigResult {
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
  saveDetectorConfig: (config: DetectorConfig) => Promise<SaveConfigResult>
  loadDetectorConfig: () => Promise<LoadConfigResult>
  saveCsv: (content: string, defaultFilename: string) => Promise<SaveConfigResult>
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export {}
