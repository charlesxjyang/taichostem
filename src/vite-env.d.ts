/// <reference types="vite/client" />

interface ElectronAPI {
  platform: string
  onFileSelected: (callback: (filePath: string) => void) => void
  removeFileSelectedListener: () => void
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export {}
