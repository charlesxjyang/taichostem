import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  onFileSelected: (callback: (filePath: string) => void) => {
    ipcRenderer.on('file-selected', (_event, filePath: string) => callback(filePath))
  },
  removeFileSelectedListener: () => {
    ipcRenderer.removeAllListeners('file-selected')
  },
})
