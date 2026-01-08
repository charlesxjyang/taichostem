import { app, BrowserWindow, Menu, dialog, ipcMain } from 'electron'
import fs from 'node:fs'
import { spawn, ChildProcess } from 'node:child_process'
import path from 'node:path'

process.env.DIST = path.join(__dirname, '../dist')
process.env.VITE_PUBLIC = app.isPackaged ? process.env.DIST : path.join(process.env.DIST, '../public')

let win: BrowserWindow | null
let pythonProcess: ChildProcess | null = null

const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
const BACKEND_PORT = 8000

/**
 * Spawns the Python FastAPI backend as a subprocess.
 * Runs uvicorn on localhost:8000.
 */
function startPythonBackend(): void {
  const backendPath = path.join(__dirname, '..', 'backend')

  pythonProcess = spawn('uvicorn', ['app.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)], {
    cwd: backendPath,
    shell: true,
  })

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    console.log(`[Python Backend] ${data.toString().trim()}`)
  })

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    console.error(`[Python Backend] ${data.toString().trim()}`)
  })

  pythonProcess.on('error', (err: Error) => {
    console.error('[Python Backend] Failed to start:', err.message)
  })

  pythonProcess.on('close', (code: number | null) => {
    console.log(`[Python Backend] Process exited with code ${code}`)
    pythonProcess = null
  })
}

/**
 * Kills the Python backend subprocess if running.
 */
function stopPythonBackend(): void {
  if (pythonProcess) {
    console.log('[Python Backend] Shutting down...')
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    title: '4D-STEM Viewer',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  })

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    win.loadFile(path.join(process.env.DIST, 'index.html'))
  }
}

/**
 * Opens a native file dialog for selecting 4D-STEM datasets.
 * Sends the selected file path to the renderer process.
 */
async function openDatasetDialog(): Promise<void> {
  if (!win) return

  const result = await dialog.showOpenDialog(win, {
    title: 'Open Dataset',
    filters: [
      { name: '4D-STEM Datasets', extensions: ['dm4', 'hdf5', 'h5', 'mrc'] },
      { name: 'All Files', extensions: ['*'] },
    ],
    properties: ['openFile'],
  })

  if (!result.canceled && result.filePaths.length > 0) {
    win.webContents.send('file-selected', result.filePaths[0])
  }
}

/**
 * Creates the application menu bar.
 */
function createMenu(): void {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Dataset...',
          accelerator: 'CmdOrCtrl+O',
          click: () => openDatasetDialog(),
        },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { role: 'close' },
      ],
    },
  ]

  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopPythonBackend()
    app.quit()
    win = null
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.on('before-quit', () => {
  stopPythonBackend()
})

// IPC handler for saving detector config
ipcMain.handle('save-detector-config', async (_event, config: object) => {
  if (!win) return { success: false, error: 'No window' }

  const result = await dialog.showSaveDialog(win, {
    title: 'Save Detector Configuration',
    defaultPath: 'detector-config.json',
    filters: [
      { name: 'JSON Files', extensions: ['json'] },
    ],
  })

  if (result.canceled || !result.filePath) {
    return { success: false, canceled: true }
  }

  try {
    fs.writeFileSync(result.filePath, JSON.stringify(config, null, 2), 'utf-8')
    return { success: true, filePath: result.filePath }
  } catch (err) {
    return { success: false, error: (err as Error).message }
  }
})

// IPC handler for saving CSV files
ipcMain.handle('save-csv', async (_event, content: string, defaultFilename: string) => {
  if (!win) return { success: false, error: 'No window' }

  const result = await dialog.showSaveDialog(win, {
    title: 'Export CSV',
    defaultPath: defaultFilename,
    filters: [
      { name: 'CSV Files', extensions: ['csv'] },
    ],
  })

  if (result.canceled || !result.filePath) {
    return { success: false, canceled: true }
  }

  try {
    fs.writeFileSync(result.filePath, content, 'utf-8')
    return { success: true, filePath: result.filePath }
  } catch (err) {
    return { success: false, error: (err as Error).message }
  }
})

// IPC handler for loading detector config
ipcMain.handle('load-detector-config', async () => {
  if (!win) return { success: false, error: 'No window' }

  const result = await dialog.showOpenDialog(win, {
    title: 'Load Detector Configuration',
    filters: [
      { name: 'JSON Files', extensions: ['json'] },
    ],
    properties: ['openFile'],
  })

  if (result.canceled || result.filePaths.length === 0) {
    return { success: false, canceled: true }
  }

  try {
    const content = fs.readFileSync(result.filePaths[0], 'utf-8')
    const config = JSON.parse(content)
    return { success: true, config, filePath: result.filePaths[0] }
  } catch (err) {
    return { success: false, error: (err as Error).message }
  }
})

app.whenReady().then(() => {
  createMenu()
  startPythonBackend()
  createWindow()
})
