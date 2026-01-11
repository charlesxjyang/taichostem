import { app, BrowserWindow, Menu, dialog, ipcMain } from 'electron'
import fs from 'node:fs'
import { spawn, ChildProcess } from 'node:child_process'
import path from 'node:path'

process.env.DIST = path.join(__dirname, '../dist')
process.env.VITE_PUBLIC = app.isPackaged ? process.env.DIST : path.join(process.env.DIST, '../public')

let win: BrowserWindow | null
let workflowWin: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null

// Shared state for workflow windows
let currentDatasetInfo: {
  filePath: string | null
  datasetPath: string | null
  shape: number[] | null
} = {
  filePath: null,
  datasetPath: null,
  shape: null,
}

// Current clicked position in real space (shared with workflow windows)
let currentClickedPosition: { x: number; y: number } | null = null

// Current selection geometry (shared with workflow windows)
let currentSelection: {
  type: 'rectangle' | 'ellipse' | 'polygon'
  points: [number, number][]
} | null = null

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
 * Creates a workflow window for complex analysis workflows.
 * @param workflowType - The type of workflow (e.g., 'disk-detection')
 */
function createWorkflowWindow(workflowType: string): void {
  // If workflow window already exists, focus it
  if (workflowWin && !workflowWin.isDestroyed()) {
    workflowWin.focus()
    return
  }

  workflowWin = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1000,
    minHeight: 700,
    title: 'Disk Detection Workflow',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  })

  // Load the workflow window HTML
  if (VITE_DEV_SERVER_URL) {
    workflowWin.loadURL(`${VITE_DEV_SERVER_URL}workflow.html?type=${workflowType}`)
  } else {
    workflowWin.loadFile(path.join(process.env.DIST, 'workflow.html'), {
      query: { type: workflowType },
    })
  }

  workflowWin.on('closed', () => {
    workflowWin = null
    // Notify main window that workflow window closed
    if (win && !win.isDestroyed()) {
      win.webContents.send('workflow-window-closed')
    }
  })

  // Notify main window that workflow window opened
  if (win && !win.isDestroyed()) {
    win.webContents.send('workflow-window-opened')
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
      label: 'Preprocess',
      submenu: [
        {
          label: 'Filter Hot Pixels...',
          click: () => {
            if (win) {
              win.webContents.send('show-filter-hot-pixels-dialog')
            }
          },
        },
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

// IPC handler for saving images (PNG from base64)
ipcMain.handle('save-image', async (_event, base64Data: string, defaultFilename: string) => {
  if (!win) return { success: false, error: 'No window' }

  const result = await dialog.showSaveDialog(win, {
    title: 'Export Image',
    defaultPath: defaultFilename,
    filters: [
      { name: 'PNG Image', extensions: ['png'] },
      { name: 'TIFF Image', extensions: ['tiff', 'tif'] },
    ],
  })

  if (result.canceled || !result.filePath) {
    return { success: false, canceled: true }
  }

  try {
    // Convert base64 to buffer and write to file
    const buffer = Buffer.from(base64Data, 'base64')
    fs.writeFileSync(result.filePath, buffer)
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

// IPC handler for opening workflow window
ipcMain.handle('open-workflow-window', (_event, workflowType: string) => {
  createWorkflowWindow(workflowType)
  return { success: true }
})

// IPC handler for updating dataset info (called by main window when dataset is loaded)
ipcMain.handle('update-dataset-info', (_event, info: typeof currentDatasetInfo) => {
  currentDatasetInfo = info
  // Notify workflow window if it exists
  if (workflowWin && !workflowWin.isDestroyed()) {
    workflowWin.webContents.send('dataset-info-updated', currentDatasetInfo)
  }
  return { success: true }
})

// IPC handler for getting current dataset info (called by workflow window)
ipcMain.handle('get-dataset-info', () => {
  return currentDatasetInfo
})

// IPC handler for main window to send position clicks to workflow window
ipcMain.on('position-clicked', (_event, position: { x: number; y: number }) => {
  if (workflowWin && !workflowWin.isDestroyed()) {
    workflowWin.webContents.send('position-clicked', position)
  }
})

// IPC handler for workflow window to request position highlight in main window
ipcMain.on('highlight-position', (_event, position: { x: number; y: number }) => {
  if (win && !win.isDestroyed()) {
    win.webContents.send('highlight-position', position)
  }
})

// IPC handler for main window to update clicked position
ipcMain.handle('update-clicked-position', (_event, position: { x: number; y: number } | null) => {
  currentClickedPosition = position
  return { success: true }
})

// IPC handler for workflow window to get current clicked position
ipcMain.handle('get-clicked-position', () => {
  return currentClickedPosition
})

// IPC handler for main window to update current selection
ipcMain.handle('update-selection', (_event, selection: typeof currentSelection) => {
  currentSelection = selection
  return { success: true }
})

// IPC handler for workflow window to get current selection
ipcMain.handle('get-selection', () => {
  return currentSelection
})

app.whenReady().then(() => {
  createMenu()
  startPythonBackend()
  createWindow()
})
