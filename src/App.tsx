import { useState, useEffect, useCallback, useRef } from 'react'
import { BackendStatus } from './components/BackendStatus'
import { DatasetPickerModal } from './components/DatasetPickerModal'
import type {
  ProbeResponse,
  HDF5DatasetInfo,
  DatasetInfo,
  MeanDiffractionResponse,
  VirtualImageResponse,
  DiffractionPatternResponse,
} from './types/dataset'

const BACKEND_URL = 'http://127.0.0.1:8000'

function App() {
  const [currentFile, setCurrentFile] = useState<string | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Modal state
  const [showPicker, setShowPicker] = useState(false)
  const [pendingFilePath, setPendingFilePath] = useState<string | null>(null)
  const [hdf5Datasets, setHdf5Datasets] = useState<HDF5DatasetInfo[]>([])

  // Mean diffraction image
  const [meanDiffraction, setMeanDiffraction] = useState<MeanDiffractionResponse | null>(null)

  // Virtual bright-field image
  const [virtualImage, setVirtualImage] = useState<VirtualImageResponse | null>(null)

  // Clicked position in real space (image coordinates)
  const [clickedPosition, setClickedPosition] = useState<{ x: number; y: number } | null>(null)
  const realSpaceImageRef = useRef<HTMLImageElement>(null)

  // Diffraction pattern at clicked position
  const [diffractionPattern, setDiffractionPattern] = useState<DiffractionPatternResponse | null>(null)

  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Display controls (placeholder state - not yet functional)
  const [logScale, setLogScale] = useState(false)
  const [contrastMin, setContrastMin] = useState(0)
  const [contrastMax, setContrastMax] = useState(100)

  /**
   * Probe a file to determine its structure.
   */
  const probeFile = useCallback(async (filePath: string): Promise<ProbeResponse> => {
    const response = await fetch(`${BACKEND_URL}/dataset/probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: filePath }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Load a dataset, optionally with a specific internal path for HDF5 files.
   */
  const loadDataset = useCallback(async (
    filePath: string,
    datasetPath?: string
  ): Promise<DatasetInfo> => {
    const body: { path: string; dataset_path?: string } = { path: filePath }
    if (datasetPath) {
      body.dataset_path = datasetPath
    }

    const response = await fetch(`${BACKEND_URL}/dataset/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    const data = await response.json()
    return {
      ...data,
      filePath,
    }
  }, [])

  /**
   * Fetch the mean diffraction pattern for the currently loaded dataset.
   */
  const fetchMeanDiffraction = useCallback(async (
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<MeanDiffractionResponse> => {
    const params = new URLSearchParams({
      log_scale: String(logScale),
      contrast_min: String(contrastMinVal),
      contrast_max: String(contrastMaxVal),
    })
    const response = await fetch(`${BACKEND_URL}/dataset/diffraction/mean?${params}`)

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Fetch the virtual bright-field image for the currently loaded dataset.
   */
  const fetchVirtualImage = useCallback(async (
    inner: number = 0,
    outer: number = 20,
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<VirtualImageResponse> => {
    const params = new URLSearchParams({
      inner: String(inner),
      outer: String(outer),
      log_scale: String(logScale),
      contrast_min: String(contrastMinVal),
      contrast_max: String(contrastMaxVal),
    })
    const response = await fetch(`${BACKEND_URL}/dataset/virtual-image?${params}`)

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Fetch the diffraction pattern at a specific scan position.
   */
  const fetchDiffractionPattern = useCallback(async (
    x: number,
    y: number,
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<DiffractionPatternResponse> => {
    const params = new URLSearchParams({
      x: String(x),
      y: String(y),
      log_scale: String(logScale),
      contrast_min: String(contrastMinVal),
      contrast_max: String(contrastMaxVal),
    })
    const response = await fetch(`${BACKEND_URL}/dataset/diffraction?${params}`)

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Handle click on the real space image.
   * Converts click position to image coordinates and fetches diffraction pattern.
   */
  const handleRealSpaceClick = useCallback(async (event: React.MouseEvent<HTMLImageElement>) => {
    const img = realSpaceImageRef.current
    if (!img || !virtualImage) return

    const rect = img.getBoundingClientRect()

    // Calculate the actual displayed size of the image (accounting for object-fit: contain)
    const imgAspect = virtualImage.width / virtualImage.height
    const containerAspect = rect.width / rect.height

    let displayedWidth: number
    let displayedHeight: number
    let offsetX: number
    let offsetY: number

    if (imgAspect > containerAspect) {
      // Image is wider than container - width is limiting
      displayedWidth = rect.width
      displayedHeight = rect.width / imgAspect
      offsetX = 0
      offsetY = (rect.height - displayedHeight) / 2
    } else {
      // Image is taller than container - height is limiting
      displayedHeight = rect.height
      displayedWidth = rect.height * imgAspect
      offsetX = (rect.width - displayedWidth) / 2
      offsetY = 0
    }

    // Get click position relative to the image element
    const clickX = event.clientX - rect.left
    const clickY = event.clientY - rect.top

    // Check if click is within the actual image area
    if (clickX < offsetX || clickX > offsetX + displayedWidth ||
        clickY < offsetY || clickY > offsetY + displayedHeight) {
      return
    }

    // Convert to image coordinates
    const imageX = Math.round(((clickX - offsetX) / displayedWidth) * virtualImage.width)
    const imageY = Math.round(((clickY - offsetY) / displayedHeight) * virtualImage.height)

    setClickedPosition({ x: imageX, y: imageY })

    // Fetch diffraction pattern at clicked position
    try {
      const pattern = await fetchDiffractionPattern(imageX, imageY, logScale, contrastMin, contrastMax)
      setDiffractionPattern(pattern)
    } catch (err) {
      console.error('Failed to fetch diffraction pattern:', err)
    }
  }, [virtualImage, fetchDiffractionPattern, logScale, contrastMin, contrastMax])

  /**
   * Handle file selection from Electron.
   * Probes the file first, then either loads directly or shows the picker.
   */
  const handleFileSelected = useCallback(async (filePath: string) => {
    setCurrentFile(filePath)
    setDatasetInfo(null)
    setMeanDiffraction(null)
    setVirtualImage(null)
    setClickedPosition(null)
    setDiffractionPattern(null)
    setError(null)
    setShowPicker(false)
    setIsLoading(true)

    try {
      const probeResult = await probeFile(filePath)

      if (probeResult.type === 'single') {
        // Single datacube - load directly
        const info = await loadDataset(filePath)
        setDatasetInfo(info)
        // Fetch images in parallel
        const [diffraction, virtual] = await Promise.all([
          fetchMeanDiffraction(logScale, contrastMin, contrastMax),
          fetchVirtualImage(0, 20, logScale, contrastMin, contrastMax),
        ])
        setMeanDiffraction(diffraction)
        setVirtualImage(virtual)
      } else if (probeResult.type === 'hdf5_tree') {
        // HDF5 file - check if we need to show picker
        const datasets4d = probeResult.datasets.filter(d => d.is_4d)

        if (datasets4d.length === 1) {
          // Only one 4D dataset - load it directly
          const info = await loadDataset(filePath, datasets4d[0].path)
          setDatasetInfo(info)
          // Fetch images in parallel
          const [diffraction, virtual] = await Promise.all([
            fetchMeanDiffraction(logScale, contrastMin, contrastMax),
            fetchVirtualImage(0, 20, logScale, contrastMin, contrastMax),
          ])
          setMeanDiffraction(diffraction)
          setVirtualImage(virtual)
        } else if (probeResult.datasets.length === 1) {
          // Only one dataset total - load it directly
          const info = await loadDataset(filePath, probeResult.datasets[0].path)
          setDatasetInfo(info)
          // Fetch images in parallel
          const [diffraction, virtual] = await Promise.all([
            fetchMeanDiffraction(logScale, contrastMin, contrastMax),
            fetchVirtualImage(0, 20, logScale, contrastMin, contrastMax),
          ])
          setMeanDiffraction(diffraction)
          setVirtualImage(virtual)
        } else if (probeResult.datasets.length > 1) {
          // Multiple datasets - show picker
          setPendingFilePath(filePath)
          setHdf5Datasets(probeResult.datasets)
          setShowPicker(true)
        } else {
          throw new Error('No datasets found in HDF5 file')
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset')
    } finally {
      setIsLoading(false)
    }
  }, [probeFile, loadDataset, fetchMeanDiffraction, fetchVirtualImage, logScale, contrastMin, contrastMax])

  /**
   * Handle dataset selection from the picker modal.
   */
  const handleDatasetSelect = useCallback(async (datasetPath: string) => {
    if (!pendingFilePath) return

    setShowPicker(false)
    setIsLoading(true)
    setError(null)
    setMeanDiffraction(null)
    setVirtualImage(null)

    try {
      const info = await loadDataset(pendingFilePath, datasetPath)
      setDatasetInfo(info)
      // Fetch images in parallel
      const [diffraction, virtual] = await Promise.all([
        fetchMeanDiffraction(logScale, contrastMin, contrastMax),
        fetchVirtualImage(0, 20, logScale, contrastMin, contrastMax),
      ])
      setMeanDiffraction(diffraction)
      setVirtualImage(virtual)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset')
    } finally {
      setIsLoading(false)
      setPendingFilePath(null)
      setHdf5Datasets([])
    }
  }, [pendingFilePath, loadDataset, fetchMeanDiffraction, fetchVirtualImage, logScale, contrastMin, contrastMax])

  /**
   * Refetch images when display settings change.
   */
  useEffect(() => {
    if (!datasetInfo) return

    const refetchImages = async () => {
      try {
        // Refetch virtual image
        const virtual = await fetchVirtualImage(0, 20, logScale, contrastMin, contrastMax)
        setVirtualImage(virtual)

        // Refetch diffraction (either mean or at clicked position)
        if (clickedPosition) {
          const pattern = await fetchDiffractionPattern(
            clickedPosition.x,
            clickedPosition.y,
            logScale,
            contrastMin,
            contrastMax
          )
          setDiffractionPattern(pattern)
        } else {
          const mean = await fetchMeanDiffraction(logScale, contrastMin, contrastMax)
          setMeanDiffraction(mean)
        }
      } catch (err) {
        console.error('Failed to refetch images:', err)
      }
    }

    refetchImages()
  }, [logScale, contrastMin, contrastMax]) // Only refetch when display settings change

  /**
   * Handle picker modal cancel.
   */
  const handlePickerCancel = useCallback(() => {
    setShowPicker(false)
    setPendingFilePath(null)
    setHdf5Datasets([])
    setCurrentFile(null)
  }, [])

  useEffect(() => {
    window.electronAPI.onFileSelected(handleFileSelected)

    return () => {
      window.electronAPI.removeFileSelectedListener()
    }
  }, [handleFileSelected])

  const filename = currentFile ? currentFile.split('/').pop() ?? currentFile : null

  const getStatusText = (): string => {
    if (!filename) return 'No dataset loaded'
    if (isLoading) return `Loading: ${filename}...`
    if (showPicker) return `Selecting dataset from: ${filename}`
    if (error) return `Error: ${error}`
    if (datasetInfo) {
      const shapeStr = datasetInfo.shape.join('x')
      // Show full path for HDF5 files: filename.h5:/internal/path (shape)
      if (datasetInfo.dataset_path) {
        return `Loaded: ${filename}:${datasetInfo.dataset_path} (${shapeStr})`
      }
      return `Loaded: ${filename} (${shapeStr})`
    }
    return filename
  }

  const getStatusClass = (): string => {
    if (error) return 'status-bar-file status-error'
    if (isLoading || showPicker) return 'status-bar-file status-loading'
    if (datasetInfo) return 'status-bar-file status-success'
    return 'status-bar-file'
  }

  return (
    <div className="app">
      <div className="menu-bar" />
      <div className="app-body">
        {/* Collapsible Sidebar */}
        <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? '›' : '‹'}
          </button>
          {!sidebarCollapsed && (
            <div className="sidebar-content">
              <div className="sidebar-section">
                <h3 className="sidebar-section-title">Display</h3>
                <div className="sidebar-control">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={logScale}
                      onChange={(e) => setLogScale(e.target.checked)}
                    />
                    Log scale
                  </label>
                </div>
                <div className="sidebar-control">
                  <label className="slider-label">Contrast</label>
                  <div className="contrast-sliders">
                    <div className="slider-row">
                      <span className="slider-label-small">Min</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={contrastMin}
                        onChange={(e) => {
                          const newMin = Number(e.target.value)
                          setContrastMin(newMin)
                          // Ensure max stays above min
                          if (newMin >= contrastMax) {
                            setContrastMax(Math.min(100, newMin + 1))
                          }
                        }}
                        className="slider"
                      />
                      <span className="slider-value">{contrastMin}</span>
                    </div>
                    <div className="slider-row">
                      <span className="slider-label-small">Max</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={contrastMax}
                        onChange={(e) => {
                          const newMax = Number(e.target.value)
                          setContrastMax(newMax)
                          // Ensure min stays below max
                          if (newMax <= contrastMin) {
                            setContrastMin(Math.max(0, newMax - 1))
                          }
                        }}
                        className="slider"
                      />
                      <span className="slider-value">{contrastMax}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="main-content">
          <div className="panel real-space">
          <span className="panel-label">Real Space</span>
          {virtualImage && (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <img
                ref={realSpaceImageRef}
                src={`data:image/png;base64,${virtualImage.image_base64}`}
                alt="Virtual bright-field image"
                onClick={handleRealSpaceClick}
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                  display: 'block',
                  cursor: 'crosshair',
                }}
              />
              {clickedPosition && (
                <svg
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                  }}
                >
                  {/* Crosshair - position as percentage of image dimensions */}
                  <line
                    x1={`${(clickedPosition.x / virtualImage.width) * 100 - 2}%`}
                    y1={`${(clickedPosition.y / virtualImage.height) * 100}%`}
                    x2={`${(clickedPosition.x / virtualImage.width) * 100 + 2}%`}
                    y2={`${(clickedPosition.y / virtualImage.height) * 100}%`}
                    stroke="red"
                    strokeWidth="2"
                  />
                  <line
                    x1={`${(clickedPosition.x / virtualImage.width) * 100}%`}
                    y1={`${(clickedPosition.y / virtualImage.height) * 100 - 2}%`}
                    x2={`${(clickedPosition.x / virtualImage.width) * 100}%`}
                    y2={`${(clickedPosition.y / virtualImage.height) * 100 + 2}%`}
                    stroke="red"
                    strokeWidth="2"
                  />
                </svg>
              )}
            </div>
          )}
        </div>
        <div className="panel-divider" />
        <div className="panel reciprocal-space">
          <span className="panel-label">Reciprocal Space</span>
          {(diffractionPattern || meanDiffraction) && (
            <img
              src={`data:image/png;base64,${(diffractionPattern || meanDiffraction)!.image_base64}`}
              alt={diffractionPattern ? 'Diffraction pattern' : 'Mean diffraction pattern'}
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
              }}
            />
          )}
          </div>
        </div>
      </div>
      <div className="status-bar">
        <span className={getStatusClass()}>
          {getStatusText()}
        </span>
        {clickedPosition && (
          <span className="status-bar-position">
            Position: ({clickedPosition.x}, {clickedPosition.y})
          </span>
        )}
        <BackendStatus className="status-bar-backend" />
      </div>

      <DatasetPickerModal
        isOpen={showPicker}
        datasets={hdf5Datasets}
        filename={filename ?? ''}
        onSelect={handleDatasetSelect}
        onCancel={handlePickerCancel}
      />
    </div>
  )
}

export default App
