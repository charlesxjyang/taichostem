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

  // Diffraction view mode and cached patterns
  const [diffractionViewMode, setDiffractionViewMode] = useState<'live' | 'mean' | 'max'>('live')
  const [cachedMeanDiffraction, setCachedMeanDiffraction] = useState<MeanDiffractionResponse | null>(null)
  const [cachedMaxDiffraction, setCachedMaxDiffraction] = useState<MeanDiffractionResponse | null>(null)

  // Debounce timer ref for detector changes
  const detectorDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Display controls (placeholder state - not yet functional)
  const [logScale, setLogScale] = useState(false)
  const [contrastMin, setContrastMin] = useState(0)
  const [contrastMax, setContrastMax] = useState(100)

  // Detector controls
  const [detectorType, setDetectorType] = useState<'bf' | 'adf'>('bf')
  const [bfRadius, setBfRadius] = useState(20)
  const [adfInner, setAdfInner] = useState(20)
  const [adfOuter, setAdfOuter] = useState(60)

  // Workflow panel state
  const [workflowCollapsed, setWorkflowCollapsed] = useState(false)
  const [workflowTab, setWorkflowTab] = useState<'virtual-detector' | 'atom-detection'>('virtual-detector')

  // Atom detection controls
  const [atomThreshold, setAtomThreshold] = useState(0.3)
  const [atomMinDistance, setAtomMinDistance] = useState(5)
  const [atomGaussianRefinement, setAtomGaussianRefinement] = useState(true)
  const [atomDetectionResult, setAtomDetectionResult] = useState<string | null>(null)
  const [atomDetectionLoading, setAtomDetectionLoading] = useState(false)
  const [atomDetectionError, setAtomDetectionError] = useState<string | null>(null)
  const [atomPositions, setAtomPositions] = useState<[number, number][]>([])
  const [showAtomOverlay, setShowAtomOverlay] = useState(true)

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
   * Fetch the max diffraction pattern for the currently loaded dataset.
   */
  const fetchMaxDiffraction = useCallback(async (
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<MeanDiffractionResponse> => {
    const params = new URLSearchParams({
      log_scale: String(logScale),
      contrast_min: String(contrastMinVal),
      contrast_max: String(contrastMaxVal),
    })
    const response = await fetch(`${BACKEND_URL}/dataset/diffraction/max?${params}`)

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Fetch the virtual image for the currently loaded dataset.
   */
  const fetchVirtualImage = useCallback(async (
    type: 'bf' | 'adf' = 'bf',
    inner: number = 0,
    outer: number = 20,
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<VirtualImageResponse> => {
    const params = new URLSearchParams({
      type,
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

    // Only fetch diffraction pattern when in live mode
    if (diffractionViewMode === 'live') {
      try {
        const pattern = await fetchDiffractionPattern(imageX, imageY, logScale, contrastMin, contrastMax)
        setDiffractionPattern(pattern)
      } catch (err) {
        console.error('Failed to fetch diffraction pattern:', err)
      }
    }
  }, [virtualImage, fetchDiffractionPattern, logScale, contrastMin, contrastMax, diffractionViewMode])

  /**
   * Handle diffraction view mode change.
   * Fetches and caches mean/max patterns on first selection.
   */
  const handleDiffractionViewModeChange = useCallback(async (mode: 'live' | 'mean' | 'max') => {
    setDiffractionViewMode(mode)

    if (mode === 'mean') {
      // Fetch and cache mean pattern if not already cached
      if (!cachedMeanDiffraction) {
        try {
          const pattern = await fetchMeanDiffraction(logScale, contrastMin, contrastMax)
          setCachedMeanDiffraction(pattern)
        } catch (err) {
          console.error('Failed to fetch mean diffraction:', err)
        }
      }
    } else if (mode === 'max') {
      // Fetch and cache max pattern if not already cached
      if (!cachedMaxDiffraction) {
        try {
          const pattern = await fetchMaxDiffraction(logScale, contrastMin, contrastMax)
          setCachedMaxDiffraction(pattern)
        } catch (err) {
          console.error('Failed to fetch max diffraction:', err)
        }
      }
    }
  }, [cachedMeanDiffraction, cachedMaxDiffraction, fetchMeanDiffraction, fetchMaxDiffraction, logScale, contrastMin, contrastMax])

  /**
   * Handle file selection from Electron.
   * Probes the file first, then either loads directly or shows the picker.
   */
  const handleFileSelected = useCallback(async (filePath: string) => {
    setCurrentFile(filePath)
    setDatasetInfo(null)
    setMeanDiffraction(null)
    setVirtualImage(null)
    setCachedMeanDiffraction(null)
    setCachedMaxDiffraction(null)
    setDiffractionViewMode('live')
    setClickedPosition(null)
    setDiffractionPattern(null)
    setError(null)
    setShowPicker(false)
    setIsLoading(true)

    try {
      const probeResult = await probeFile(filePath)

      // Compute initial detector radii
      const initialInner = detectorType === 'bf' ? 0 : adfInner
      const initialOuter = detectorType === 'bf' ? bfRadius : adfOuter

      if (probeResult.type === 'single') {
        // Single datacube - load directly
        const info = await loadDataset(filePath)
        setDatasetInfo(info)
        // Fetch images in parallel
        const [diffraction, virtual] = await Promise.all([
          fetchMeanDiffraction(logScale, contrastMin, contrastMax),
          fetchVirtualImage(detectorType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
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
            fetchVirtualImage(detectorType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
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
            fetchVirtualImage(detectorType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
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
  }, [probeFile, loadDataset, fetchMeanDiffraction, fetchVirtualImage, logScale, contrastMin, contrastMax, detectorType, bfRadius, adfInner, adfOuter])

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
      // Compute detector radii
      const inner = detectorType === 'bf' ? 0 : adfInner
      const outer = detectorType === 'bf' ? bfRadius : adfOuter
      // Fetch images in parallel
      const [diffraction, virtual] = await Promise.all([
        fetchMeanDiffraction(logScale, contrastMin, contrastMax),
        fetchVirtualImage(detectorType, inner, outer, logScale, contrastMin, contrastMax),
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
  }, [pendingFilePath, loadDataset, fetchMeanDiffraction, fetchVirtualImage, logScale, contrastMin, contrastMax, detectorType, bfRadius, adfInner, adfOuter])

  /**
   * Compute detector inner/outer radii based on current settings.
   */
  const getDetectorRadii = useCallback(() => {
    if (detectorType === 'bf') {
      return { inner: 0, outer: bfRadius }
    } else {
      return { inner: adfInner, outer: adfOuter }
    }
  }, [detectorType, bfRadius, adfInner, adfOuter])

  /**
   * Refetch images when display settings change.
   */
  useEffect(() => {
    if (!datasetInfo) return

    const refetchImages = async () => {
      try {
        const { inner, outer } = getDetectorRadii()
        // Refetch virtual image with current detector settings
        const virtual = await fetchVirtualImage(detectorType, inner, outer, logScale, contrastMin, contrastMax)
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
   * Refetch virtual image when detector settings change (debounced).
   */
  useEffect(() => {
    if (!datasetInfo) return

    // Clear any pending debounce
    if (detectorDebounceRef.current) {
      clearTimeout(detectorDebounceRef.current)
    }

    // Debounce detector changes by 200ms
    detectorDebounceRef.current = setTimeout(async () => {
      try {
        const { inner, outer } = getDetectorRadii()
        const virtual = await fetchVirtualImage(detectorType, inner, outer, logScale, contrastMin, contrastMax)
        setVirtualImage(virtual)
      } catch (err) {
        console.error('Failed to refetch virtual image:', err)
      }
    }, 200)

    return () => {
      if (detectorDebounceRef.current) {
        clearTimeout(detectorDebounceRef.current)
      }
    }
  }, [detectorType, bfRadius, adfInner, adfOuter, getDetectorRadii, fetchVirtualImage, logScale, contrastMin, contrastMax, datasetInfo])

  /**
   * Handle picker modal cancel.
   */
  const handlePickerCancel = useCallback(() => {
    setShowPicker(false)
    setPendingFilePath(null)
    setHdf5Datasets([])
    setCurrentFile(null)
  }, [])

  /**
   * Save current detector configuration to a JSON file.
   */
  const handleSaveConfig = useCallback(async () => {
    const { inner, outer } = getDetectorRadii()
    const config = {
      type: detectorType,
      inner,
      outer,
    }

    const result = await window.electronAPI.saveDetectorConfig(config)
    if (!result.success && !result.canceled) {
      console.error('Failed to save config:', result.error)
    }
  }, [detectorType, getDetectorRadii])

  /**
   * Load detector configuration from a JSON file.
   */
  const handleLoadConfig = useCallback(async () => {
    const result = await window.electronAPI.loadDetectorConfig()
    if (result.success && result.config) {
      const { type, inner, outer } = result.config
      setDetectorType(type)
      if (type === 'bf') {
        setBfRadius(outer)
      } else {
        setAdfInner(inner)
        setAdfOuter(outer)
      }
    } else if (!result.canceled) {
      console.error('Failed to load config:', result.error)
    }
  }, [])

  /**
   * Export atom positions to CSV file.
   */
  const handleExportCsv = useCallback(async () => {
    if (atomPositions.length === 0) return

    // Build CSV content with header
    const header = 'index,x,y'
    const rows = atomPositions.map((pos, idx) => `${idx},${pos[0]},${pos[1]}`)
    const csvContent = [header, ...rows].join('\n')

    const result = await window.electronAPI.saveCsv(csvContent, 'atom-positions.csv')
    if (!result.success && !result.canceled) {
      console.error('Failed to export CSV:', result.error)
    }
  }, [atomPositions])

  /**
   * Run atom detection on the current virtual image.
   */
  const handleRunAtomDetection = useCallback(async () => {
    setAtomDetectionLoading(true)
    setAtomDetectionError(null)
    setAtomDetectionResult(null)
    setAtomPositions([])

    try {
      const response = await fetch(`${BACKEND_URL}/analysis/find-atoms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold: atomThreshold,
          min_distance: atomMinDistance,
          refine: atomGaussianRefinement,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const data: { count: number; positions: [number, number][] } = await response.json()
      setAtomDetectionResult(`${data.count} atoms`)
      setAtomPositions(data.positions)
      setShowAtomOverlay(true)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to detect atoms'
      setAtomDetectionError(message)
      setAtomDetectionResult(null)
      setAtomPositions([])
    } finally {
      setAtomDetectionLoading(false)
    }
  }, [atomThreshold, atomMinDistance, atomGaussianRefinement])

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

              {/* Detector Section */}
              <div className="sidebar-section">
                <h3 className="sidebar-section-title">Detector</h3>
                <div className="sidebar-control">
                  <div className="radio-group">
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="detector"
                        value="bf"
                        checked={detectorType === 'bf'}
                        onChange={() => setDetectorType('bf')}
                      />
                      BF
                    </label>
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="detector"
                        value="adf"
                        checked={detectorType === 'adf'}
                        onChange={() => setDetectorType('adf')}
                      />
                      ADF
                    </label>
                    <label className="radio-label disabled">
                      <input
                        type="radio"
                        name="detector"
                        value="custom"
                        disabled
                      />
                      Custom
                    </label>
                  </div>
                </div>

                {detectorType === 'bf' && (
                  <div className="sidebar-control">
                    <div className="slider-row">
                      <span className="slider-label-small">Radius</span>
                      <input
                        type="range"
                        min="1"
                        max="100"
                        value={bfRadius}
                        onChange={(e) => setBfRadius(Number(e.target.value))}
                        className="slider"
                      />
                      <span className="slider-value">{bfRadius}</span>
                    </div>
                  </div>
                )}

                {detectorType === 'adf' && (
                  <div className="sidebar-control">
                    <div className="contrast-sliders">
                      <div className="slider-row">
                        <span className="slider-label-small">Inner</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={adfInner}
                          onChange={(e) => {
                            const newInner = Number(e.target.value)
                            setAdfInner(newInner)
                            if (newInner >= adfOuter) {
                              setAdfOuter(Math.min(100, newInner + 1))
                            }
                          }}
                          className="slider"
                        />
                        <span className="slider-value">{adfInner}</span>
                      </div>
                      <div className="slider-row">
                        <span className="slider-label-small">Outer</span>
                        <input
                          type="range"
                          min="1"
                          max="100"
                          value={adfOuter}
                          onChange={(e) => {
                            const newOuter = Number(e.target.value)
                            setAdfOuter(newOuter)
                            if (newOuter <= adfInner) {
                              setAdfInner(Math.max(0, newOuter - 1))
                            }
                          }}
                          className="slider"
                        />
                        <span className="slider-value">{adfOuter}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="config-buttons">
                  <button className="config-button" onClick={handleSaveConfig}>
                    Save Config
                  </button>
                  <button className="config-button" onClick={handleLoadConfig}>
                    Load Config
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="main-content">
          <div className="panels-area">
            <div className="panel real-space">
              <div className="panel-header">
                <span className="panel-label">Real Space</span>
              </div>
              <div className="panel-content">
                {virtualImage && (
                  <div className="panel-image-container">
                    <img
                      ref={realSpaceImageRef}
                      className="panel-image clickable"
                      src={`data:image/png;base64,${virtualImage.image_base64}`}
                      alt="Virtual bright-field image"
                      onClick={handleRealSpaceClick}
                    />
                    <svg
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none',
                      }}
                      viewBox={`0 0 ${virtualImage.width} ${virtualImage.height}`}
                      preserveAspectRatio="xMidYMid meet"
                    >
                      {/* Atom positions overlay */}
                      {showAtomOverlay && atomPositions.map((pos, idx) => (
                        <circle
                          key={idx}
                          cx={pos[0]}
                          cy={pos[1]}
                          r={3}
                          fill="none"
                          stroke="#ff3333"
                          strokeWidth="1.5"
                        />
                      ))}
                      {/* Crosshair at clicked position */}
                      {clickedPosition && (
                        <>
                          <line
                            x1={clickedPosition.x - 8}
                            y1={clickedPosition.y}
                            x2={clickedPosition.x + 8}
                            y2={clickedPosition.y}
                            stroke="#00ff00"
                            strokeWidth="2"
                          />
                          <line
                            x1={clickedPosition.x}
                            y1={clickedPosition.y - 8}
                            x2={clickedPosition.x}
                            y2={clickedPosition.y + 8}
                            stroke="#00ff00"
                            strokeWidth="2"
                          />
                        </>
                      )}
                    </svg>
                  </div>
                )}
              </div>
            </div>
            <div className="panel-divider" />
            <div className="panel reciprocal-space">
              <div className="panel-header">
                <span className="panel-label">Reciprocal Space</span>
                <select
                  className="diffraction-mode-select"
                  value={diffractionViewMode}
                  onChange={(e) => handleDiffractionViewModeChange(e.target.value as 'live' | 'mean' | 'max')}
                >
                  <option value="live">Live</option>
                  <option value="mean">Mean</option>
                  <option value="max">Max</option>
                </select>
              </div>
              <div className="panel-content">
                {(() => {
                  // Determine which pattern to display based on view mode
                  let pattern: MeanDiffractionResponse | DiffractionPatternResponse | null = null
                  if (diffractionViewMode === 'live') {
                    pattern = diffractionPattern || meanDiffraction
                  } else if (diffractionViewMode === 'mean') {
                    pattern = cachedMeanDiffraction || meanDiffraction
                  } else if (diffractionViewMode === 'max') {
                    pattern = cachedMaxDiffraction
                  }

                  if (!pattern) return null

                  return (
                    <div className="panel-image-container">
                      <img
                        className="panel-image"
                        src={`data:image/png;base64,${pattern.image_base64}`}
                        alt={diffractionPattern ? 'Diffraction pattern' : 'Mean diffraction pattern'}
                      />
                      <svg
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          pointerEvents: 'none',
                        }}
                        viewBox={`0 0 ${pattern.width} ${pattern.height}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {detectorType === 'bf' ? (
                          // BF: filled semi-transparent circle
                          <circle
                            cx={pattern.width / 2}
                            cy={pattern.height / 2}
                            r={bfRadius}
                            fill="rgba(0, 162, 255, 0.3)"
                            stroke="rgba(0, 162, 255, 0.8)"
                            strokeWidth="1"
                          />
                        ) : (
                          // ADF: semi-transparent annulus (outer circle with inner hole)
                          <>
                            <defs>
                              <mask id="annulus-mask">
                                <rect width="100%" height="100%" fill="white" />
                                <circle
                                  cx={pattern.width / 2}
                                  cy={pattern.height / 2}
                                  r={adfInner}
                                  fill="black"
                                />
                              </mask>
                            </defs>
                            <circle
                              cx={pattern.width / 2}
                              cy={pattern.height / 2}
                              r={adfOuter}
                              fill="rgba(255, 162, 0, 0.3)"
                              mask="url(#annulus-mask)"
                            />
                            <circle
                              cx={pattern.width / 2}
                              cy={pattern.height / 2}
                              r={adfOuter}
                              fill="none"
                              stroke="rgba(255, 162, 0, 0.8)"
                              strokeWidth="1"
                            />
                            <circle
                              cx={pattern.width / 2}
                              cy={pattern.height / 2}
                              r={adfInner}
                              fill="none"
                              stroke="rgba(255, 162, 0, 0.8)"
                              strokeWidth="1"
                            />
                          </>
                        )}
                      </svg>
                    </div>
                  )
                })()}
              </div>
            </div>
          </div>

          {/* Workflow Panel */}
          <div className={`workflow-panel ${workflowCollapsed ? 'collapsed' : ''}`}>
            <div className="workflow-header">
              <div className="workflow-tabs">
                <button
                  className={`workflow-tab ${workflowTab === 'virtual-detector' ? 'active' : ''}`}
                  onClick={() => setWorkflowTab('virtual-detector')}
                >
                  Virtual Detector
                </button>
                <button
                  className={`workflow-tab ${workflowTab === 'atom-detection' ? 'active' : ''}`}
                  onClick={() => setWorkflowTab('atom-detection')}
                >
                  Atom Detection
                </button>
              </div>
              <button
                className="workflow-toggle"
                onClick={() => setWorkflowCollapsed(!workflowCollapsed)}
                title={workflowCollapsed ? 'Expand panel' : 'Collapse panel'}
              >
                {workflowCollapsed ? '▲' : '▼'}
              </button>
            </div>
            {!workflowCollapsed && (
              <div className="workflow-content">
                {workflowTab === 'virtual-detector' ? (
                  <div className="workflow-tab-content">
                    <span className="workflow-placeholder">Detector controls are in the sidebar</span>
                  </div>
                ) : (
                  <div className="workflow-tab-content atom-detection-content">
                    <div className="atom-detection-controls">
                      <div className="atom-control">
                        <label className="atom-control-label">Threshold</label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={atomThreshold}
                          onChange={(e) => setAtomThreshold(Number(e.target.value))}
                          className="slider"
                        />
                        <span className="atom-control-value">{atomThreshold.toFixed(2)}</span>
                      </div>
                      <div className="atom-control">
                        <label className="atom-control-label">Min distance</label>
                        <input
                          type="range"
                          min="1"
                          max="20"
                          step="1"
                          value={atomMinDistance}
                          onChange={(e) => setAtomMinDistance(Number(e.target.value))}
                          className="slider"
                        />
                        <span className="atom-control-value">{atomMinDistance}px</span>
                      </div>
                      <div className="atom-control">
                        <label className="checkbox-label">
                          <input
                            type="checkbox"
                            checked={atomGaussianRefinement}
                            onChange={(e) => setAtomGaussianRefinement(e.target.checked)}
                          />
                          Gaussian refinement
                        </label>
                      </div>
                    </div>
                    <div className="atom-detection-actions">
                      <button
                        className="atom-run-button"
                        onClick={handleRunAtomDetection}
                        disabled={atomDetectionLoading || !virtualImage}
                      >
                        {atomDetectionLoading ? (
                          <span className="loading-spinner" />
                        ) : (
                          'Run'
                        )}
                      </button>
                      <button
                        className={`atom-toggle-button ${showAtomOverlay ? 'active' : ''}`}
                        onClick={() => setShowAtomOverlay(!showAtomOverlay)}
                        disabled={atomPositions.length === 0}
                      >
                        {showAtomOverlay ? 'Hide Overlay' : 'Show Overlay'}
                      </button>
                      <button
                        className="atom-export-button"
                        onClick={handleExportCsv}
                        disabled={atomPositions.length === 0}
                      >
                        Export CSV
                      </button>
                      <span className={`atom-results ${atomDetectionError ? 'error' : ''}`}>
                        Results: {atomDetectionError ?? atomDetectionResult ?? '—'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
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
