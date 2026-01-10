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
  SelectionTool,
  SelectionGeometry,
  RegionDiffractionResponse,
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

  // Region selection state
  const [selectionTool, setSelectionTool] = useState<SelectionTool>('rectangle')
  const [selection, setSelection] = useState<SelectionGeometry | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawingPoints, setDrawingPoints] = useState<[number, number][]>([])
  const [regionDiffraction, setRegionDiffraction] = useState<RegionDiffractionResponse | null>(null)

  // Debounce timer ref for detector changes
  const detectorDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Display controls (placeholder state - not yet functional)
  const [logScale, setLogScale] = useState(false)
  const [contrastMin, setContrastMin] = useState(0)
  const [contrastMax, setContrastMax] = useState(100)

  // Detector controls
  const [detectorType, setDetectorType] = useState<'none' | 'bf' | 'abf' | 'adf'>('bf')
  const [bfRadius, setBfRadius] = useState(20)
  const [abfInner, setAbfInner] = useState(22)
  const [abfOuter, setAbfOuter] = useState(35)
  const [adfInner, setAdfInner] = useState(40)
  const [adfOuter, setAdfOuter] = useState(80)

  // Detector overlay settings (advanced, in workflow panel)
  const [showDetectorOverlay, setShowDetectorOverlay] = useState(true)
  const [overlayOpacity, setOverlayOpacity] = useState(50)
  const [overlayColor, setOverlayColor] = useState<'auto' | 'yellow' | 'cyan' | 'green' | 'orange' | 'magenta'>('auto')
  const [centerOffsetX, setCenterOffsetX] = useState(0)
  const [centerOffsetY, setCenterOffsetY] = useState(0)
  const [pickingCenter, setPickingCenter] = useState(false)

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

  // Filter hot pixels dialog state
  const [showFilterHotPixelsDialog, setShowFilterHotPixelsDialog] = useState(false)
  const [filterHotPixelsThresh, setFilterHotPixelsThresh] = useState(8)
  const [filterHotPixelsLoading, setFilterHotPixelsLoading] = useState(false)

  // Calibration state
  const [calibrationCollapsed, setCalibrationCollapsed] = useState(false)
  const [qPixelSize, setQPixelSize] = useState(1.0)
  const [qPixelUnits, setQPixelUnits] = useState('1/Å')
  const [rPixelSize, setRPixelSize] = useState(1.0)
  const [rPixelUnits, setRPixelUnits] = useState('Å')

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
    type: 'bf' | 'abf' | 'adf' = 'bf',
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
   * Fetch region-based mean or max diffraction pattern.
   */
  const fetchRegionDiffraction = useCallback(async (
    mode: 'mean' | 'max',
    regionType: 'rectangle' | 'ellipse' | 'polygon',
    points: [number, number][],
    logScale: boolean = false,
    contrastMinVal: number = 0,
    contrastMaxVal: number = 100
  ): Promise<RegionDiffractionResponse> => {
    const response = await fetch(`${BACKEND_URL}/dataset/diffraction/region`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        region_type: regionType,
        points,
        log_scale: logScale,
        contrast_min: contrastMinVal,
        contrast_max: contrastMaxVal,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }, [])

  /**
   * Fetch calibration values from the backend.
   */
  const fetchCalibration = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/dataset/calibration`)
      if (response.ok) {
        const data = await response.json()
        setQPixelSize(data.q_pixel_size)
        setQPixelUnits(data.q_pixel_units)
        setRPixelSize(data.r_pixel_size)
        setRPixelUnits(data.r_pixel_units)
      }
    } catch (err) {
      console.error('Failed to fetch calibration:', err)
    }
  }, [])

  /**
   * Update calibration values on the backend.
   */
  const updateCalibration = useCallback(async (updates: {
    q_pixel_size?: number
    q_pixel_units?: string
    r_pixel_size?: number
    r_pixel_units?: string
  }) => {
    try {
      const response = await fetch(`${BACKEND_URL}/dataset/calibration`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      })
      if (!response.ok) {
        console.error('Failed to update calibration')
      }
    } catch (err) {
      console.error('Failed to update calibration:', err)
    }
  }, [])

  /**
   * Convert screen coordinates to image coordinates.
   */
  const screenToImageCoords = useCallback((event: React.MouseEvent<HTMLElement>): [number, number] | null => {
    const img = realSpaceImageRef.current
    if (!img || !virtualImage) return null

    const rect = img.getBoundingClientRect()

    // Calculate the actual displayed size of the image (accounting for object-fit: contain)
    const imgAspect = virtualImage.width / virtualImage.height
    const containerAspect = rect.width / rect.height

    let displayedWidth: number
    let displayedHeight: number
    let offsetX: number
    let offsetY: number

    if (imgAspect > containerAspect) {
      displayedWidth = rect.width
      displayedHeight = rect.width / imgAspect
      offsetX = 0
      offsetY = (rect.height - displayedHeight) / 2
    } else {
      displayedHeight = rect.height
      displayedWidth = rect.height * imgAspect
      offsetX = (rect.width - displayedWidth) / 2
      offsetY = 0
    }

    const clickX = event.clientX - rect.left
    const clickY = event.clientY - rect.top

    // Check if click is within the actual image area
    if (clickX < offsetX || clickX > offsetX + displayedWidth ||
        clickY < offsetY || clickY > offsetY + displayedHeight) {
      return null
    }

    const imageX = ((clickX - offsetX) / displayedWidth) * virtualImage.width
    const imageY = ((clickY - offsetY) / displayedHeight) * virtualImage.height

    return [imageX, imageY]
  }, [virtualImage])

  /**
   * Handle mouse down on the real space image.
   * Starts drawing selection or fetches diffraction pattern.
   */
  const handleRealSpaceMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const coords = screenToImageCoords(event)
    if (!coords) return

    const [imageX, imageY] = coords

    // If in Mean/Max mode with a selection tool, start drawing
    if (diffractionViewMode !== 'live' && selectionTool) {
      setIsDrawing(true)
      setDrawingPoints([[imageX, imageY]])
      event.preventDefault()
    }
  }, [screenToImageCoords, diffractionViewMode, selectionTool])

  /**
   * Handle mouse move on the real space image.
   * Updates drawing preview.
   */
  const handleRealSpaceMouseMove = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing) return

    const coords = screenToImageCoords(event)
    if (!coords) return

    const [imageX, imageY] = coords

    if (selectionTool === 'lasso') {
      // For lasso, add point to the path
      setDrawingPoints(prev => [...prev, [imageX, imageY]])
    } else if (selectionTool === 'ellipse') {
      // For circle, calculate radius as distance from center to current point
      const [cx, cy] = drawingPoints[0]
      const radius = Math.sqrt((imageX - cx) ** 2 + (imageY - cy) ** 2)
      // Store edge point at angle 0 (to the right) for consistent radius
      setDrawingPoints(prev => [prev[0], [cx + radius, cy]])
    } else {
      // For rectangle, update the second point
      setDrawingPoints(prev => [prev[0], [imageX, imageY]])
    }
  }, [isDrawing, screenToImageCoords, selectionTool, drawingPoints])

  /**
   * Handle mouse up on the real space image.
   * Completes drawing and triggers region diffraction fetch.
   */
  const handleRealSpaceMouseUp = useCallback(async (event: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing) return

    const coords = screenToImageCoords(event)
    if (!coords) {
      setIsDrawing(false)
      setDrawingPoints([])
      return
    }

    const [imageX, imageY] = coords
    setIsDrawing(false)

    // Finalize the selection
    let finalPoints: [number, number][] = []
    let regionType: 'rectangle' | 'ellipse' | 'polygon' = 'rectangle'

    if (selectionTool === 'rectangle') {
      if (drawingPoints.length >= 1) {
        finalPoints = [drawingPoints[0], [imageX, imageY]]
        regionType = 'rectangle'
      }
    } else if (selectionTool === 'ellipse') {
      // For circle, calculate radius and store edge point
      if (drawingPoints.length >= 1) {
        const [cx, cy] = drawingPoints[0]
        const radius = Math.sqrt((imageX - cx) ** 2 + (imageY - cy) ** 2)
        finalPoints = [drawingPoints[0], [cx + radius, cy]]
        regionType = 'ellipse'
      }
    } else if (selectionTool === 'lasso') {
      finalPoints = [...drawingPoints, [imageX, imageY]]
      regionType = 'polygon'
    }

    setDrawingPoints([])

    // Validate selection has minimum size
    if (finalPoints.length < 2) return
    if (selectionTool === 'lasso' && finalPoints.length < 3) return

    // For rectangle/ellipse, check minimum size
    if (selectionTool !== 'lasso') {
      const dx = Math.abs(finalPoints[1][0] - finalPoints[0][0])
      const dy = Math.abs(finalPoints[1][1] - finalPoints[0][1])
      if (dx < 2 && dy < 2) return
    }

    // Save the selection
    const newSelection: SelectionGeometry = {
      type: regionType,
      points: finalPoints,
    }
    setSelection(newSelection)

    // Fetch region diffraction
    const mode = diffractionViewMode as 'mean' | 'max'
    try {
      const result = await fetchRegionDiffraction(
        mode,
        regionType,
        finalPoints,
        logScale,
        contrastMin,
        contrastMax
      )
      setRegionDiffraction(result)
    } catch (err) {
      console.error('Failed to fetch region diffraction:', err)
    }
  }, [isDrawing, screenToImageCoords, selectionTool, drawingPoints, diffractionViewMode, fetchRegionDiffraction, logScale, contrastMin, contrastMax])

  /**
   * Handle click on the real space image.
   * In live mode, fetches diffraction pattern at clicked position.
   */
  const handleRealSpaceClick = useCallback(async (event: React.MouseEvent<HTMLImageElement>) => {
    // Only handle click for live mode (drawing is handled by mouse up)
    if (diffractionViewMode !== 'live') return

    const coords = screenToImageCoords(event)
    if (!coords) return

    const [imageX, imageY] = coords.map(Math.round)

    setClickedPosition({ x: imageX, y: imageY })

    try {
      const pattern = await fetchDiffractionPattern(imageX, imageY, logScale, contrastMin, contrastMax)
      setDiffractionPattern(pattern)
    } catch (err) {
      console.error('Failed to fetch diffraction pattern:', err)
    }
  }, [screenToImageCoords, diffractionViewMode, fetchDiffractionPattern, logScale, contrastMin, contrastMax])

  /**
   * Clear the current selection.
   */
  const handleClearSelection = useCallback(() => {
    setSelection(null)
    setRegionDiffraction(null)
  }, [])

  /**
   * Handle keyboard shortcuts for selection tools.
   */
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && selection) {
        handleClearSelection()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selection, handleClearSelection])

  /**
   * Handle diffraction view mode change.
   * Fetches and caches mean/max patterns on first selection.
   * Restores region selection when switching between Mean/Max modes.
   */
  const handleDiffractionViewModeChange = useCallback(async (mode: 'live' | 'mean' | 'max') => {
    setDiffractionViewMode(mode)

    if (mode === 'live') {
      // Clear region diffraction when switching to live mode
      setRegionDiffraction(null)
      return
    }

    // For Mean/Max mode, check if there's a selection and fetch region diffraction
    if (selection) {
      try {
        const result = await fetchRegionDiffraction(
          mode,
          selection.type,
          selection.points,
          logScale,
          contrastMin,
          contrastMax
        )
        setRegionDiffraction(result)
      } catch (err) {
        console.error('Failed to fetch region diffraction:', err)
        setRegionDiffraction(null)
      }
    } else {
      // No selection - fetch whole-image mean/max
      setRegionDiffraction(null)
      if (mode === 'mean') {
        if (!cachedMeanDiffraction) {
          try {
            const pattern = await fetchMeanDiffraction(logScale, contrastMin, contrastMax)
            setCachedMeanDiffraction(pattern)
          } catch (err) {
            console.error('Failed to fetch mean diffraction:', err)
          }
        }
      } else if (mode === 'max') {
        if (!cachedMaxDiffraction) {
          try {
            const pattern = await fetchMaxDiffraction(logScale, contrastMin, contrastMax)
            setCachedMaxDiffraction(pattern)
          } catch (err) {
            console.error('Failed to fetch max diffraction:', err)
          }
        }
      }
    }
  }, [selection, cachedMeanDiffraction, cachedMaxDiffraction, fetchMeanDiffraction, fetchMaxDiffraction, fetchRegionDiffraction, logScale, contrastMin, contrastMax])

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
    setSelection(null)
    setRegionDiffraction(null)
    setError(null)
    setShowPicker(false)
    setIsLoading(true)

    try {
      const probeResult = await probeFile(filePath)

      // Compute initial detector type and radii
      const effectiveType = detectorType === 'none' ? 'bf' : detectorType
      const initialInner = detectorType === 'bf' || detectorType === 'none' ? 0 :
                           detectorType === 'abf' ? abfInner : adfInner
      const initialOuter = detectorType === 'bf' || detectorType === 'none' ? bfRadius :
                           detectorType === 'abf' ? abfOuter : adfOuter

      if (probeResult.type === 'single') {
        // Single datacube - load directly
        const info = await loadDataset(filePath)
        setDatasetInfo(info)
        // Fetch images in parallel
        const [diffraction, virtual] = await Promise.all([
          fetchMeanDiffraction(logScale, contrastMin, contrastMax),
          fetchVirtualImage(effectiveType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
        ])
        setMeanDiffraction(diffraction)
        setVirtualImage(virtual)
        fetchCalibration()
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
            fetchVirtualImage(effectiveType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
          ])
          setMeanDiffraction(diffraction)
          setVirtualImage(virtual)
          fetchCalibration()
        } else if (probeResult.datasets.length === 1) {
          // Only one dataset total - load it directly
          const info = await loadDataset(filePath, probeResult.datasets[0].path)
          setDatasetInfo(info)
          // Fetch images in parallel
          const [diffraction, virtual] = await Promise.all([
            fetchMeanDiffraction(logScale, contrastMin, contrastMax),
            fetchVirtualImage(effectiveType, initialInner, initialOuter, logScale, contrastMin, contrastMax),
          ])
          setMeanDiffraction(diffraction)
          setVirtualImage(virtual)
          fetchCalibration()
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
  }, [probeFile, loadDataset, fetchMeanDiffraction, fetchVirtualImage, fetchCalibration, logScale, contrastMin, contrastMax, detectorType, bfRadius, abfInner, abfOuter, adfInner, adfOuter])

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
      // Compute detector type and radii
      const effectiveType = detectorType === 'none' ? 'bf' : detectorType
      const inner = detectorType === 'bf' || detectorType === 'none' ? 0 :
                    detectorType === 'abf' ? abfInner : adfInner
      const outer = detectorType === 'bf' || detectorType === 'none' ? bfRadius :
                    detectorType === 'abf' ? abfOuter : adfOuter
      // Fetch images in parallel
      const [diffraction, virtual] = await Promise.all([
        fetchMeanDiffraction(logScale, contrastMin, contrastMax),
        fetchVirtualImage(effectiveType, inner, outer, logScale, contrastMin, contrastMax),
      ])
      setMeanDiffraction(diffraction)
      setVirtualImage(virtual)
      fetchCalibration()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset')
    } finally {
      setIsLoading(false)
      setPendingFilePath(null)
      setHdf5Datasets([])
    }
  }, [pendingFilePath, loadDataset, fetchMeanDiffraction, fetchVirtualImage, fetchCalibration, logScale, contrastMin, contrastMax, detectorType, bfRadius, abfInner, abfOuter, adfInner, adfOuter])

  /**
   * Compute detector inner/outer radii based on current settings.
   */
  const getDetectorRadii = useCallback(() => {
    if (detectorType === 'none' || detectorType === 'bf') {
      return { inner: 0, outer: bfRadius }
    } else if (detectorType === 'abf') {
      return { inner: abfInner, outer: abfOuter }
    } else {
      return { inner: adfInner, outer: adfOuter }
    }
  }, [detectorType, bfRadius, abfInner, abfOuter, adfInner, adfOuter])

  /**
   * Get the effective detector type for API calls (maps 'none' to 'bf').
   */
  const getEffectiveDetectorType = useCallback((): 'bf' | 'abf' | 'adf' => {
    return detectorType === 'none' ? 'bf' : detectorType
  }, [detectorType])

  /**
   * Get the overlay fill color based on detector type and user preference.
   */
  const getOverlayColor = useCallback(() => {
    if (overlayColor !== 'auto') {
      const colorMap: Record<string, { fill: string; stroke: string }> = {
        yellow: { fill: 'rgba(255, 220, 0, VAR)', stroke: 'rgba(255, 220, 0, 0.8)' },
        cyan: { fill: 'rgba(0, 162, 255, VAR)', stroke: 'rgba(0, 162, 255, 0.8)' },
        green: { fill: 'rgba(0, 200, 150, VAR)', stroke: 'rgba(0, 200, 150, 0.8)' },
        orange: { fill: 'rgba(255, 162, 0, VAR)', stroke: 'rgba(255, 162, 0, 0.8)' },
        magenta: { fill: 'rgba(255, 0, 200, VAR)', stroke: 'rgba(255, 0, 200, 0.8)' },
      }
      const opacity = overlayOpacity / 100 * 0.5 // Scale to max 0.5 fill opacity
      return {
        fill: colorMap[overlayColor].fill.replace('VAR', String(opacity)),
        stroke: colorMap[overlayColor].stroke,
      }
    }
    // Auto colors based on detector type
    const opacity = overlayOpacity / 100 * 0.5
    if (detectorType === 'bf') {
      return { fill: `rgba(0, 162, 255, ${opacity})`, stroke: 'rgba(0, 162, 255, 0.8)' }
    } else if (detectorType === 'abf') {
      return { fill: `rgba(0, 200, 150, ${opacity})`, stroke: 'rgba(0, 200, 150, 0.8)' }
    } else {
      return { fill: `rgba(255, 162, 0, ${opacity})`, stroke: 'rgba(255, 162, 0, 0.8)' }
    }
  }, [overlayColor, overlayOpacity, detectorType])

  /**
   * Refetch images when display settings change.
   */
  useEffect(() => {
    if (!datasetInfo) return

    const refetchImages = async () => {
      try {
        const { inner, outer } = getDetectorRadii()
        const effectiveType = getEffectiveDetectorType()
        // Refetch virtual image with current detector settings
        const virtual = await fetchVirtualImage(effectiveType, inner, outer, logScale, contrastMin, contrastMax)
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
        const effectiveType = getEffectiveDetectorType()
        const virtual = await fetchVirtualImage(effectiveType, inner, outer, logScale, contrastMin, contrastMax)
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
  }, [detectorType, bfRadius, abfInner, abfOuter, adfInner, adfOuter, getDetectorRadii, getEffectiveDetectorType, fetchVirtualImage, logScale, contrastMin, contrastMax, datasetInfo])

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
      if (type === 'none' || type === 'bf') {
        setBfRadius(outer)
      } else if (type === 'abf') {
        setAbfInner(inner)
        setAbfOuter(outer)
      } else if (type === 'adf') {
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
   * Export the current virtual image as PNG.
   */
  const handleExportVirtualImage = useCallback(async () => {
    if (!virtualImage) return

    // Convert base64 to blob and save
    const result = await window.electronAPI.saveImage(
      virtualImage.image_base64,
      `virtual-${detectorType}.png`
    )
    if (!result.success && !result.canceled) {
      console.error('Failed to export virtual image:', result.error)
    }
  }, [virtualImage, detectorType])

  /**
   * Calculate detector statistics from the current diffraction pattern.
   */
  const getDetectorStats = useCallback(() => {
    if (!virtualImage) return null
    const { inner, outer } = getDetectorRadii()
    // Approximate detector area in pixels (area of annulus)
    const area = Math.PI * (outer * outer - inner * inner)
    return {
      area: Math.round(area),
      inner,
      outer,
    }
  }, [virtualImage, getDetectorRadii])

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

  /**
   * Run hot pixel filtering on the current dataset.
   */
  const handleFilterHotPixels = useCallback(async () => {
    setFilterHotPixelsLoading(true)

    try {
      const response = await fetch(`${BACKEND_URL}/dataset/preprocess/filter-hot-pixels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thresh: filterHotPixelsThresh }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      // Close dialog
      setShowFilterHotPixelsDialog(false)

      // Refresh visualizations
      const { inner, outer } = getDetectorRadii()
      const effectiveType = getEffectiveDetectorType()

      // Refetch all images
      const [diffraction, virtual] = await Promise.all([
        fetchMeanDiffraction(logScale, contrastMin, contrastMax),
        fetchVirtualImage(effectiveType, inner, outer, logScale, contrastMin, contrastMax),
      ])

      setMeanDiffraction(diffraction)
      setVirtualImage(virtual)

      // Clear cached patterns so they get refreshed
      setCachedMeanDiffraction(null)
      setCachedMaxDiffraction(null)

      // Clear clicked position diffraction (will be refetched on next click)
      if (clickedPosition) {
        const pattern = await fetchDiffractionPattern(
          clickedPosition.x,
          clickedPosition.y,
          logScale,
          contrastMin,
          contrastMax
        )
        setDiffractionPattern(pattern)
      }

      // Clear atom positions since the data changed
      setAtomPositions([])
      setAtomDetectionResult(null)

    } catch (err) {
      console.error('Failed to filter hot pixels:', err)
      setError(err instanceof Error ? err.message : 'Failed to filter hot pixels')
    } finally {
      setFilterHotPixelsLoading(false)
    }
  }, [filterHotPixelsThresh, getDetectorRadii, getEffectiveDetectorType, fetchMeanDiffraction, fetchVirtualImage, fetchDiffractionPattern, logScale, contrastMin, contrastMax, clickedPosition])

  useEffect(() => {
    window.electronAPI.onFileSelected(handleFileSelected)

    return () => {
      window.electronAPI.removeFileSelectedListener()
    }
  }, [handleFileSelected])

  // Listen for filter hot pixels dialog trigger from menu
  useEffect(() => {
    window.electronAPI.onShowFilterHotPixelsDialog(() => {
      if (datasetInfo) {
        setShowFilterHotPixelsDialog(true)
      }
    })

    return () => {
      window.electronAPI.removeFilterHotPixelsDialogListener()
    }
  }, [datasetInfo])

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
                  <div className="radio-group detector-radio-group">
                    <label className="radio-label" title="Bright Field - center disk">
                      <input
                        type="radio"
                        name="detector"
                        value="bf"
                        checked={detectorType === 'bf'}
                        onChange={() => setDetectorType('bf')}
                      />
                      BF
                    </label>
                    <label className="radio-label" title="Annular Bright Field - just outside BF">
                      <input
                        type="radio"
                        name="detector"
                        value="abf"
                        checked={detectorType === 'abf'}
                        onChange={() => setDetectorType('abf')}
                      />
                      ABF
                    </label>
                    <label className="radio-label" title="Annular Dark Field - outer ring">
                      <input
                        type="radio"
                        name="detector"
                        value="adf"
                        checked={detectorType === 'adf'}
                        onChange={() => setDetectorType('adf')}
                      />
                      ADF
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

                {detectorType === 'abf' && (
                  <div className="sidebar-control">
                    <div className="contrast-sliders">
                      <div className="slider-row">
                        <span className="slider-label-small">Inner</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={abfInner}
                          onChange={(e) => {
                            const newInner = Number(e.target.value)
                            setAbfInner(newInner)
                            if (newInner >= abfOuter) {
                              setAbfOuter(Math.min(100, newInner + 1))
                            }
                          }}
                          className="slider"
                        />
                        <span className="slider-value">{abfInner}</span>
                      </div>
                      <div className="slider-row">
                        <span className="slider-label-small">Outer</span>
                        <input
                          type="range"
                          min="1"
                          max="100"
                          value={abfOuter}
                          onChange={(e) => {
                            const newOuter = Number(e.target.value)
                            setAbfOuter(newOuter)
                            if (newOuter <= abfInner) {
                              setAbfInner(Math.max(0, newOuter - 1))
                            }
                          }}
                          className="slider"
                        />
                        <span className="slider-value">{abfOuter}</span>
                      </div>
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
              </div>

              {/* Calibration Section */}
              <div className="sidebar-section">
                <div
                  className="sidebar-section-header"
                  onClick={() => setCalibrationCollapsed(!calibrationCollapsed)}
                >
                  <h3 className="sidebar-section-title">Calibration</h3>
                  <span className="sidebar-collapse-icon">
                    {calibrationCollapsed ? '+' : '−'}
                  </span>
                </div>
                {!calibrationCollapsed && (
                  <div className="calibration-content">
                    <div className="calibration-group">
                      <label className="calibration-label">Q (Reciprocal)</label>
                      <div className="calibration-row">
                        <input
                          type="number"
                          className="calibration-input"
                          value={qPixelSize}
                          onChange={(e) => {
                            const val = Number(e.target.value)
                            if (val > 0) {
                              setQPixelSize(val)
                              updateCalibration({ q_pixel_size: val })
                            }
                          }}
                          step="0.001"
                          min="0.001"
                          disabled={!datasetInfo}
                        />
                        <select
                          className="calibration-select"
                          value={qPixelUnits}
                          onChange={(e) => {
                            setQPixelUnits(e.target.value)
                            updateCalibration({ q_pixel_units: e.target.value })
                          }}
                          disabled={!datasetInfo}
                        >
                          <option value="1/Å">1/Å</option>
                          <option value="1/nm">1/nm</option>
                          <option value="mrad">mrad</option>
                        </select>
                      </div>
                    </div>
                    <div className="calibration-group">
                      <label className="calibration-label">R (Real Space)</label>
                      <div className="calibration-row">
                        <input
                          type="number"
                          className="calibration-input"
                          value={rPixelSize}
                          onChange={(e) => {
                            const val = Number(e.target.value)
                            if (val > 0) {
                              setRPixelSize(val)
                              updateCalibration({ r_pixel_size: val })
                            }
                          }}
                          step="0.001"
                          min="0.001"
                          disabled={!datasetInfo}
                        />
                        <select
                          className="calibration-select"
                          value={rPixelUnits}
                          onChange={(e) => {
                            setRPixelUnits(e.target.value)
                            updateCalibration({ r_pixel_units: e.target.value })
                          }}
                          disabled={!datasetInfo}
                        >
                          <option value="Å">Å</option>
                          <option value="nm">nm</option>
                          <option value="pm">pm</option>
                        </select>
                      </div>
                    </div>
                  </div>
                )}
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
                {/* Selection tools - only visible in Mean/Max mode */}
                {diffractionViewMode !== 'live' && (
                  <div className="selection-tools">
                    <button
                      className={`selection-tool-btn ${selectionTool === 'rectangle' ? 'active' : ''}`}
                      onClick={() => setSelectionTool('rectangle')}
                      title="Rectangle selection"
                    >
                      ▢
                    </button>
                    <button
                      className={`selection-tool-btn ${selectionTool === 'ellipse' ? 'active' : ''}`}
                      onClick={() => setSelectionTool('ellipse')}
                      title="Circle selection"
                    >
                      ◯
                    </button>
                    <button
                      className={`selection-tool-btn ${selectionTool === 'lasso' ? 'active' : ''}`}
                      onClick={() => setSelectionTool('lasso')}
                      title="Lasso selection"
                    >
                      ✎
                    </button>
                    {selection && (
                      <button
                        className="selection-tool-btn clear-btn"
                        onClick={handleClearSelection}
                        title="Clear selection"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                )}
              </div>
              <div className="panel-content">
                {virtualImage && (
                  <div
                    className="panel-image-container"
                    style={{ cursor: diffractionViewMode !== 'live' && selectionTool ? 'crosshair' : undefined }}
                    onMouseDown={handleRealSpaceMouseDown}
                    onMouseMove={handleRealSpaceMouseMove}
                    onMouseUp={handleRealSpaceMouseUp}
                    onMouseLeave={() => {
                      if (isDrawing) {
                        setIsDrawing(false)
                        setDrawingPoints([])
                      }
                    }}
                  >
                    <img
                      ref={realSpaceImageRef}
                      className="panel-image clickable"
                      src={`data:image/png;base64,${virtualImage.image_base64}`}
                      alt="Virtual bright-field image"
                      onClick={handleRealSpaceClick}
                      draggable={false}
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
                      {/* Selection overlay - show only in Mean/Max mode */}
                      {diffractionViewMode !== 'live' && selection && (
                        <>
                          {selection.type === 'rectangle' && selection.points.length >= 2 && (
                            <rect
                              x={Math.min(selection.points[0][0], selection.points[1][0])}
                              y={Math.min(selection.points[0][1], selection.points[1][1])}
                              width={Math.abs(selection.points[1][0] - selection.points[0][0])}
                              height={Math.abs(selection.points[1][1] - selection.points[0][1])}
                              fill="rgba(0, 200, 255, 0.3)"
                              stroke="rgba(0, 200, 255, 0.9)"
                              strokeWidth="1.5"
                            />
                          )}
                          {selection.type === 'ellipse' && selection.points.length >= 2 && (
                            <circle
                              cx={selection.points[0][0]}
                              cy={selection.points[0][1]}
                              r={Math.abs(selection.points[1][0] - selection.points[0][0])}
                              fill="rgba(0, 200, 255, 0.3)"
                              stroke="rgba(0, 200, 255, 0.9)"
                              strokeWidth="1.5"
                            />
                          )}
                          {selection.type === 'polygon' && selection.points.length >= 3 && (
                            <polygon
                              points={selection.points.map(p => `${p[0]},${p[1]}`).join(' ')}
                              fill="rgba(0, 200, 255, 0.3)"
                              stroke="rgba(0, 200, 255, 0.9)"
                              strokeWidth="1.5"
                            />
                          )}
                        </>
                      )}
                      {/* Drawing preview */}
                      {isDrawing && drawingPoints.length > 0 && (
                        <>
                          {selectionTool === 'rectangle' && drawingPoints.length >= 2 && (
                            <rect
                              x={Math.min(drawingPoints[0][0], drawingPoints[1][0])}
                              y={Math.min(drawingPoints[0][1], drawingPoints[1][1])}
                              width={Math.abs(drawingPoints[1][0] - drawingPoints[0][0])}
                              height={Math.abs(drawingPoints[1][1] - drawingPoints[0][1])}
                              fill="rgba(255, 220, 0, 0.3)"
                              stroke="rgba(255, 220, 0, 0.9)"
                              strokeWidth="1.5"
                              strokeDasharray="4 2"
                            />
                          )}
                          {selectionTool === 'ellipse' && drawingPoints.length >= 2 && (
                            <circle
                              cx={drawingPoints[0][0]}
                              cy={drawingPoints[0][1]}
                              r={Math.abs(drawingPoints[1][0] - drawingPoints[0][0])}
                              fill="rgba(255, 220, 0, 0.3)"
                              stroke="rgba(255, 220, 0, 0.9)"
                              strokeWidth="1.5"
                              strokeDasharray="4 2"
                            />
                          )}
                          {selectionTool === 'lasso' && drawingPoints.length >= 2 && (
                            <polyline
                              points={drawingPoints.map(p => `${p[0]},${p[1]}`).join(' ')}
                              fill="none"
                              stroke="rgba(255, 220, 0, 0.9)"
                              strokeWidth="1.5"
                              strokeDasharray="4 2"
                            />
                          )}
                        </>
                      )}
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
                      {/* Crosshair at clicked position - only in live mode */}
                      {diffractionViewMode === 'live' && clickedPosition && (
                        <>
                          <line
                            x1={clickedPosition.x - 4}
                            y1={clickedPosition.y}
                            x2={clickedPosition.x + 4}
                            y2={clickedPosition.y}
                            stroke="#00ff00"
                            strokeWidth="1"
                          />
                          <line
                            x1={clickedPosition.x}
                            y1={clickedPosition.y - 4}
                            x2={clickedPosition.x}
                            y2={clickedPosition.y + 4}
                            stroke="#00ff00"
                            strokeWidth="1"
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
                <div className="diffraction-mode-container">
                  <select
                    className="diffraction-mode-select"
                    value={diffractionViewMode}
                    onChange={(e) => handleDiffractionViewModeChange(e.target.value as 'live' | 'mean' | 'max')}
                  >
                    <option value="live">Live</option>
                    <option value="mean">Mean</option>
                    <option value="max">Max</option>
                  </select>
                  {/* Region indicator */}
                  {diffractionViewMode !== 'live' && regionDiffraction && (
                    <span className="region-indicator" title={`${regionDiffraction.pixels_in_region} pixels selected`}>
                      (region)
                    </span>
                  )}
                </div>
              </div>
              <div className="panel-content">
                {(() => {
                  // Determine which pattern to display based on view mode
                  let pattern: MeanDiffractionResponse | DiffractionPatternResponse | RegionDiffractionResponse | null = null
                  if (diffractionViewMode === 'live') {
                    pattern = diffractionPattern || meanDiffraction
                  } else if (diffractionViewMode === 'mean') {
                    // Use region diffraction if available, otherwise use cached mean
                    pattern = regionDiffraction || cachedMeanDiffraction || meanDiffraction
                  } else if (diffractionViewMode === 'max') {
                    // Use region diffraction if available, otherwise use cached max
                    pattern = regionDiffraction || cachedMaxDiffraction
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
                        {showDetectorOverlay && detectorType !== 'none' && (() => {
                          const colors = getOverlayColor()
                          const centerX = pattern.width / 2 + centerOffsetX
                          const centerY = pattern.height / 2 + centerOffsetY

                          if (detectorType === 'bf') {
                            // BF: filled semi-transparent circle
                            return (
                              <circle
                                cx={centerX}
                                cy={centerY}
                                r={bfRadius}
                                fill={colors.fill}
                                stroke={colors.stroke}
                                strokeWidth="1"
                              />
                            )
                          } else if (detectorType === 'abf') {
                            // ABF: semi-transparent annulus
                            return (
                              <>
                                <defs>
                                  <mask id="abf-annulus-mask">
                                    <rect width="100%" height="100%" fill="white" />
                                    <circle cx={centerX} cy={centerY} r={abfInner} fill="black" />
                                  </mask>
                                </defs>
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={abfOuter}
                                  fill={colors.fill}
                                  mask="url(#abf-annulus-mask)"
                                />
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={abfOuter}
                                  fill="none"
                                  stroke={colors.stroke}
                                  strokeWidth="1"
                                />
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={abfInner}
                                  fill="none"
                                  stroke={colors.stroke}
                                  strokeWidth="1"
                                />
                              </>
                            )
                          } else {
                            // ADF: semi-transparent annulus
                            return (
                              <>
                                <defs>
                                  <mask id="adf-annulus-mask">
                                    <rect width="100%" height="100%" fill="white" />
                                    <circle cx={centerX} cy={centerY} r={adfInner} fill="black" />
                                  </mask>
                                </defs>
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={adfOuter}
                                  fill={colors.fill}
                                  mask="url(#adf-annulus-mask)"
                                />
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={adfOuter}
                                  fill="none"
                                  stroke={colors.stroke}
                                  strokeWidth="1"
                                />
                                <circle
                                  cx={centerX}
                                  cy={centerY}
                                  r={adfInner}
                                  fill="none"
                                  stroke={colors.stroke}
                                  strokeWidth="1"
                                />
                              </>
                            )
                          }
                        })()}
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
              {workflowCollapsed ? (
                <div className="workflow-summary">
                  {workflowTab === 'virtual-detector' ? (
                    <span className="workflow-summary-text">
                      Virtual Detector: {detectorType === 'none' ? 'None' : (
                        detectorType === 'bf' ? `BF (0-${bfRadius} px)` :
                        detectorType === 'abf' ? `ABF (${abfInner}-${abfOuter} px)` :
                        `ADF (${adfInner}-${adfOuter} px)`
                      )}
                    </span>
                  ) : (
                    <span className="workflow-summary-text">
                      Atom Detection: {atomDetectionResult ?? 'Not run'}
                    </span>
                  )}
                </div>
              ) : (
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
              )}
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
                  <div className="workflow-tab-content virtual-detector-content">
                    <div className="detector-controls-row">
                      {/* Overlay Settings */}
                      <div className="detector-control-group">
                        <label className="checkbox-label">
                          <input
                            type="checkbox"
                            checked={showDetectorOverlay}
                            onChange={(e) => setShowDetectorOverlay(e.target.checked)}
                            disabled={detectorType === 'none'}
                          />
                          Show overlay
                        </label>
                      </div>

                      <div className="detector-control-group">
                        <label className="detector-control-label">Opacity</label>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={overlayOpacity}
                          onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                          className="slider detector-slider"
                          disabled={!showDetectorOverlay || detectorType === 'none'}
                        />
                        <span className="detector-control-value">{overlayOpacity}%</span>
                      </div>

                      <div className="detector-control-group">
                        <label className="detector-control-label">Color</label>
                        <select
                          className="detector-select"
                          value={overlayColor}
                          onChange={(e) => setOverlayColor(e.target.value as typeof overlayColor)}
                          disabled={!showDetectorOverlay || detectorType === 'none'}
                        >
                          <option value="auto">Auto</option>
                          <option value="yellow">Yellow</option>
                          <option value="cyan">Cyan</option>
                          <option value="green">Green</option>
                          <option value="orange">Orange</option>
                          <option value="magenta">Magenta</option>
                        </select>
                      </div>

                      {/* Center Offset */}
                      <div className="detector-control-group">
                        <label className="detector-control-label">Center X</label>
                        <input
                          type="number"
                          className="detector-input"
                          value={centerOffsetX}
                          onChange={(e) => setCenterOffsetX(Number(e.target.value))}
                          disabled={detectorType === 'none'}
                        />
                      </div>
                      <div className="detector-control-group">
                        <label className="detector-control-label">Center Y</label>
                        <input
                          type="number"
                          className="detector-input"
                          value={centerOffsetY}
                          onChange={(e) => setCenterOffsetY(Number(e.target.value))}
                          disabled={detectorType === 'none'}
                        />
                      </div>
                      <button
                        className={`detector-pick-button ${pickingCenter ? 'active' : ''}`}
                        onClick={() => setPickingCenter(!pickingCenter)}
                        disabled={detectorType === 'none'}
                        title="Click on diffraction pattern to set center"
                      >
                        {pickingCenter ? 'Cancel' : 'Pick'}
                      </button>
                    </div>

                    <div className="detector-actions-row">
                      <div className="config-buttons">
                        <button className="config-button" onClick={handleSaveConfig}>
                          Save Config
                        </button>
                        <button className="config-button" onClick={handleLoadConfig}>
                          Load Config
                        </button>
                      </div>
                      <button
                        className="detector-export-button"
                        onClick={handleExportVirtualImage}
                        disabled={!virtualImage}
                      >
                        Export Image
                      </button>
                      {(() => {
                        const stats = getDetectorStats()
                        if (!stats || detectorType === 'none') return null
                        return (
                          <span className="detector-stats">
                            Area: {stats.area} px | Range: {stats.inner}-{stats.outer} px
                          </span>
                        )
                      })()}
                    </div>
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

      {/* Filter Hot Pixels Dialog */}
      {showFilterHotPixelsDialog && (
        <div className="modal-overlay">
          <div className="modal-dialog filter-hot-pixels-dialog">
            <h3 className="modal-title">Filter Hot Pixels</h3>
            <p className="modal-description">
              Remove anomalously bright pixels from the dataset.
              Pixels more than N standard deviations above the mean will be filtered.
            </p>
            <div className="modal-form">
              <label className="modal-label">
                Threshold (σ)
                <input
                  type="number"
                  className="modal-input"
                  value={filterHotPixelsThresh}
                  onChange={(e) => setFilterHotPixelsThresh(Number(e.target.value))}
                  min={1}
                  max={20}
                  step={0.5}
                  disabled={filterHotPixelsLoading}
                />
              </label>
            </div>
            <div className="modal-actions">
              <button
                className="modal-button modal-button-secondary"
                onClick={() => setShowFilterHotPixelsDialog(false)}
                disabled={filterHotPixelsLoading}
              >
                Cancel
              </button>
              <button
                className="modal-button modal-button-primary"
                onClick={handleFilterHotPixels}
                disabled={filterHotPixelsLoading}
              >
                {filterHotPixelsLoading ? (
                  <>
                    <span className="loading-spinner" />
                    Filtering...
                  </>
                ) : (
                  'Apply'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
