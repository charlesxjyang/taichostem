import { useState, useEffect, useCallback } from 'react'

const BACKEND_URL = 'http://127.0.0.1:8000'

interface DatasetInfo {
  filePath: string | null
  datasetPath: string | null
  shape: number[] | null
}

interface ProbeTemplateInfo {
  shape: number[]
  max_intensity: number
  center_x: number
  center_y: number
  preview: string
  position: number[] | null  // null for region selection
  source_type: 'point' | 'region'
  pixels_averaged: number
}

interface KernelInfo {
  kernel_preview: string
  kernel_lineprofile: string | null
  kernel_shape: number[]
  kernel_type: string
  radial_boundary: number
  sigmoid_width: number
}

interface DetectedDisk {
  qx: number
  qy: number
  correlation: number
}

interface PositionTestResult {
  position: number[]
  disks: DetectedDisk[]
  pattern_overlay: string
  disk_count: number
}

interface TestResults {
  results: PositionTestResult[]
  correlation_histogram: string | null
}

type WorkflowStep = 'template' | 'parameters' | 'results'
type KernelType = 'sigmoid' | 'gaussian' | 'raw'

export function WorkflowWindow() {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo>({
    filePath: null,
    datasetPath: null,
    shape: null,
  })
  const [isConnected, setIsConnected] = useState(false)
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('template')

  // Step 1: Template Setup state
  const [probeTemplate, setProbeTemplate] = useState<ProbeTemplateInfo | null>(null)
  const [isSettingProbe, setIsSettingProbe] = useState(false)
  const [probeConfirmed, setProbeConfirmed] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Step 2: Kernel Generation state
  const [kernelType, setKernelType] = useState<KernelType>('sigmoid')
  const [radialBoundary, setRadialBoundary] = useState(0.5)
  const [sigmoidWidth, setSigmoidWidth] = useState(0.1)
  const [kernelInfo, setKernelInfo] = useState<KernelInfo | null>(null)
  const [isGeneratingKernel, setIsGeneratingKernel] = useState(false)
  const [kernelConfirmed, setKernelConfirmed] = useState(false)

  // Step 3: Test Detection state
  const [testPositions, setTestPositions] = useState<number[][]>([])
  const [correlationThreshold, setCorrelationThreshold] = useState(0.3)
  const [minSpacing, setMinSpacing] = useState(5)
  const [subpixelRefinement, setSubpixelRefinement] = useState(true)
  const [edgeBoundary, setEdgeBoundary] = useState(2)
  const [testResults, setTestResults] = useState<TestResults | null>(null)
  const [isRunningTest, setIsRunningTest] = useState(false)
  const [selectedResultIndex, setSelectedResultIndex] = useState(0)
  const [realSpaceImage, setRealSpaceImage] = useState<string | null>(null)
  const [isAddingFromMain, setIsAddingFromMain] = useState(false)

  // Current selection from main window
  const [currentSelection, setCurrentSelection] = useState<{
    type: 'rectangle' | 'ellipse' | 'polygon'
    points: [number, number][]
  } | null>(null)

  // Get initial dataset info and set up listeners
  useEffect(() => {
    window.electronAPI.getDatasetInfo().then((info) => {
      setDatasetInfo(info)
      setIsConnected(true)
    })

    window.electronAPI.onDatasetInfoUpdated((info) => {
      setDatasetInfo(info)
      // Reset workflow when dataset changes
      setProbeTemplate(null)
      setProbeConfirmed(false)
    })

    return () => {
      window.electronAPI.removeDatasetInfoUpdatedListener()
    }
  }, [])

  // Poll for current selection from main window (for template step)
  useEffect(() => {
    if (currentStep !== 'template' || probeConfirmed) return

    const pollSelection = async () => {
      const selection = await window.electronAPI.getSelection()
      setCurrentSelection(selection)
    }

    // Poll immediately and then every 500ms
    pollSelection()
    const interval = setInterval(pollSelection, 500)

    return () => clearInterval(interval)
  }, [currentStep, probeConfirmed])

  /**
   * Calculate selection attributes for display.
   */
  const getSelectionAttributes = useCallback((): { label: string; value: string } | null => {
    if (!currentSelection) return null

    if (currentSelection.type === 'rectangle' && currentSelection.points.length >= 2) {
      const width = Math.abs(currentSelection.points[1][0] - currentSelection.points[0][0])
      const height = Math.abs(currentSelection.points[1][1] - currentSelection.points[0][1])
      return { label: 'Rectangle', value: `${width.toFixed(0)} x ${height.toFixed(0)} px` }
    } else if (currentSelection.type === 'ellipse' && currentSelection.points.length >= 2) {
      const radius = Math.abs(currentSelection.points[1][0] - currentSelection.points[0][0])
      return { label: 'Circle', value: `r = ${radius.toFixed(0)} px` }
    } else if (currentSelection.type === 'polygon' && currentSelection.points.length >= 3) {
      // Calculate polygon area using shoelace formula
      let area = 0
      const n = currentSelection.points.length
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n
        area += currentSelection.points[i][0] * currentSelection.points[j][1]
        area -= currentSelection.points[j][0] * currentSelection.points[i][1]
      }
      area = Math.abs(area) / 2
      return { label: 'Lasso', value: `${area.toFixed(0)} px²` }
    }
    return null
  }, [currentSelection])

  /**
   * Handle "I've selected the vacuum region" button click.
   * Checks for region selection first, then falls back to clicked position.
   * Region selection averages multiple patterns for better noise reduction.
   */
  const handleGrabPosition = useCallback(async () => {
    setError(null)
    setIsSettingProbe(true)

    try {
      // First check for region selection (rectangle, ellipse, or polygon)
      const selection = await window.electronAPI.getSelection()

      // Also get clicked position as fallback
      const position = await window.electronAPI.getClickedPosition()

      if (!selection && !position) {
        throw new Error('No selection in the main window. Please draw a region (rectangle, ellipse, or lasso) or click on a vacuum area first.')
      }

      // Build request body - prefer region selection if available
      let requestBody: object
      if (selection && selection.points.length >= 2) {
        // Use region selection
        requestBody = {
          region_type: selection.type,
          points: selection.points,
        }
      } else if (position) {
        // Use point selection
        requestBody = { x: position.x, y: position.y }
      } else {
        throw new Error('No valid selection found. Please draw a region or click on a vacuum area.')
      }

      // Extract probe template from the selection
      const response = await fetch(`${BACKEND_URL}/analysis/disk-detection/set-probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const data: ProbeTemplateInfo = await response.json()
      setProbeTemplate(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to extract probe template')
      setProbeTemplate(null)
    } finally {
      setIsSettingProbe(false)
    }
  }, [])

  /**
   * Confirm the current probe template and proceed.
   */
  const handleConfirmTemplate = useCallback(() => {
    if (probeTemplate) {
      setProbeConfirmed(true)
    }
  }, [probeTemplate])

  /**
   * Reset selection to try a different position.
   */
  const handleSelectDifferent = useCallback(async () => {
    setProbeTemplate(null)
    setProbeConfirmed(false)
    setError(null)

    // Clear the probe on backend
    try {
      await fetch(`${BACKEND_URL}/analysis/disk-detection/probe`, {
        method: 'DELETE',
      })
    } catch {
      // Ignore errors when clearing
    }
  }, [])

  /**
   * Generate cross-correlation kernel from probe template.
   */
  const generateKernel = useCallback(async () => {
    setError(null)
    setIsGeneratingKernel(true)

    try {
      const response = await fetch(`${BACKEND_URL}/analysis/disk-detection/generate-kernel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          kernel_type: kernelType,
          radial_boundary: radialBoundary,
          sigmoid_width: sigmoidWidth,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const data: KernelInfo = await response.json()
      setKernelInfo(data)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate kernel'
      // Add helpful context for common errors
      if (errorMessage.includes('Not Found') || errorMessage.includes('404')) {
        setError('Backend endpoint not found. Please restart the backend server.')
      } else if (errorMessage.includes('fetch') || errorMessage.includes('network')) {
        setError('Cannot connect to backend. Please ensure the backend is running.')
      } else {
        setError(errorMessage)
      }
      setKernelInfo(null)
    } finally {
      setIsGeneratingKernel(false)
    }
  }, [kernelType, radialBoundary, sigmoidWidth])

  // Auto-generate kernel when entering Step 2 or when parameters change
  useEffect(() => {
    if (currentStep === 'parameters' && probeTemplate) {
      generateKernel()
    }
  }, [currentStep, kernelType, radialBoundary, sigmoidWidth, generateKernel, probeTemplate])

  /**
   * Confirm the current kernel and proceed.
   */
  const handleConfirmKernel = useCallback(() => {
    if (kernelInfo) {
      setKernelConfirmed(true)
    }
  }, [kernelInfo])

  // Fetch real space image when entering Step 3
  const fetchRealSpaceImage = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/dataset/virtual-image?detector_type=bf&inner_radius=0&outer_radius=50`)
      if (response.ok) {
        const data = await response.json()
        setRealSpaceImage(data.image_base64)
      }
    } catch {
      // Ignore errors - image is optional
    }
  }, [])

  // Load real space image when entering Step 3
  useEffect(() => {
    if (currentStep === 'results') {
      fetchRealSpaceImage()
    }
  }, [currentStep, fetchRealSpaceImage])

  /**
   * Add a test position from clicking on the real space image.
   */
  const handleRealSpaceClick = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (!datasetInfo.shape) return

    const img = e.currentTarget
    const rect = img.getBoundingClientRect()
    const x = Math.floor((e.clientX - rect.left) / rect.width * datasetInfo.shape[1])
    const y = Math.floor((e.clientY - rect.top) / rect.height * datasetInfo.shape[0])

    // Don't add duplicates
    const exists = testPositions.some(p => p[0] === x && p[1] === y)
    if (!exists) {
      setTestPositions(prev => [...prev, [x, y]])
    }
  }, [datasetInfo.shape, testPositions])

  /**
   * Add position from main window.
   */
  const handleAddFromMainWindow = useCallback(async () => {
    setIsAddingFromMain(true)
    try {
      const position = await window.electronAPI.getClickedPosition()
      if (position) {
        const exists = testPositions.some(p => p[0] === position.x && p[1] === position.y)
        if (!exists) {
          setTestPositions(prev => [...prev, [position.x, position.y]])
        }
      }
    } finally {
      setIsAddingFromMain(false)
    }
  }, [testPositions])

  /**
   * Remove a test position.
   */
  const handleRemovePosition = useCallback((index: number) => {
    setTestPositions(prev => prev.filter((_, i) => i !== index))
    // Adjust selected result index if needed
    if (selectedResultIndex >= index && selectedResultIndex > 0) {
      setSelectedResultIndex(prev => prev - 1)
    }
  }, [selectedResultIndex])

  /**
   * Run disk detection test on selected positions.
   */
  const runDetectionTest = useCallback(async () => {
    if (testPositions.length === 0) {
      setError('Please select at least one test position.')
      return
    }

    setError(null)
    setIsRunningTest(true)

    try {
      const response = await fetch(`${BACKEND_URL}/analysis/disk-detection/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          positions: testPositions,
          correlation_threshold: correlationThreshold,
          min_spacing: minSpacing,
          subpixel: subpixelRefinement,
          edge_boundary: edgeBoundary,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const data: TestResults = await response.json()
      setTestResults(data)
      setSelectedResultIndex(0)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to run detection test'
      if (errorMessage.includes('Not Found') || errorMessage.includes('404')) {
        setError('Backend endpoint not found. Please restart the backend server.')
      } else {
        setError(errorMessage)
      }
      setTestResults(null)
    } finally {
      setIsRunningTest(false)
    }
  }, [testPositions, correlationThreshold, minSpacing, subpixelRefinement, edgeBoundary])

  /**
   * Handle Back button - close the window or go to previous step.
   */
  const handleBack = useCallback(() => {
    if (currentStep === 'template') {
      window.close()
    } else if (currentStep === 'parameters') {
      // Reset kernel state when going back
      setKernelInfo(null)
      setKernelConfirmed(false)
      setCurrentStep('template')
    } else if (currentStep === 'results') {
      // Reset test state when going back
      setTestResults(null)
      setCurrentStep('parameters')
    }
  }, [currentStep])

  /**
   * Handle Next button - proceed to next step.
   */
  const handleNext = useCallback(() => {
    if (currentStep === 'template' && probeConfirmed) {
      setCurrentStep('parameters')
    } else if (currentStep === 'parameters' && kernelConfirmed) {
      setCurrentStep('results')
    }
  }, [currentStep, probeConfirmed, kernelConfirmed])

  // Get display name for dataset
  const getDatasetName = (): string => {
    if (!datasetInfo.filePath) {
      return 'No dataset loaded'
    }
    const fileName = datasetInfo.filePath.split('/').pop() || datasetInfo.filePath
    if (datasetInfo.datasetPath) {
      return `${fileName} / ${datasetInfo.datasetPath}`
    }
    return fileName
  }

  // Get real space dimensions for display
  const getRealSpaceDims = (): string => {
    if (!datasetInfo.shape || datasetInfo.shape.length < 2) return ''
    return `${datasetInfo.shape[0]} x ${datasetInfo.shape[1]}`
  }

  return (
    <div className="workflow-window">
      <div className="workflow-window-header">
        <h1 className="workflow-window-title">Disk Detection Workflow</h1>
        <div className={`workflow-connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {isConnected ? 'Connected' : 'Connecting...'}
        </div>
      </div>

      {/* Step indicator */}
      <div className="workflow-step-indicator">
        <div className={`workflow-step ${currentStep === 'template' ? 'active' : probeConfirmed ? 'completed' : ''}`}>
          <span className="step-circle">1</span>
          <span className="step-label">Template Setup</span>
        </div>
        <div className="step-connector" />
        <div className={`workflow-step ${currentStep === 'parameters' ? 'active' : kernelConfirmed ? 'completed' : ''}`}>
          <span className="step-circle">2</span>
          <span className="step-label">Kernel</span>
        </div>
        <div className="step-connector" />
        <div className={`workflow-step ${currentStep === 'results' ? 'active' : ''}`}>
          <span className="step-circle">3</span>
          <span className="step-label">Detection</span>
        </div>
      </div>

      {/* Dataset info bar */}
      <div className="workflow-dataset-bar">
        <span className="dataset-label">Dataset:</span>
        <span className="dataset-name">{getDatasetName()}</span>
        {datasetInfo.shape && (
          <span className="dataset-dims">Real space: {getRealSpaceDims()}</span>
        )}
      </div>

      {/* Main content area */}
      <div className="workflow-main-content">
        {currentStep === 'template' && (
          <div className="workflow-step-content template-step">
            <div className="template-selection-panel">
              {/* Instructions */}
              <div className="selection-instructions">
                <div className="instruction-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
                    <path d="M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  </svg>
                </div>
                <h2 className="instruction-title">Select Vacuum Region</h2>
                <p className="instruction-text">
                  In the <strong>main window</strong>, select a <strong>vacuum region</strong> (no sample) in the real space image.
                </p>
                <p className="instruction-hint">
                  <strong>Recommended:</strong> Use the rectangle, ellipse, or lasso tool to draw a region. This averages multiple positions for a cleaner probe.
                </p>
                <p className="instruction-hint">
                  Alternatively, click a single point. The probe should show a clean diffraction pattern of the direct beam without sample scattering.
                </p>
              </div>

              {/* Status/Preview area */}
              <div className="selection-status">
                {isSettingProbe ? (
                  <div className="status-loading">
                    <span className="loading-spinner" />
                    <span>Extracting probe template...</span>
                  </div>
                ) : probeTemplate ? (
                  <div className="probe-result">
                    <div className="probe-preview-section">
                      <div className="probe-image-wrapper">
                        <img
                          className="probe-preview-image"
                          src={`data:image/png;base64,${probeTemplate.preview}`}
                          alt="Probe template"
                        />
                      </div>
                      <div className="probe-details">
                        <h3 className="probe-details-title">Probe Template</h3>
                        <div className="probe-detail-row">
                          <span className="detail-label">Source:</span>
                          <span className="detail-value">
                            {probeTemplate.source_type === 'region'
                              ? `Region (${probeTemplate.pixels_averaged} positions averaged)`
                              : `Point (${probeTemplate.position?.[0]}, ${probeTemplate.position?.[1]})`
                            }
                          </span>
                        </div>
                        <div className="probe-detail-row">
                          <span className="detail-label">Shape:</span>
                          <span className="detail-value">{probeTemplate.shape.join(' x ')}</span>
                        </div>
                        <div className="probe-detail-row">
                          <span className="detail-label">Max Intensity:</span>
                          <span className="detail-value">{probeTemplate.max_intensity.toExponential(2)}</span>
                        </div>
                        <div className="probe-detail-row">
                          <span className="detail-label">Center:</span>
                          <span className="detail-value">({probeTemplate.center_x.toFixed(1)}, {probeTemplate.center_y.toFixed(1)})</span>
                        </div>
                      </div>
                    </div>

                    <div className="probe-actions">
                      {probeConfirmed ? (
                        <div className="probe-confirmed-state">
                          <span className="confirmed-badge">Template Confirmed</span>
                          <button className="secondary-button" onClick={handleSelectDifferent}>
                            Select Different Position
                          </button>
                        </div>
                      ) : (
                        <>
                          <button className="primary-button" onClick={handleConfirmTemplate}>
                            Use This Template
                          </button>
                          <button className="secondary-button" onClick={handleSelectDifferent}>
                            Try Different Position
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="grab-position-section">
                    {/* Current selection status */}
                    {currentSelection ? (
                      <div className="current-selection-info">
                        <span className="selection-label">{getSelectionAttributes()?.label}:</span>
                        <span className="selection-value">{getSelectionAttributes()?.value}</span>
                      </div>
                    ) : (
                      <div className="no-selection-info">
                        No region selected in main window
                      </div>
                    )}
                    <p className="grab-instruction">
                      Draw a region or click a point in the main window, then click the button below.
                    </p>
                    <button
                      className="grab-position-button"
                      onClick={handleGrabPosition}
                      disabled={!datasetInfo.filePath}
                    >
                      Use Selection from Main Window
                    </button>
                  </div>
                )}
              </div>

              {/* Error display */}
              {error && (
                <div className="workflow-error">
                  <span className="error-icon">!</span>
                  <span className="error-message">{error}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {currentStep === 'parameters' && (
          <div className="workflow-step-content parameters-step">
            <div className="kernel-generation-panel">
              {/* Two-column layout: Probe on left, Kernel on right */}
              <div className="kernel-columns">
                {/* Left column: Probe template */}
                <div className="kernel-column probe-column">
                  <div className="column-header">
                    <h3 className="column-title">Probe Template</h3>
                  </div>
                  <div className="column-content">
                    {probeTemplate && (
                      <div className="kernel-image-container">
                        <img
                          className="kernel-preview-image"
                          src={`data:image/png;base64,${probeTemplate.preview}`}
                          alt="Probe template"
                        />
                      </div>
                    )}
                    <div className="kernel-info-box">
                      <p className="info-text">
                        This is the diffraction pattern from the vacuum region you selected.
                        It will be used as the template for cross-correlation.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Right column: Kernel preview and parameters */}
                <div className="kernel-column kernel-preview-column">
                  <div className="column-header">
                    <h3 className="column-title">Cross-Correlation Kernel</h3>
                  </div>
                  <div className="column-content">
                    {/* Kernel previews - image and line profile */}
                    <div className="kernel-previews-row">
                      {/* Kernel diffraction image */}
                      <div className="kernel-image-container kernel-diffraction">
                        {isGeneratingKernel ? (
                          <div className="status-loading">
                            <span className="loading-spinner" />
                            <span>Generating...</span>
                          </div>
                        ) : kernelInfo ? (
                          <img
                            className="kernel-preview-image"
                            src={`data:image/png;base64,${kernelInfo.kernel_preview}`}
                            alt="Kernel diffraction"
                          />
                        ) : (
                          <div className="panel-empty">
                            <p>No kernel</p>
                          </div>
                        )}
                      </div>

                      {/* Kernel line profile from py4DSTEM */}
                      <div className="kernel-image-container kernel-lineprofile">
                        {isGeneratingKernel ? (
                          <div className="status-loading">
                            <span className="loading-spinner" />
                          </div>
                        ) : kernelInfo?.kernel_lineprofile ? (
                          <img
                            className="kernel-preview-image"
                            src={`data:image/png;base64,${kernelInfo.kernel_lineprofile}`}
                            alt="Kernel line profile"
                          />
                        ) : kernelInfo ? (
                          <div className="panel-empty">
                            <p>Line profile unavailable</p>
                          </div>
                        ) : null}
                      </div>
                    </div>

                    {/* Parameter controls */}
                    <div className="kernel-parameters">
                      <div className="parameter-row">
                        <label className="parameter-label">Kernel Type:</label>
                        <select
                          className="parameter-select"
                          value={kernelType}
                          onChange={(e) => setKernelType(e.target.value as KernelType)}
                          disabled={kernelConfirmed}
                        >
                          <option value="sigmoid">Sigmoid (Edge-Enhanced)</option>
                          <option value="gaussian">Gaussian (Subtracted)</option>
                          <option value="raw">Raw (Unprocessed)</option>
                        </select>
                      </div>

                      {kernelType === 'sigmoid' && (
                        <>
                          <div className="parameter-row">
                            <label className="parameter-label">Radial Boundary:</label>
                            <div className="parameter-slider-group">
                              <input
                                type="range"
                                className="parameter-slider"
                                min="0.1"
                                max="0.9"
                                step="0.05"
                                value={radialBoundary}
                                onChange={(e) => setRadialBoundary(parseFloat(e.target.value))}
                                disabled={kernelConfirmed}
                              />
                              <span className="parameter-value">{radialBoundary.toFixed(2)}</span>
                            </div>
                          </div>
                          <div className="parameter-row">
                            <label className="parameter-label">Sigmoid Width:</label>
                            <div className="parameter-slider-group">
                              <input
                                type="range"
                                className="parameter-slider"
                                min="0.02"
                                max="0.3"
                                step="0.02"
                                value={sigmoidWidth}
                                onChange={(e) => setSigmoidWidth(parseFloat(e.target.value))}
                                disabled={kernelConfirmed}
                              />
                              <span className="parameter-value">{sigmoidWidth.toFixed(2)}</span>
                            </div>
                          </div>
                        </>
                      )}

                      <div className="kernel-info-box">
                        {kernelType === 'sigmoid' && (
                          <p className="info-text">
                            <strong>Sigmoid kernel</strong> emphasizes the edges of the Bragg disk.
                            Radial boundary controls where the transition occurs (fraction of estimated probe radius).
                            Sigmoid width controls the sharpness of the transition.
                          </p>
                        )}
                        {kernelType === 'gaussian' && (
                          <p className="info-text">
                            <strong>Gaussian kernel</strong> subtracts a smoothed version of the probe,
                            similar to py4DSTEM's <code>get_probe_kernel_subtrgaussian</code>.
                            This enhances edges while maintaining the overall disk shape.
                          </p>
                        )}
                        {kernelType === 'raw' && (
                          <p className="info-text">
                            <strong>Raw kernel</strong> uses the probe directly as the correlation template
                            with zero-mean normalization. Best when the probe has clear, sharp edges.
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Kernel actions */}
                    <div className="kernel-actions">
                      {kernelConfirmed ? (
                        <div className="kernel-confirmed-state">
                          <span className="confirmed-badge">Kernel Confirmed</span>
                          <button
                            className="secondary-button"
                            onClick={() => setKernelConfirmed(false)}
                          >
                            Adjust Parameters
                          </button>
                        </div>
                      ) : error ? (
                        <button
                          className="primary-button"
                          onClick={generateKernel}
                        >
                          Retry Kernel Generation
                        </button>
                      ) : (
                        <button
                          className="primary-button"
                          onClick={handleConfirmKernel}
                          disabled={!kernelInfo || isGeneratingKernel}
                        >
                          Use This Kernel
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Error display */}
              {error && (
                <div className="workflow-error">
                  <span className="error-icon">!</span>
                  <span className="error-message">{error}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {currentStep === 'results' && (
          <div className="workflow-step-content results-step">
            <div className="test-detection-panel">
              {/* Top row: Real space + Diffraction pattern */}
              <div className="test-top-row">
                {/* Left: Real space image for position selection */}
                <div className="test-realspace-panel">
                  <div className="panel-header-small">
                    <span className="panel-title-small">Select Test Positions</span>
                  </div>
                  <div className="test-realspace-content">
                    {realSpaceImage ? (
                      <div className="test-realspace-wrapper">
                        <img
                          className="test-realspace-image"
                          src={`data:image/png;base64,${realSpaceImage}`}
                          alt="Real space"
                          onClick={handleRealSpaceClick}
                        />
                        {/* Position markers overlay */}
                        {datasetInfo.shape && testPositions.length > 0 && (
                          <svg className="test-positions-overlay" viewBox={`0 0 ${datasetInfo.shape[1]} ${datasetInfo.shape[0]}`}>
                            {testPositions.map((pos, idx) => (
                              <g key={idx}>
                                <circle
                                  cx={pos[0]}
                                  cy={pos[1]}
                                  r={Math.max(2, Math.min(datasetInfo.shape![0], datasetInfo.shape![1]) / 40)}
                                  fill="none"
                                  stroke="#4ade80"
                                  strokeWidth={Math.max(1, Math.min(datasetInfo.shape![0], datasetInfo.shape![1]) / 100)}
                                />
                                <circle
                                  cx={pos[0]}
                                  cy={pos[1]}
                                  r={Math.max(1, Math.min(datasetInfo.shape![0], datasetInfo.shape![1]) / 80)}
                                  fill="#4ade80"
                                />
                              </g>
                            ))}
                          </svg>
                        )}
                      </div>
                    ) : (
                      <div className="panel-empty">
                        <span>Loading...</span>
                      </div>
                    )}
                  </div>
                  <div className="test-positions-list">
                    <div className="positions-header">
                      <span className="positions-count">{testPositions.length} positions</span>
                      <button
                        className="add-from-main-button"
                        onClick={handleAddFromMainWindow}
                        disabled={isAddingFromMain}
                      >
                        {isAddingFromMain ? 'Adding...' : '+ From Main'}
                      </button>
                    </div>
                    <div className="positions-chips">
                      {testPositions.map((pos, idx) => (
                        <span key={idx} className="position-chip">
                          ({pos[0]}, {pos[1]})
                          <button
                            className="remove-position"
                            onClick={() => handleRemovePosition(idx)}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                      {testPositions.length === 0 && (
                        <span className="no-positions">Click image to add positions</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right: Diffraction pattern with overlay */}
                <div className="test-diffraction-panel">
                  <div className="panel-header-small">
                    <span className="panel-title-small">
                      Detection Result
                      {testResults && testResults.results.length > 0 && (
                        <span className="result-position">
                          {' '}at ({testResults.results[selectedResultIndex]?.position[0]}, {testResults.results[selectedResultIndex]?.position[1]})
                        </span>
                      )}
                    </span>
                    {testResults && testResults.results.length > 1 && (
                      <div className="result-nav">
                        <button
                          className="result-nav-button"
                          onClick={() => setSelectedResultIndex(i => Math.max(0, i - 1))}
                          disabled={selectedResultIndex === 0}
                        >
                          ‹
                        </button>
                        <span className="result-nav-count">
                          {selectedResultIndex + 1}/{testResults.results.length}
                        </span>
                        <button
                          className="result-nav-button"
                          onClick={() => setSelectedResultIndex(i => Math.min(testResults.results.length - 1, i + 1))}
                          disabled={selectedResultIndex === testResults.results.length - 1}
                        >
                          ›
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="test-diffraction-content">
                    {isRunningTest ? (
                      <div className="status-loading">
                        <span className="loading-spinner" />
                        <span>Running detection...</span>
                      </div>
                    ) : testResults && testResults.results[selectedResultIndex] ? (
                      <img
                        className="test-diffraction-image"
                        src={`data:image/png;base64,${testResults.results[selectedResultIndex].pattern_overlay}`}
                        alt="Detection result"
                      />
                    ) : (
                      <div className="panel-empty">
                        <p>Run test to see results</p>
                      </div>
                    )}
                  </div>
                  {testResults && testResults.results[selectedResultIndex] && (
                    <div className="detection-stats">
                      <span className="disk-count">
                        Detected: <strong>{testResults.results[selectedResultIndex].disk_count}</strong> disks
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom: Parameters and histogram */}
              <div className="test-bottom-row">
                {/* Parameters */}
                <div className="test-parameters-panel">
                  <div className="panel-header-small">
                    <span className="panel-title-small">Detection Parameters</span>
                  </div>
                  <div className="test-parameters-content">
                    <div className="test-param-row">
                      <label className="test-param-label">Correlation Threshold:</label>
                      <div className="test-param-control">
                        <input
                          type="range"
                          className="test-param-slider"
                          min="0"
                          max="1"
                          step="0.05"
                          value={correlationThreshold}
                          onChange={(e) => setCorrelationThreshold(parseFloat(e.target.value))}
                        />
                        <span className="test-param-value">{correlationThreshold.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="test-param-row">
                      <label className="test-param-label">Min Peak Spacing:</label>
                      <div className="test-param-control">
                        <input
                          type="range"
                          className="test-param-slider"
                          min="1"
                          max="20"
                          step="1"
                          value={minSpacing}
                          onChange={(e) => setMinSpacing(parseInt(e.target.value))}
                        />
                        <span className="test-param-value">{minSpacing} px</span>
                      </div>
                    </div>
                    <div className="test-param-row">
                      <label className="test-param-label">Edge Boundary:</label>
                      <div className="test-param-control">
                        <input
                          type="range"
                          className="test-param-slider"
                          min="0"
                          max="20"
                          step="1"
                          value={edgeBoundary}
                          onChange={(e) => setEdgeBoundary(parseInt(e.target.value))}
                        />
                        <span className="test-param-value">{edgeBoundary} px</span>
                      </div>
                    </div>
                    <div className="test-param-row checkbox-row">
                      <label className="test-checkbox-label">
                        <input
                          type="checkbox"
                          checked={subpixelRefinement}
                          onChange={(e) => setSubpixelRefinement(e.target.checked)}
                        />
                        Subpixel Refinement
                      </label>
                    </div>
                    <button
                      className="run-test-button"
                      onClick={runDetectionTest}
                      disabled={isRunningTest || testPositions.length === 0}
                    >
                      {isRunningTest ? 'Running...' : 'Run Test'}
                    </button>
                  </div>
                </div>

                {/* Histogram */}
                <div className="test-histogram-panel">
                  <div className="panel-header-small">
                    <span className="panel-title-small">Correlation Distribution</span>
                  </div>
                  <div className="test-histogram-content">
                    {testResults?.correlation_histogram ? (
                      <img
                        className="test-histogram-image"
                        src={`data:image/png;base64,${testResults.correlation_histogram}`}
                        alt="Correlation histogram"
                      />
                    ) : (
                      <div className="panel-empty">
                        <p>Run test to see distribution</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Error display */}
              {error && (
                <div className="workflow-error">
                  <span className="error-icon">!</span>
                  <span className="error-message">{error}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Navigation buttons */}
      <div className="workflow-navigation">
        <button className="nav-button back-button" onClick={handleBack}>
          {currentStep === 'template' ? 'Close' : 'Back'}
        </button>
        <div className="nav-spacer" />
        <button
          className="nav-button next-button"
          onClick={handleNext}
          disabled={
            (currentStep === 'template' && !probeConfirmed) ||
            (currentStep === 'parameters' && !kernelConfirmed) ||
            (currentStep === 'results')
          }
        >
          {currentStep === 'template' ? 'Next' : currentStep === 'parameters' ? 'Test Detection' : 'Run Full Detection'}
        </button>
      </div>
    </div>
  )
}
