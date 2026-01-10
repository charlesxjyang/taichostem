/**
 * TypeScript types for dataset API endpoints.
 */

/** Information about a single dataset within an HDF5 file. */
export interface HDF5DatasetInfo {
  path: string
  shape: number[]
  dtype: string
  is_4d: boolean
}

/** Response for single-datacube files (.dm4, .mrc). */
export interface SingleProbeResponse {
  type: 'single'
  shape: number[]
  dtype: string
}

/** Response for HDF5 files with a tree structure. */
export interface HDF5TreeProbeResponse {
  type: 'hdf5_tree'
  datasets: HDF5DatasetInfo[]
}

/** Union type for probe endpoint responses. */
export type ProbeResponse = SingleProbeResponse | HDF5TreeProbeResponse

/** Response from the dataset load endpoint. */
export interface DatasetLoadResponse {
  shape: number[]
  dtype: string
  dataset_path: string | null
}

/** Extended dataset info including file path and internal path. */
export interface DatasetInfo extends DatasetLoadResponse {
  filePath: string
}

/** Response from the mean diffraction endpoint. */
export interface MeanDiffractionResponse {
  image_base64: string
  width: number
  height: number
}

/** Response from the virtual image endpoint. */
export interface VirtualImageResponse {
  image_base64: string
  width: number
  height: number
}

/** Response from the diffraction pattern endpoint. */
export interface DiffractionPatternResponse {
  image_base64: string
  width: number
  height: number
  x: number
  y: number
}

/** Selection tool types for region selection. */
export type SelectionTool = 'rectangle' | 'ellipse' | 'lasso' | null

/** Region types for the API. */
export type RegionType = 'rectangle' | 'ellipse' | 'polygon'

/** Selection geometry stored in state. */
export interface SelectionGeometry {
  type: RegionType
  points: [number, number][]
}

/** Request for region-based diffraction pattern. */
export interface RegionDiffractionRequest {
  mode: 'mean' | 'max'
  region_type: RegionType
  points: [number, number][]
  log_scale?: boolean
  contrast_min?: number
  contrast_max?: number
}

/** Response from the region diffraction endpoint. */
export interface RegionDiffractionResponse {
  image_base64: string
  width: number
  height: number
  mode: string
  region_type: string
  pixels_in_region: number
}
