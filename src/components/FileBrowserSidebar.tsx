/**
 * VS Code-style file browser sidebar for navigating HDF5 files with multiple datacubes.
 */

import { useState, useCallback, useEffect } from 'react'
import type { HDF5TreeNode, HDF5TreeProbeResponse } from '../types/dataset'

interface FileBrowserSidebarProps {
  /** Whether the sidebar is collapsed */
  collapsed: boolean
  /** Toggle collapse state */
  onToggleCollapse: () => void
  /** The probed file tree (null if no file is open) */
  fileTree: HDF5TreeProbeResponse | null
  /** Currently loaded dataset path */
  selectedDatasetPath: string | null
  /** Whether a dataset is currently being loaded */
  isLoading: boolean
  /** Path of dataset currently being loaded */
  loadingDatasetPath: string | null
  /** Callback when a 4D dataset is clicked */
  onDatasetSelect: (datasetPath: string) => void
  /** Callback when refresh button is clicked */
  onRefresh: () => void
}

/**
 * Format a shape array for display (e.g., [256, 256, 128, 128] -> "256x256x128x128")
 */
function formatShape(shape: number[]): string {
  return shape.join('\u00D7')
}

/**
 * Recursive tree node component for rendering HDF5 structure
 */
function TreeNode({
  node,
  depth,
  selectedPath,
  loadingPath,
  expandedPaths,
  onToggleExpand,
  onSelect,
}: {
  node: HDF5TreeNode
  depth: number
  selectedPath: string | null
  loadingPath: string | null
  expandedPaths: Set<string>
  onToggleExpand: (path: string) => void
  onSelect: (path: string) => void
}) {
  const isGroup = node.type === 'group'
  const isDataset = node.type === 'dataset'
  const is4D = node.is_4d === true
  const isExpanded = expandedPaths.has(node.path)
  const isSelected = selectedPath === node.path
  const isLoadingThis = loadingPath === node.path
  const hasChildren = isGroup && node.children && node.children.length > 0

  const handleClick = useCallback(() => {
    if (isGroup && hasChildren) {
      onToggleExpand(node.path)
    } else if (isDataset && is4D) {
      onSelect(node.path)
    }
  }, [isGroup, isDataset, is4D, hasChildren, node.path, onToggleExpand, onSelect])

  const handleChevronClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (isGroup && hasChildren) {
      onToggleExpand(node.path)
    }
  }, [isGroup, hasChildren, node.path, onToggleExpand])

  // Determine icon based on node type
  let icon: string
  if (isGroup) {
    icon = isExpanded ? '\u{1F4C2}' : '\u{1F4C1}' // Open/closed folder
  } else if (is4D) {
    icon = '\u{1F4CB}' // 4D dataset - clipboard icon as cube placeholder
  } else {
    icon = '\u{1F5BC}' // 2D dataset - image icon
  }

  return (
    <div className="tree-node-wrapper">
      <div
        className={`tree-node ${isDataset && !is4D ? 'disabled' : ''} ${isSelected ? 'selected' : ''} ${isLoadingThis ? 'loading' : ''}`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={handleClick}
        title={isDataset ? `${node.path}\n${node.shape ? formatShape(node.shape) : ''}\n${node.dtype || ''}` : node.path}
      >
        {/* Chevron for groups */}
        <span
          className={`tree-chevron ${!hasChildren ? 'hidden' : ''}`}
          onClick={handleChevronClick}
        >
          {isExpanded ? '\u25BC' : '\u25B6'}
        </span>

        {/* Icon */}
        <span className="tree-icon">{icon}</span>

        {/* Name */}
        <span className={`tree-name ${is4D ? 'is-4d' : ''}`}>
          {node.name}
        </span>

        {/* 4D badge */}
        {is4D && <span className="tree-4d-badge">4D</span>}

        {/* Shape (for datasets) */}
        {isDataset && node.shape && (
          <span className="tree-shape">({formatShape(node.shape)})</span>
        )}

        {/* Loading spinner */}
        {isLoadingThis && <span className="tree-loading-spinner" />}
      </div>

      {/* Children */}
      {isGroup && isExpanded && node.children && (
        <div className="tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              loadingPath={loadingPath}
              expandedPaths={expandedPaths}
              onToggleExpand={onToggleExpand}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Check if a tree node or any of its descendants contains a 4D dataset
 */
function containsFourD(node: HDF5TreeNode): boolean {
  if (node.type === 'dataset' && node.is_4d) {
    return true
  }
  if (node.type === 'group' && node.children) {
    return node.children.some(containsFourD)
  }
  return false
}

/**
 * Get all group paths that contain 4D datasets (for auto-expansion)
 */
function getPathsWithFourD(node: HDF5TreeNode, paths: Set<string> = new Set()): Set<string> {
  if (node.type === 'group' && node.children) {
    if (node.children.some(containsFourD)) {
      paths.add(node.path)
    }
    for (const child of node.children) {
      getPathsWithFourD(child, paths)
    }
  }
  return paths
}

export function FileBrowserSidebar({
  collapsed,
  onToggleCollapse,
  fileTree,
  selectedDatasetPath,
  isLoading,
  loadingDatasetPath,
  onDatasetSelect,
  onRefresh,
}: FileBrowserSidebarProps) {
  // Track which nodes are expanded
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set(['/']))

  // Auto-expand paths containing 4D datasets when fileTree changes
  useEffect(() => {
    if (fileTree?.root) {
      const paths = getPathsWithFourD(fileTree.root)
      paths.add('/') // Always expand root
      setExpandedPaths(paths)
    } else {
      setExpandedPaths(new Set(['/']))
    }
  }, [fileTree])

  const handleToggleExpand = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }, [])

  const handleRefresh = useCallback(() => {
    onRefresh()
    // Reset expanded paths - they'll be recalculated when new tree arrives
  }, [onRefresh])

  if (collapsed) {
    return (
      <div className="file-browser-sidebar collapsed">
        <button
          className="file-browser-expand-btn"
          onClick={onToggleCollapse}
          title="Expand file browser"
        >
          <span className="file-browser-expand-icon">{'\u{1F4C1}'}</span>
        </button>
      </div>
    )
  }

  return (
    <div className="file-browser-sidebar">
      {/* Header */}
      <div className="file-browser-header">
        <span className="file-browser-title">EXPLORER</span>
        <div className="file-browser-header-actions">
          <button
            className="file-browser-refresh-btn"
            onClick={handleRefresh}
            title="Refresh file structure"
            disabled={!fileTree || isLoading}
          >
            {'\u21BB'}
          </button>
          <button
            className="file-browser-collapse-btn"
            onClick={onToggleCollapse}
            title="Collapse file browser"
          >
            {'\u2039'}
          </button>
        </div>
      </div>

      {/* Tree content */}
      <div className="file-browser-content">
        {!fileTree ? (
          <div className="file-browser-empty">
            <span className="file-browser-empty-text">No file open</span>
            <span className="file-browser-empty-hint">Use File &rarr; Open to load a dataset</span>
          </div>
        ) : (
          <div className="file-browser-tree">
            {/* File name as root */}
            <div
              className="tree-node tree-root"
              onClick={() => handleToggleExpand('/')}
            >
              <span className="tree-chevron">
                {expandedPaths.has('/') ? '\u25BC' : '\u25B6'}
              </span>
              <span className="tree-icon">{'\u{1F4C4}'}</span>
              <span className="tree-name tree-filename">{fileTree.filename}</span>
            </div>

            {/* Tree children */}
            {expandedPaths.has('/') && fileTree.root.children && (
              <div className="tree-children">
                {fileTree.root.children.map((child) => (
                  <TreeNode
                    key={child.path}
                    node={child}
                    depth={1}
                    selectedPath={selectedDatasetPath}
                    loadingPath={loadingDatasetPath}
                    expandedPaths={expandedPaths}
                    onToggleExpand={handleToggleExpand}
                    onSelect={onDatasetSelect}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
