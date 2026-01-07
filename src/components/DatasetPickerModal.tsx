import { useState, useMemo } from 'react'
import type { HDF5DatasetInfo } from '../types/dataset'

interface DatasetPickerModalProps {
  /** Whether the modal is visible */
  isOpen: boolean
  /** List of datasets from the HDF5 file */
  datasets: HDF5DatasetInfo[]
  /** Filename to display in the title */
  filename: string
  /** Callback when user selects a dataset and clicks Load */
  onSelect: (datasetPath: string) => void
  /** Callback when user cancels */
  onCancel: () => void
}

/** Represents a node in the HDF5 tree structure */
interface TreeNode {
  name: string
  fullPath: string
  children: Map<string, TreeNode>
  dataset: HDF5DatasetInfo | null
}

/**
 * Build a tree structure from flat dataset paths.
 */
function buildTree(datasets: HDF5DatasetInfo[]): TreeNode {
  const root: TreeNode = {
    name: '',
    fullPath: '',
    children: new Map(),
    dataset: null,
  }

  for (const ds of datasets) {
    const parts = ds.path.split('/').filter(Boolean)
    let current = root

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      const isLast = i === parts.length - 1

      if (!current.children.has(part)) {
        current.children.set(part, {
          name: part,
          fullPath: '/' + parts.slice(0, i + 1).join('/'),
          children: new Map(),
          dataset: null,
        })
      }

      current = current.children.get(part)!

      if (isLast) {
        current.dataset = ds
      }
    }
  }

  return root
}

interface TreeNodeProps {
  node: TreeNode
  depth: number
  selectedPath: string | null
  onSelect: (path: string) => void
}

/**
 * Renders a single node in the tree with its children.
 */
function TreeNodeComponent({ node, depth, selectedPath, onSelect }: TreeNodeProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const hasChildren = node.children.size > 0
  const isDataset = node.dataset !== null
  const isSelected = selectedPath === node.fullPath

  const handleClick = () => {
    if (isDataset) {
      onSelect(node.fullPath)
    } else if (hasChildren) {
      setIsExpanded(!isExpanded)
    }
  }

  const formatShape = (shape: number[]): string => {
    return shape.join(' x ')
  }

  return (
    <div className="tree-node">
      <div
        className={`tree-node-row ${isDataset ? 'dataset' : 'group'} ${isSelected ? 'selected' : ''}`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={handleClick}
      >
        {hasChildren && !isDataset && (
          <span className="tree-expand-icon">{isExpanded ? '▼' : '▶'}</span>
        )}
        {!hasChildren && !isDataset && <span className="tree-expand-icon" />}
        {isDataset && <span className="tree-expand-icon">□</span>}

        <span className="tree-node-name">{node.name}</span>

        {isDataset && node.dataset && (
          <span className="tree-node-meta">
            <span className="tree-node-shape">[{formatShape(node.dataset.shape)}]</span>
            <span className="tree-node-dtype">{node.dataset.dtype}</span>
            {node.dataset.is_4d && <span className="tree-node-badge">4D-STEM</span>}
          </span>
        )}
      </div>

      {hasChildren && isExpanded && (
        <div className="tree-children">
          {Array.from(node.children.values()).map((child) => (
            <TreeNodeComponent
              key={child.fullPath}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Modal dialog for selecting a dataset from an HDF5 file tree.
 */
export function DatasetPickerModal({
  isOpen,
  datasets,
  filename,
  onSelect,
  onCancel,
}: DatasetPickerModalProps) {
  const [selectedPath, setSelectedPath] = useState<string | null>(null)

  const tree = useMemo(() => buildTree(datasets), [datasets])

  if (!isOpen) {
    return null
  }

  const handleLoad = () => {
    if (selectedPath) {
      onSelect(selectedPath)
    }
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onCancel()
    }
  }

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-container">
        <div className="modal-header">
          <h2 className="modal-title">Select 4D-STEM Dataset</h2>
          <span className="modal-filename">{filename}</span>
        </div>

        <div className="modal-body">
          <div className="tree-container">
            {tree.children.size > 0 ? (
              Array.from(tree.children.values()).map((child) => (
                <TreeNodeComponent
                  key={child.fullPath}
                  node={child}
                  depth={0}
                  selectedPath={selectedPath}
                  onSelect={setSelectedPath}
                />
              ))
            ) : (
              <div className="tree-empty">No datasets found in file</div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button className="modal-button cancel" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="modal-button primary"
            onClick={handleLoad}
            disabled={!selectedPath}
          >
            Load Selected
          </button>
        </div>
      </div>
    </div>
  )
}
