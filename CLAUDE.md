# CLAUDE.md - 4D-STEM Visualization Platform

## Project Overview

Enterprise-grade visualization and analysis software for 4D-STEM microscopy data, analogous to what cryoSPARC is for cryo-EM. This is a prototype/MVP phase focusing on core workflows and user experience validation.

### Target Users
- Academic researchers (primary, initial audience)
- Facility managers at electron microscopy centers
- Eventually: industry R&D labs

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Desktop vs Web | Desktop app | 4D-STEM datasets are too large (1GB+) to stream; compute must stay close to data |
| Frontend framework | Electron | Larger talent pool than Tauri, acceptable memory overhead for prototype, can migrate later if needed |
| Backend | Python (FastAPI subprocess) | Native ecosystem for 4D-STEM (py4DSTEM, ptyrad, hyperspy); scientific community standard |
| IPC | HTTP to localhost | FastAPI on 127.0.0.1, simple and debuggable |

### Future Considerations
- May evolve to cryoSPARC-style local web server for multi-user/cluster deployment
- Tauri migration possible if memory footprint becomes problematic
- Job queuing (Celery/Redis) for long-running reconstructions

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Electron App                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  Renderer Process (Frontend)                  │  │
│  │  - React + TypeScript                         │  │
│  │  - WebGL for visualization                    │  │
│  │  - Dataset browser, workflow builder          │  │
│  └──────────────────┬────────────────────────────┘  │
│                     │ HTTP (localhost:8000)          │
│  ┌──────────────────▼────────────────────────────┐  │
│  │  Python Backend (subprocess)                  │  │
│  │  - FastAPI server                             │  │
│  │  - py4DSTEM, ptyrad, hyperspy                 │  │
│  │  - Dask for out-of-core processing            │  │
│  │  - Serves slices/projections, not full data   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Data Flow Principles
- **Never load full dataset into memory** - use memory-mapped access (Dask, Zarr)
- **Server-side slicing** - frontend requests specific frames/projections
- **Lazy computation** - defer processing until results are needed
- **Thumbnail/preview pipeline** - fast low-res previews, full-res on demand

---

## Demo Dataset

Using a **1GB 4D-STEM dataset** for development and testing.

Location: `data/demo/` (gitignored)

When implementing features, always test against this dataset to ensure:
- Memory usage stays reasonable (<2GB total app footprint)
- Slice extraction is fast (<100ms for single frame)
- UI remains responsive during processing

---

## Development Standards

### Code Quality

**Python Backend:**
- Type hints on all function signatures
- Docstrings in NumPy format
- Pydantic models for all API request/response schemas
- Async endpoints where appropriate

**TypeScript Frontend:**
- Strict mode enabled
- Interfaces for all data structures
- React functional components with hooks
- No `any` types without justification

### Documentation Requirements

This is enterprise-grade software. Every feature needs:

1. **Code documentation**
   - Docstrings/JSDoc on all public functions
   - Inline comments for non-obvious logic
   - Type annotations everywhere

2. **API documentation**
   - OpenAPI/Swagger auto-generated from FastAPI
   - Example requests/responses
   - Error codes and meanings

3. **User documentation** (in `docs/`)
   - Feature guides with screenshots
   - Workflow tutorials
   - Troubleshooting guide

4. **Architecture decision records** (in `docs/adr/`)
   - Document significant technical decisions
   - Include context, options considered, rationale

### Git Practices

- **Conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
- **Feature branches**: `feature/virtual-detector`, `fix/memory-leak`
- **PR descriptions**: Include what, why, and how to test
- **No large files in git**: Data files go in `.gitignore`

### Testing

- **Backend**: pytest with fixtures for sample data
- **Frontend**: Vitest + React Testing Library
- **E2E**: Playwright for critical workflows
- **Performance benchmarks**: Track memory/time for key operations

---

## Project Structure

```
4dstem-viewer/
├── CLAUDE.md                 # This file
├── README.md                 # User-facing documentation
├── package.json              # Electron/frontend dependencies
├── electron/
│   ├── main.ts               # Electron main process
│   ├── preload.ts            # Preload scripts for IPC
│   └── python.ts             # Python subprocess management
├── src/                      # React frontend
│   ├── components/
│   ├── hooks/
│   ├── services/             # API client
│   └── types/
├── backend/                  # Python backend
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py           # FastAPI app
│   │   ├── routers/
│   │   ├── services/         # Business logic
│   │   └── schemas/          # Pydantic models
│   └── tests/
├── data/                     # Gitignored, local datasets
│   └── demo/
├── docs/
│   ├── adr/                  # Architecture decision records
│   ├── api/                  # API documentation
│   └── user/                 # User guides
└── scripts/                  # Build, dev, utility scripts
```

---

## Common Commands

```bash
# Install dependencies
npm install
cd backend && pip install -e ".[dev]"

# Development
npm run dev              # Start Electron + React dev server
cd backend && uvicorn app.main:app --reload  # Backend only

# Testing
npm test                 # Frontend tests
cd backend && pytest     # Backend tests

# Build
npm run build            # Production build
```

---

## Key Libraries

### Python Backend
- **FastAPI** - API framework
- **py4DSTEM** - 4D-STEM analysis
- **ptyrad** / **pyptyRAD** - Ptychography
- **hyperspy** - Multi-dimensional data
- **Dask** - Out-of-core computation
- **Zarr** - Chunked array storage
- **numpy**, **scipy** - Numerical computing

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **WebGL** / **Three.js** or **regl** - Visualization
- **TanStack Query** - API state management
- **Zustand** - UI state management

---

## Claude Code Guidelines

### When implementing features:
1. Check this file for architectural context
2. Follow the documentation standards above
3. Test against the 1GB demo dataset
4. Keep memory efficiency in mind

### When asked to "just make it work":
- Still add basic type hints and docstrings
- Still handle errors gracefully
- Note any technical debt with `// TODO:` comments

### When refactoring:
- Preserve existing API contracts
- Update tests to match
- Update documentation

### Useful context commands:
```
# Show project structure
find . -type f -name "*.py" -o -name "*.ts" -o -name "*.tsx" | head -50

# Check memory usage patterns
grep -r "load\|read\|open" backend/app/

# Find TODOs
grep -r "TODO\|FIXME\|HACK" --include="*.py" --include="*.ts"
```

---

## Current Status

**Phase: Prototype / MVP**

### Immediate priorities:
1. [ ] Scaffold Electron + React + FastAPI project
2. [ ] Load and display 4D-STEM dataset (single frame)
3. [ ] Virtual detector placement UI
4. [ ] Basic ptychography workflow

### Not yet in scope:
- Multi-user support
- Cluster job scheduling
- Plugin system
- Advanced reconstruction algorithms

---

## Contacts & Resources

- **py4DSTEM docs**: https://py4dstem.readthedocs.io/
- **Electron docs**: https://www.electronjs.org/docs
- **FastAPI docs**: https://fastapi.tiangolo.com/
- **4D-STEM background**: [add relevant papers/resources]
