# CLAUDE.md - 4D-STEM Visualization Platform

## Project Overview

Enterprise-grade visualization and analysis software for 4D-STEM microscopy data, analogous to what cryoSPARC is for cryo-EM. This is a prototype/MVP phase focusing on core workflows and user experience validation.

We are using py4dstem for our backend. Reference their code at: backend/venv/lib/python3.11/site-packages/py4DSTEM
Use py4dstem over scipy or numpy wherever possible

---

## Demo Dataset

Using a **1GB 4D-STEM dataset** for development and testing.

Location: `raw_data/sim_Au_data_all_binned.h5/` (gitignored)

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

### Testing

- **Backend**: pytest with fixtures for sample data
- **Frontend**: Vitest + React Testing Library
- **E2E**: Playwright for critical workflows
- **Performance benchmarks**: Track memory/time for key operations


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

## Claude Code Guidelines


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

## Contacts & Resources

- **py4DSTEM docs**: https://py4dstem.readthedocs.io/
- **Electron docs**: https://www.electronjs.org/docs
- **FastAPI docs**: https://fastapi.tiangolo.com/
- **4D-STEM background**: [add relevant papers/resources]
