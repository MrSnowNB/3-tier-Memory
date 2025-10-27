# CyberMesh Replication Notes

---
title: "CyberMesh Replication Notes"
version: "1.0"
date: "2025-10-27"
purpose: "Environment-specific knowledge for reproducible setup"
scope: "Deployment and development environments"
---

## Golden Environment

### Hardware & Platform
- **Platform**: macOS 15.1, Apple Silicon M3 Max
- **RAM**: 32GB
- **GPU**: 40-core Apple M3 Max GPU
- **Storage**: 1TB SSD

### Software Stack
- **Python**: 3.12.0 via uv package manager
- **Key Dependencies**:
  - torch==2.5.0+metal (MPS-enabled)
  - fastapi==0.104.1
  - ollama>=0.2.1 with nomic-embed-text:137m-v1.5-fp16
- **Environment Variables**:
  - MPS_MEMORY_FRACTION=0.8
  - OMP_NUM_THREADS=8

## Replicable Setup Checklist

- [ ] Install uv package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Clone repo and cd to project root
- [ ] Create virtual environment: `uv venv .venv && source .venv/bin/activate`
- [ ] Install dependencies: `uv pip install -e .`
- [ ] Copy environment variables: `cp .env.example .env` and edit as needed
- [ ] Install Ollama: `brew install ollama` (macOS) or follow OS-specific instructions
- [ ] Pull embedding model: `ollama pull nomic-embed-text:137m-v1.5-fp16`
- [ ] Run smoke tests: `pytest tests/test_smoke.py` to verify setup

## Known Pitfalls to Avoid Next Run

### MPS-Specific Issues
- **M3 Max MPS warmup**: First embedding call takes 2-5s, add warmup batch to initialization
- **MPS Memory Fragmentation**: Long-running processes may cause OOM even with sufficient RAM; restart periodically
- **MPS Kernel Precompilation**: First run after dependency change recompiles kernels, expect ~30s delay

### Conway Pattern Edge Effects
- Use minimum 16x16 grid for stable glider propagation
- Edge wrapping disabled to avoid artificial boundary conditions
- Pattern stability degrades below 8x8 grid size

### WebSocket Frame Drops at High Frequency
- Limit broadcast payload to <10KB when tick rate >5Hz
- Implement client-side frame dropping for performance monitoring
- Use binary serialization (msgpack) instead of JSON for state updates

### Pytest Import Conflicts
- Run tests with `python -m pytest` to avoid path issues
- Use `--tb=short` for readable tracebacks in CI
- Disable parallel execution with `-n0` for debugging memory issues

### FastAPI Development Server
- Use `uvicorn main:app --reload` for development
- Set `--host 0.0.0.0` for network access during testing
- Configure `--workers 1` to avoid multiprocessing issues with MPS

### Memory Allocation Strategies
- Set MPS_MEMORY_FRACTION=0.8 to prevent system hangs
- Monitor with `torch.mps.current_allocated_memory()` in debug mode
- Use `torch.mps.empty_cache()` after large tensor operations

### AsyncIO Stress Test Pitfalls
- Stress test timeout context managers must be used within asyncio tasks
- Ollama client verification fails outside proper async context
- Always wrap timeout operations in `asyncio.create_task()` or equivalent

## Performance Baselines (Golden Environment)

### CA Grid Operations
- Single 32x32 grid update: <10ms
- 1000-step glider propagation: <5s
- Memory usage for 32x32 grid: ~50MB

### Embedding Operations
- Single text embedding (512 tokens): <50ms warmup, <10ms subsequent
- Batch of 32 embeddings: <500ms total
- Memory per embedding model: ~500MB

### Shard Routing Benchmarks
- Single-hop delivery: <100ms end-to-end
- Multi-hop 3x3 grid: <1s with energy hints
- Concurrent shards (K=16): <2s p95 latency

## Environment-Specific Configurations

### pytest.ini Overrides
```ini
# For lower-memory systems (<16GB RAM)
addopts =
    --maxfail=1
    --durations=5
    -n0  # Disable parallel execution
```

### torch MPS Settings
```python
# In main application code
import torch
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(0.8)
    torch.mps.empty_cache()  # At startup
```
# TODO LIST RECOMMENDED
When starting a new task, it is recommended to create a todo list.




1. Include the task_progress parameter in your next tool call

2. Create a comprehensive checklist of all steps needed

3. Use markdown format: - [ ] for incomplete, - [x] for complete



**Benefits of creating a todo list now:**

	- Clear roadmap for implementation

	- Progress tracking throughout the task

	- Nothing gets forgotten or missed

	- Users can see, monitor, and edit the plan



**Example structure:**
```

- [ ] Analyze requirements

- [ ] Set up necessary files

- [ ] Implement main functionality

- [ ] Handle edge cases

- [ ] Test the implementation

- [ ] Verify results
```



Keeping the todo list updated helps track progress and ensures nothing is missed.
