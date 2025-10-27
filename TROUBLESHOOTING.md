# CyberMesh Troubleshooting Guide

---
title: "CyberMesh Troubleshooting Guide"
version: "1.0"
date: "2025-10-27"
purpose: "Institutional knowledge for rapid problem resolution"
scope: "All development phases"
---

## Guidelines

**When to add an entry:**
- Repeat failure patterns emerge
- Non-obvious fixes discovered
- Environment-specific issues encountered
- Performance anomalies identified

**Entry format:**
- Context: When/where this occurs
- Symptom: Observable behavior
- Error Snippet: Actual error message/stack trace
- Probable Cause: Root cause analysis
- Quick Fix: Immediate workaround
- Permanent Fix: Long-term solution
- Prevention: How to avoid in future
- Tags: gate_type, phase, severity (P0/P1/P2/P3)

---

## Known Issues

### Dependency Resolution Loop
**Context**: Fresh environment setup or requirements.txt update
**Symptom**: pip install hangs or produces conflicting requirements
**Error Snippet**:
```
ERROR: Could not find a version that satisfies the requirement...
```
**Probable Cause**: Unpinned indirect dependencies causing conflicts
**Quick Fix**: Use `pip install --force-reinstall` with specific versions
**Permanent Fix**: Generate constraints.txt with `pip-tools compile`
**Prevention**: Pin all indirect deps; run `pip check` in CI
**Tags**: setup, dependency, P1

### Embedding Model OOM
**Context**: Batch processing >32 embeddings on M3 Max 32GB
**Symptom**: Process killed, MPS out of memory error
**Error Snippet**:
```
torch.cuda.OutOfMemoryError: MPS backend out of memory
```
**Probable Cause**: Batch size exceeds MPS memory limits
**Quick Fix**: Reduce batch_size to 16; truncate sequences to 512 tokens
**Permanent Fix**: Implement adaptive batching with memory monitoring
**Prevention**: Add memory usage tests; set MPS_MEMORY_FRACTION=0.8
**Tags**: embedding, memory, P0

### Socket Port Conflicts in Local Orchestrator
**Context**: Multiple test runs or dev server restarts
**Symptom**: FastAPI server fails to bind, "Address already in use"
**Error Snippet**:
```
OSError: [Errno 48] Address already in use: ('127.0.0.1', 8000)
```
**Probable Cause**: Previous server process not cleanly terminated
**Quick Fix**: `lsof -ti:8000 | xargs kill -9`
**Permanent Fix**: Use randomized port binding with retries
**Prevention**: Proper signal handling for graceful shutdown
**Tags**: networking, testing, P2

---

## Quick Reference Commands

**Check for port conflicts**:
```bash
lsof -i :8000  # Check port 8000
lsof -ti:8000 | xargs kill -9  # Kill processes on port 8000
```

**Memory usage monitoring**:
```bash
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

**Environment validation**:
```bash
python --version && which python && which pip && which uv
```

---

## Escalation Tags

- **P0**: Blocks all progress, system down, critical data loss risk
- **P1**: Major functionality broken, no viable workaround
- **P2**: Minor functionality broken, workarounds available
- **P3**: Cosmetic issues, inefficiencies, minor annoyances
