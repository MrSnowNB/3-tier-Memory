---
title: "[FAILURE] Stress Test Timeout Context Manager Bug"
date: "2025-10-27 13:10:00"
severity: "P1"
assigned_to: "human"
status: "open"
---

## Failure Summary
**Phase**: Validate (Phase 2 stress test)
**Gate Failed**: perf (stress testing)
**Task**: Run 10-minute embedding stress test to verify "1-hour stress test" requirement

## Reproduction Steps
1. Start Ollama service: `brew services start ollama`
2. Pull model: `ollama pull nomic-embed-text:137m-v1.5-fp16`
3. Run stress test: `python scripts/stress_embeddings.py --duration 600 --concurrency 4`

## Error Details
```
OSError: Timeout context manager should be used inside a task
2025-10-27 13:10:00,492 - ERROR - Stress test failed: Ollama service not available
```

## Environment Context
- **Git Commit**: Latest commit on main branch
- **Branch**: main
- **Python**: 3.12.0 (uv venv)
- **Platform**: macOS 15.1, Apple Silicon M3 Max, 32GB RAM
- **Modified Files**: scripts/stress_embeddings.py (timeout usage issue)

## Logs Collected
- Ollama service is running (confirmed via `ps aux`)
- Virtual environment activated correctly
- Previous routing tests: PASS (multi-hop: 100%, performance: OK)
- Embedding generation works in non-async context (used successfully in routing)

## Living Docs Updated
- [x] TROUBLESHOOTING.md: Added "AsyncIO Timeout Error in Stress Test" entry
- [x] REPLICATION-NOTES.md: Added "AsyncIO Stress Test Pitfalls" section

## Next Steps Required
- [ ] Investigate asyncio timeout context manager usage in scripts/stress_embeddings.py
- [ ] Fix timeout context to work within asyncio task boundaries
- [ ] Re-run 10-minute stress test to verify Phase 2 memory leak requirements
- [ ] Document permanent fix in codebase to prevent regression

## Blocking Dependencies
**Phase 2 completion cannot be validated** without functional stress testing.
Phase 2 requires proving "No memory leaks in 1-hour stress test" via functional 10-minute test.

**Critical Path Blocked:**
- Phase 2 validation gate cannot pass without stress test
- Policy protocol: "On any failure: update living docs → stop for human input"
- Human investigation required for asyncio timeout issue resolution
