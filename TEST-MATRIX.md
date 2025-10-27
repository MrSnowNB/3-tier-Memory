# CyberMesh Test Matrix

---
title: "CyberMesh Test Matrix"
version: "1.0"
date: "2025-10-27"
purpose: "Track all test cases and their success criteria across phases"
scope: "Unit, integration, performance, hardware-dependent tests"
---

## Test Categories

- **unit**: Fast, isolated unit tests (no external dependencies)
- **integration**: Slower tests with dependencies (network, filesystem)
- **performance**: Benchmark tests against thresholds
- **hardware**: M3 Max GPU and memory-specific tests

## Matrix Format

Each test entry follows this template:

```yaml
---
test_id: "phase_X_test_name"
category: "unit|integration|performance|hardware"
description: "Brief description of what the test validates"
success_criteria: "Specific pass/fail conditions"
phase: X
dependencies: "External systems required"
tags: ["tag1", "tag2"]
expected_runtime_ms: 100
memory_threshold_mb: 512
---

## Test Implementation
[Code snippet or reference to test file]

## Failure Scenarios
[Common failure modes and troubleshooting]
```

---

## Phase 0: Infrastructure Tests

### unit_pyproject_imports
---
test_id: "phase0_imports"
category: "unit"
description: "Verify all configured packages import correctly"
success_criteria: "All imports in pyproject.toml succeed without ImportError"
phase: 0
dependencies: "Python virtual environment"
tags: ["setup", "imports"]
expected_runtime_ms: 500
memory_threshold_mb: 50
---

### unit_tool_configs
---
test_id: "phase0_tools"
category: "unit"
description: "Validate ruff, mypy, pytest configurations"
success_criteria: "Tools run without configuration errors"
phase: 0
dependencies: "Installed linters and type checkers"
tags: ["setup", "tools"]
expected_runtime_ms: 2000
memory_threshold_mb: 100
---

## Phase 1: Conway CA Grid Tests

### unit_conway_rules
---
test_id: "phase1_conway_rules"
category: "unit"
description: "Test Conway's Game of Life cell state transitions"
success_criteria: "Live/dead rules applied correctly for all 9 neighborhood patterns"
phase: 1
dependencies: "None"
tags: ["conway", "rules", "cellular"]
expected_runtime_ms: 100
memory_threshold_mb: 10
---

### unit_glider_propagation
---
test_id: "phase1_glider"
category: "unit"
description: "Verify glider pattern moves correctly across grid"
success_criteria: "Glider translates 4 steps over 1000 ticks without distortion"
phase: 1
dependencies: "None"
tags: ["conway", "patterns", "glider"]
expected_runtime_ms: 50
memory_threshold_mb: 50
---

### performance_ca_grid_update
---
test_id: "phase1_grid_perf"
category: "performance"
description: "Benchmark CA grid updates for 32x32 and larger grids"
success_criteria: "Update latency <10ms for 32x32 grid, scales linearly"
phase: 1
dependencies: "None"
tags: ["performance", "ca_grid", "benchmark"]
expected_runtime_ms: 150
memory_threshold_mb: 200
---

### hardware_mps_memory
---
test_id: "phase1_mps_mem"
category: "hardware"
description: "Test MPS memory allocation and cleanup"
success_criteria: "No memory leaks after 1000 CA updates"
phase: 1
dependencies: "Apple Silicon MPS"
tags: ["hardware", "mps", "memory"]
expected_runtime_ms: 5000
memory_threshold_mb: 100
---

## Phase 2: Shard Routing Tests

### unit_single_hop_routing
---
test_id: "phase2_single_hop"
category: "unit"
description: "Test shard delivery to adjacent CA cells"
success_criteria: "Single-hop delivery succeeds >=95% with perfect grid connectivity"
phase: 2
dependencies: "None"
tags: ["routing", "shards", "single_hop"]
expected_runtime_ms: 100
memory_threshold_mb: 50
---

### integration_embedding_generation
---
test_id: "phase2_embeddings"
category: "integration"
description: "Generate and cache embeddings for routing decisions"
success_criteria: "Ollama returns vectors within 50ms, cache hit <1ms"
phase: 2
dependencies: "Ollama service running"
tags: ["embeddings", "ollama", "integration"]
expected_runtime_ms: 50
memory_threshold_mb: 600
---

### performance_batch_embeddings
---
test_id: "phase2_embed_perf"
category: "performance"
description: "Benchmark embedding batch processing"
success_criteria: "32 embeddings computed in <500ms on M3 Max"
phase: 2
dependencies: "Ollama service"
tags: ["performance", "embeddings", "batch"]
expected_runtime_ms: 500
memory_threshold_mb: 800
---

## Phase 3: CA + Shard Integration Tests

### integration_shard_tracking
---
test_id: "phase3_shard_track"
category: "integration"
description: "Track shards moving with gliders in CA grid"
success_criteria: "Shards stay within 2.5 cells of glider center, tracking accuracy >85%"
phase: 3
dependencies: "Integrated CA + routing system"
tags: ["integration", "tracking", "accuracy"]
expected_runtime_ms: 2000
memory_threshold_mb: 300
---

### performance_end_to_end
---
test_id: "phase3_e2e_perf"
category: "performance"
description: "Full pipeline: CA update → embedding → routing → tracking"
success_criteria: "Average latency <100ms per shard update cycle"
phase: 3
dependencies: "Full system integration"
tags: ["performance", "end_to_end", "latency"]
expected_runtime_ms: 100
memory_threshold_mb: 1000
---

## Test Execution Status

| Test ID | Status | Last Run | Result | Notes |
|---------|--------|----------|--------|-------|
| phase0_imports | Pending | - | - | - |
| phase0_tools | Pending | - | - | - |
| phase1_conway_rules | Pending | - | - | - |
| phase1_glider | Pending | - | - | - |
| phase1_grid_perf | Pending | - | - | - |
| phase1_mps_mem | Pending | - | - | - |
| phase2_single_hop | Pending | - | - | - |
| phase2_embeddings | Pending | - | - | - |
| phase2_embed_perf | Pending | - | - | - |
| phase3_shard_track | Pending | - | - | - |
| phase3_e2e_perf | Pending | - | - | - |

## Test Data Management

- **Baseline Results**: Stored in `tests/baselines/`
- **Performance Logs**: Collected in `logs/performance/`
- **Failure Snapshots**: Automatically captured for debugging

## Continuous Integration

Tests are automatically run in CI with:
- `pytest --junit-xml=test-results.xml`
- Coverage reporting with `pytest-cov`
- Performance regression detection
