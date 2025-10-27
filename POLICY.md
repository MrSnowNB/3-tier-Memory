# CyberMesh Development Policy

---
title: "CyberMesh Development Policy"
version: "1.0"
date: "2025-10-27"
purpose: "Atomic, testable, gated development lifecycle for 3-tier memory architecture POC"
scope: "LoRA wave propagation via vector shard passing in Conway CA substrate"
---

## Core Principles

**File Format**: All files are markdown with YAML frontmatter or pure YAML.

**Task Structure**: Each task is atomic, testable, and gated.

**Failure Protocol**: On any failure/uncertainty: update living docs → stop for human input.

**Development Flow**: Sequential only - no parallel phases or shortcuts.

---

## Lifecycle Stages

```mermaid
graph LR
    A[Plan] --> B[Build]
    B --> C[Validate]
    C --> D[Review]
    D --> E[Release]
    E --> F[Next Iteration]
```

### 1. Plan
- **Input**: Requirements, architectural decisions, success criteria
- **Output**: Task specification with YAML frontmatter
- **Artifacts**: 
  - Task definition with clear acceptance criteria
  - Updated TEST-MATRIX.md with test cases
  - Architecture decision record (if applicable)
- **Gate**: Human approval of plan before proceeding

### 2. Build
- **Input**: Approved task specification
- **Output**: Implementation artifacts
- **Artifacts**:
  - Source code with docstrings
  - Unit tests scaffolded
  - Configuration files
  - Documentation updates
- **Gate**: Code compiles/imports without errors

### 3. Validate
- **Input**: Built artifacts
- **Output**: Test results and quality metrics
- **Gates** (ALL must pass):
  - `unit`: `pytest -q` green (all tests pass)
  - `lint`: `ruff check .` clean (no style violations)
  - `type`: `mypy src/` clean (no type errors)  
  - `docs`: spec drift check passes (docs match implementation)
  - `perf` (if applicable): Performance benchmarks within thresholds
  - `security` (if applicable): No security vulnerabilities detected

### 4. Review
- **Input**: Validated artifacts
- **Output**: Approved artifacts ready for release
- **Activities**:
  - Code review against requirements
  - Integration testing
  - Documentation review
  - Update REPLICATION-NOTES.md with any new findings
- **Gate**: Human sign-off on quality and completeness

### 5. Release
- **Input**: Reviewed artifacts
- **Output**: Production-ready components
- **Activities**:
  - Tag release version
  - Update changelog
  - Deploy to target environment (local dev/staging)
  - Verify deployment
- **Gate**: Deployment successful, basic smoke tests pass

---

## Validation Gates Detail

### Unit Testing Gate
```bash
# Must pass completely
pytest -q --tb=short
echo $? # Must be 0
```

### Linting Gate
```bash
# Must have zero violations
ruff check . --output-format=concise
flake8 src/ tests/ --max-line-length=100
echo $? # Must be 0
```

### Type Checking Gate
```bash
# Must have zero type errors
mypy src/ --strict
# Alternative: pyright for better performance
pyright src/
echo $? # Must be 0
```

### Documentation Gate
```bash
# Spec drift detection
python scripts/check_spec_drift.py
# Verify all docstrings present
python scripts/check_docstrings.py
# Build docs without errors
mkdocs build --strict
echo $? # Must be 0
```

### Performance Gate (Phase 3+)
```bash
# Benchmark against baselines
pytest tests/test_performance.py --benchmark-only
# Check thresholds in benchmark results
python scripts/check_performance_regression.py
```

### Security Gate (Phase 4+)
```bash
# Dependency vulnerabilities
pip-audit
# Secret scanning
detect-secrets scan --all-files
# YAML schema validation
yamale -s schema.yaml config/
```

---

## Living Documents

### TROUBLESHOOTING.md
**Purpose**: Capture institutional knowledge for rapid problem resolution

**Format**:
```markdown
### [Issue Title]
**Context**: [When/where this occurs]
**Symptom**: [Observable behavior]  
**Error Snippet**: 
\`\`\`
[Actual error message/stack trace]
\`\`\`
**Probable Cause**: [Root cause analysis]
**Quick Fix**: [Immediate workaround]
**Permanent Fix**: [Long-term solution]
**Prevention**: [How to avoid in future]
**Tags**: [gate_type, phase, severity]
```

**Seeded Entries**:
```markdown
### Dependency Resolution Loop
**Context**: Fresh environment setup or requirements.txt update
**Symptom**: pip install hangs or produces conflicting requirements
**Error Snippet**: 
\`\`\`
ERROR: Could not find a version that satisfies the requirement...
\`\`\`
**Probable Cause**: Unpinned indirect dependencies causing conflicts
**Quick Fix**: Use `pip install --force-reinstall` with specific versions
**Permanent Fix**: Generate constraints.txt with `pip-tools compile`
**Prevention**: Pin all indirect deps; run `pip check` in CI
**Tags**: setup, dependency, P1

### Embedding Model OOM
**Context**: Batch processing >32 embeddings on M3 Max 32GB
**Symptom**: Process killed, MPS out of memory error
**Error Snippet**:
\`\`\`
torch.cuda.OutOfMemoryError: MPS backend out of memory
\`\`\`
**Probable Cause**: Batch size exceeds MPS memory limits
**Quick Fix**: Reduce batch_size to 16; truncate sequences to 512 tokens
**Permanent Fix**: Implement adaptive batching with memory monitoring
**Prevention**: Add memory usage tests; set MPS_MEMORY_FRACTION=0.8
**Tags**: embedding, memory, P0

### Socket Port Conflicts in Local Orchestrator
**Context**: Multiple test runs or dev server restarts
**Symptom**: FastAPI server fails to bind, "Address already in use"
**Error Snippet**:
\`\`\`
OSError: [Errno 48] Address already in use: ('127.0.0.1', 8000)
\`\`\`
**Probable Cause**: Previous server process not cleanly terminated
**Quick Fix**: `lsof -ti:8000 | xargs kill -9`
**Permanent Fix**: Use randomized port binding with retries
**Prevention**: Proper signal handling for graceful shutdown
**Tags**: networking, testing, P2
```

### REPLICATION-NOTES.md
**Purpose**: Environment-specific knowledge for reproducible setup

**Format**:
```markdown
## Golden Environment
- **Platform**: [macOS 15.1, Apple Silicon M3 Max]
- **Python**: [3.12.0 via uv]
- **Key Dependencies**: [torch==2.5.0+metal, fastapi==0.104.1]
- **Hardware**: [32GB RAM, 40-core GPU]
- **Environment Variables**: [MPS_MEMORY_FRACTION=0.8, OMP_NUM_THREADS=8]

## Replicable Setup Checklist
- [ ] Install uv package manager
- [ ] Clone repo and cd to project root  
- [ ] `uv venv .venv && source .venv/bin/activate`
- [ ] `uv pip install -r requirements.txt`
- [ ] Set environment variables from .env.example
- [ ] Run `pytest tests/test_smoke.py` to verify setup
- [ ] Start Ollama and pull nomic-embed-text:137m-v1.5-fp16

## Known Pitfalls to Avoid Next Run
- **M3 Max MPS warmup**: First embedding call takes 2-5s, add warmup batch
- **Conway pattern edge effects**: Use minimum 16x16 grid for stable gliders  
- **WebSocket frame drops**: Limit broadcast payload to <10KB at >5Hz tick rate
- **Pytest import conflicts**: Use `python -m pytest` not `pytest` command
```

### ISSUE.md Template
**Purpose**: Structured failure reporting for human intervention

**Format**:
```markdown
---
title: "[FAILURE] [Gate/Phase] Brief Description"
date: "YYYY-MM-DD HH:MM:SS"
severity: "P0|P1|P2|P3"  
assigned_to: "human"
status: "open"
---

## Failure Summary
**Phase**: [Plan|Build|Validate|Review|Release]
**Gate Failed**: [unit|lint|type|docs|perf|security]
**Task**: [Brief task description]

## Reproduction Steps
1. [Step 1]
2. [Step 2]  
3. [Step 3]

## Error Details
\`\`\`
[Full error message/stack trace]
\`\`\`

## Environment Context
- **Commit**: [git sha]
- **Branch**: [branch name]
- **Python**: [version]
- **Platform**: [OS details]
- **Modified Files**: [list of changed files]

## Logs Collected
- [ ] Application logs: `logs/app-YYYY-MM-DD.log`
- [ ] Test output: `test-results.xml`  
- [ ] Performance metrics: `perf-baseline.json`
- [ ] Memory usage: `memory-profile.txt`

## Living Docs Updated
- [ ] TROUBLESHOOTING.md entry added
- [ ] REPLICATION-NOTES.md updated with new pitfalls
- [ ] Relevant configuration documented

## Next Steps Required
- [ ] Human investigation needed
- [ ] Architecture decision required
- [ ] External dependency issue
- [ ] Resource constraint (memory/compute)

## Blocking Dependencies
[List any dependencies preventing resolution]
```

---

## Failure Handling Protocol

When ANY gate fails or uncertainty arises:

### 1. Capture Logs
```bash
# Application logs
cp logs/current.log "logs/failure-$(date +%Y%m%d-%H%M%S).log"

# Test results  
pytest --junit-xml=test-results.xml || true

# System state
ps aux | grep python > process-snapshot.txt
df -h > disk-usage.txt
free -m > memory-usage.txt  # Linux
vm_stat > memory-usage.txt  # macOS

# Git context
git log --oneline -10 > git-context.txt
git status --porcelain > git-status.txt
```

### 2. Update TROUBLESHOOTING.md
```bash
# Add new entry or update existing
echo "### New Issue $(date)" >> TROUBLESHOOTING.md
echo "**Context**: [Fill in context]" >> TROUBLESHOOTING.md
echo "**Symptom**: [Describe what happened]" >> TROUBLESHOOTING.md
# ... complete template
```

### 3. Update REPLICATION-NOTES.md  
```bash
# Add to Known Pitfalls section
echo "- **New Pitfall**: [Description and avoidance]" >> REPLICATION-NOTES.md
```

### 4. Open ISSUE.md
```bash
# Generate issue file
ISSUE_FILE="ISSUE-$(date +%Y%m%d-%H%M%S).md"
cp templates/ISSUE-template.md "$ISSUE_FILE"
# Fill in template with failure details
```

### 5. Halt and Wait for Human
```bash
echo "FAILURE DETECTED - HALTING FOR HUMAN INPUT"
echo "Issue file: $ISSUE_FILE" 
echo "Review logs and living docs before proceeding"
exit 1
```

---

## CyberMesh-Specific Gates

### Phase-Specific Performance Thresholds
```yaml
# tests/performance_thresholds.yaml
phase_1_ca_grid:
  glider_propagation: 
    max_latency_ms: 10
    accuracy_threshold: 0.95
  pattern_detection:
    max_latency_ms: 50
    false_positive_rate: 0.05

phase_2_shard_routing:
  single_hop:
    success_rate: 0.95
    max_hops: 10  
  multi_hop_3x3:
    success_rate: 0.90
    path_stretch: 1.5

phase_3_integration:  
  ca_shard_tracking:
    max_distance: 2.5
    tracking_accuracy: 0.85

phase_7_dashboard:
  websocket_latency_ms: 50
  frame_rate_min_fps: 1.0
  max_payload_kb: 10
```

### Memory Budget Gates
```bash
# Memory monitoring during tests
python scripts/memory_monitor.py &
MONITOR_PID=$!

# Run test phase
pytest tests/test_phase_X.py

# Check memory usage
kill $MONITOR_PID
python scripts/check_memory_budget.py --threshold=6GB --phase=X
```

### Event Log Validation
```bash
# Verify event log schema compliance
python scripts/validate_event_logs.py logs/events.jsonl

# Check log compression ratios
python scripts/check_compression_ratio.py --baseline=full_state_dumps --compressed=event_logs
```

---

## Tool Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = 
    --strict-markers
    --tb=short
    --maxfail=3
    --durations=10
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, dependencies)
    performance: Performance benchmarks  
    hardware: Hardware-dependent tests (M3 Max specific)
```

### pyproject.toml
```toml
[tool.ruff]
line-length = 100
target-version = "py312"
select = ["E", "F", "W", "C", "N", "D", "UP", "S"]
ignore = ["D203", "D213"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "scripts/*"]

[tool.coverage.report] 
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

---

## Success Criteria per Phase

### Phase 0: Setup & Infrastructure
- [ ] All validation gates working
- [ ] Living docs initialized
- [ ] CI pipeline functional
- [ ] Development environment reproducible

### Phase 1: CA Grid + Conway Rules  
- [ ] Glider propagation test passes
- [ ] Pattern detection accuracy >95%
- [ ] Energy field decay working
- [ ] Memory usage <1GB

### Phase 2: Shard Routing with Embeddings
- [ ] Single-hop delivery ≥95% success rate
- [ ] Multi-hop 3×3 grid ≥90% success
- [ ] Embedding latency <50ms per batch
- [ ] No memory leaks in 1-hour stress test

### Phase 3: CA + Shard Integration
- [ ] Shards track glider within 2.5 cells
- [ ] Event logging captures all hops
- [ ] Integration stable for 10-minute runs
- [ ] Memory usage <3GB for 5×5 grid

### Phase 4: LoRA Pulse Layer
- [ ] LoRA pulse increases pattern persistence by ≥30%
- [ ] Pulse application latency <100ms
- [ ] Adapter switching working correctly
- [ ] Memory usage <5GB including LoRA adapters

### Phase 5: Load Test - Parallel Shards  
- [ ] Zero drops for K≤16 concurrent shards
- [ ] p95 latency ≤2× single-shard median
- [ ] Throughput scaling validates theoretical limits
- [ ] System recovers gracefully from overload

### Phase 6: Scalar Energy Hint
- [ ] ≥10% path length reduction with energy hints  
- [ ] Compute overhead ≤5% vs baseline
- [ ] Energy trails decay correctly
- [ ] No routing loops or deadlocks

### Phase 7: Real-Time Web Dashboard
- [ ] Dashboard renders 5×5 grid at 1-2 FPS
- [ ] WebSocket latency <50ms
- [ ] Bot state changes visible (color coding)
- [ ] Glider detection overlay working
- [ ] No frame drops over 5-minute demo
- [ ] Event correlation between backend and frontend

---

## Emergency Procedures

### Critical Failure (P0)
1. Stop all automated processes immediately
2. Preserve all logs and state
3. Document failure in ISSUE.md with P0 severity  
4. Update TROUBLESHOOTING.md with emergency info
5. Notify human immediately - do not attempt auto-recovery

### Performance Degradation (P1)
1. Collect performance metrics
2. Check memory usage and system resources
3. Document in REPLICATION-NOTES.md if environment-specific
4. Attempt graceful degradation (reduce batch size, tick rate)
5. If degradation persists, escalate to P0

### Minor Issues (P2-P3)
1. Document in TROUBLESHOOTING.md
2. Attempt standard fixes from troubleshooting guide
3. Continue with task if fixes successful
4. Update REPLICATION-NOTES.md with lessons learned

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-27 | Initial policy for CyberMesh POC development |