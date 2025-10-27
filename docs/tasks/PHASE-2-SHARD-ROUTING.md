---
title: "Phase 2: Shard Routing with Embeddings"
description: "Implement vector shard passing and routing mechanics using embeddings"
version: "1.0"
phase: 2
status: "planned"
date: "2025-10-27"
assignee: "cybermesh-dev"
priority: "high"
---

## Executive Summary

Phase 2 introduces the second tier of CyberMesh: vector shards that represent memory fragments carried by mobile glider patterns. This phase implements the core routing mechanics where gliders transport embedding vectors across the CA substrate, enabling decentralized memory distribution and retrieval.

## Success Criteria

### Functional Requirements
- [ ] **Single-hop delivery**: Shards transport between adjacent CA cells ≥95% success rate
- [ ] **Multi-hop routing**: Shards navigate through 3×3 grid with ≥90% success rate
- [ ] **Vector persistence**: Embedding vectors maintained during glider movement
- [ ] **Routing efficiency**: Path finding based on embedding similarity
- [ ] **Memory integrity**: No vector corruption during transport

### Performance Requirements
- [ ] **Embedding latency**: Vector generation/comparison <50ms per operation
- [ ] **Routing overhead**: Additional computation <20% over base CA updates
- [ ] **Scalability**: Performance scales with K concurrent shards (K≤16)
- [ ] **Memory usage**: <2GB total for active routing

### Quality Requirements
- [ ] **Vector accuracy**: Cosine similarity preserved within routing operations
- [ ] **Path optimization**: Routes minimize energy cost distance
- [ ] **Error handling**: Graceful degradation on failures with logging
- [ ] **Debuggability**: Full shard tracking and event logging

---

## Detailed Requirements

### Shard Data Model

**Shard Structure:**
```python
@dataclass
class VectorShard:
    """Memory fragment with embedding vector and metadata."""
    shard_id: str
    embedding: np.ndarray  # Nomic-embed-text vector (768 dims typical)
    destination: Tuple[int, int]  # Target (x,y) coordinates
    ttl: int  # Time-to-live in steps
    priority: float  # Routing priority (0.0-1.0)
    metadata: Dict[str, Any]  # Additional routing hints
```

**Vector Generation:**
- Use nomic-embed-text:137m-v1.5-fp16 via Ollama
- Normalize vectors for cosine similarity comparisons
- Cache embeddings to avoid redundant computation

### Glider-Shard Binding

**Attachment Mechanism:**
- Shards "surf" on glider patterns using energy gradients
- Single glider carries exactly one shard
- Shard movement synchronized with glider motion
- Detachment occurs at destination or TTL expiry

**Position Tracking:**
- Shard position = glider center of mass coordinate
- Sub-cell precision maintained through interpolation
- Boundary conditions prevent shard loss

### Routing Algorithms

**Distance-Based Routing:**
```
# Vector similarity as routing cost
cost = 1 - cosine_similarity(current_embedding, target_embedding)
route = minimize_path_cost(cost_matrix)
```

**Hop-by-Hop Forwarding:**
1. At each intersection, query neighbor cells
2. Compute embedding similarities to shard destination
3. Select lowest-cost neighbor with available capacity
4. Update shard position and decrement TTL

**Route Optimization:**
- A* search with vector similarity heuristics
- Energy field gradient following (Phase 3 integration prep)
- Avoid congestion through capacity limits

### Performance Characteristics

**Benchmark Targets:**
- Single-hop delivery: <10ms end-to-end
- Multi-hop routing (3x3): <100ms total
- Embedding computation: <50ms per shard
- Vector comparison: <1ms per similarity check

### Implementation Architecture

```
src/
├── routing/
│   ├── shard.py          # VectorShard data model
│   ├── carrier.py        # Glider-shard binding logic
│   └── router.py         # Path finding and forwarding
├── embeddings/
│   ├── generator.py      # Ollama integration
│   ├── cache.py          # Embedding storage and retrieval
│   └── similarity.py     # Vector distance metrics
├── simulation/
│   ├── coordinator.py    # Multi-shard orchestration
│   └── metrics.py        # Performance and success tracking
└── events/
    ├── types.py          # Shard event definitions
    └── logger.py         # Event capture and persistence
```

---

## Task Breakdown (5 Days)

### Day 1: Shard Data Model & Embedding Infrastructure
- [ ] Implement VectorShard dataclass with validation
- [ ] Create Ollama client wrapper for nomic-embed-text
- [ ] Implement embedding caching with LRU eviction
- [ ] Add vector normalization and similarity functions
- [ ] Unit tests for embedding operations

### Day 2: Glider-Shard Binding
- [ ] Implement carrier attachment/detachment logic
- [ ] Add position synchronization with glider tracking
- [ ] Create shard inventory management per grid cell
- [ ] Handle boundary conditions and TTL management
- [ ] Integration tests with Phase 1 glider detection

### Day 3: Single-Hop Routing
- [ ] Implement neighbor discovery algorithm
- [ ] Create hop-by-hop forwarding logic
- [ ] Add capacity constraints (max shards per cell)
- [ ] Implement success rate tracking (≥95% target)
- [ ] Performance benchmarking per shard operation

### Day 4: Multi-Hop Path Finding
- [ ] Implement A* routing with vector similarity heuristics
- [ ] Create 3x3 grid routing scenarios
- [ ] Add route optimization and congestion avoidance
- [ ] Error handling for routing failures
- [ ] Comprehensive success rate validation (≥90%)

### Day 5: Multi-Shard Orchestration & Optimization
- [ ] Implement concurrent shard simulation (K≤16)
- [ ] Add performance monitoring and bottleneck detection
- [ ] Memory usage optimization and leak prevention
- [ ] Integration with existing CA simulation
- [ ] Full system validation against Phase 2 targets

---

## Validation Gates Checklist

### Unit Testing Gate
- [ ] pytest tests/test_shard_*.py - All shard operations function correctly
- [ ] pytest tests/test_embedding_*.py - Ollama integration and caching work
- [ ] pytest tests/test_routing_*.py - Path finding algorithms validated
- [ ] pytest tests/test_orchestration_*.py - Multi-shard scenarios stable

### Linting Gate
- [ ] ruff check src/ tests/ - Zero violations in new code
- [ ] All new functions have comprehensive docstrings

### Type Checking Gate
- [ ] mypy src/cybermesh/ - Zero type errors in shard/routing modules
- [ ] Strict typing for embedding vectors and coordinate systems

### Documentation Gate
- [ ] python scripts/check_spec_drift.py - Implementation matches routing spec
- [ ] python scripts/check_docstrings.py - All routing APIs documented
- [ ] mkdocs build --strict - Documentation builds with routing sections

### Performance Gate
- [ ] pytest tests/test_performance.py::test_single_hop - <10ms per hop
- [ ] pytest tests/test_performance.py::test_multi_hop - <100ms for 3x3 routing
- [ ] pytest tests/test_performance.py::test_embedding_latency - <50ms per shard
- [ ] python scripts/check_memory_budget.py --phase=2 - <2GB for active routing

---

## Acceptance Criteria

### Functional Verification
1. **Single-Hop Test**: 1000 shard deliveries, ≥95% success rate logged
2. **Multi-Hop Test**: 500 shard routes through 3×3 grid, ≥90% completion
3. **Embedding Test**: Vector accuracy preserved through full routing cycle
4. **Concurrency Test**: K=16 concurrent shards without interference

### Quality Assurance
1. **Memory Safety**: No leaks during extended multi-shard simulations
2. **Error Handling**: Graceful failures with comprehensive logging
3. **Performance Consistency**: Benchmarks within ±10% of targets
4. **API Stability**: Clean interfaces for Phase 3 integration

### Integration Readiness
1. **Phase 1 Compatibility**: Seamlessly overlays on existing CA grid
2. **Event System**: Full shard lifecycle events logged in JSONL
3. **Configuration**: All parameters externalized to .env
4. **Metrics Export**: Performance and success data available for analysis

---

## Risk Assessment

### Technical Risks
- **Embedding Latency**: Ollama model loading may exceed 50ms target initially
- **Memory Scaling**: K=16 concurrent shards may hit 2GB limit early
- **Concurrency Conflicts**: Shard collisions leading to routing deadlocks
- **Vector Precision**: Floating point drift during similarity calculations

### Mitigation Strategies
- Implement embedding warmup and caching to hit latency targets
- Profile memory usage and implement shard limits per cell
- Add collision detection and backoff mechanisms in routing
- Use fixed-point arithmetic for critical precision operations

---

## Dependencies

### External Software
- **Ollama**: nomic-embed-text:137m-v1.5-fp16 model serving
- **numpy**: Vector operations and similarity calculations
- **psutil**: Memory monitoring for performance gate

### Internal Dependencies
- **Phase 1 Complete**: CA grid, glider detection, energy field foundations
- **Validation Gates**: All Phase 1 gates passing before Phase 2 development
- **Git History**: Clean commit history with atomic Phase 1 implementation

### Expected Ollama Performance
```
Model: nomic-embed-text:137m-v1.5-fp16
Warmup: ~2-5s initial load (per REPLICATION-NOTES.md)
Throughput: ~20-50 embeddings/second on M3 Max
Memory: ~500MB model footprint
```

---

## Success Sign-off

Phase 2 complete when:
- [ ] Single-hop delivery ≥95% success rate demonstrated
- [ ] Multi-hop 3×3 grid routing ≥90% success rate achieved
- [ ] Embedding operations <50ms latency with caching
- [ ] K≤16 concurrent shards simulated without failures
- [ ] All validation gates pass (unit/lint/type/docs/perf)
- [ ] Memory usage <2GB for all tested scenarios
- [ ] Code reviewed and integration tests pass
- [ ] Living documentation (TEST-MATRIX.md, REPLICATION-NOTES.md) updated
- [ ] Event logs capture full shard lifecycle

**Gate Decision**: Manual approval required before proceeding to Phase 3.
