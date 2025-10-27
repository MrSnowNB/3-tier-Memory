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
```
[Full error message/stack trace]
```

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
