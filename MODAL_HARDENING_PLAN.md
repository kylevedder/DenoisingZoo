# Modal Infrastructure Hardening Plan

**Status:** REVIEWED AND APPROVED

## Current Issues

From `EXPERIMENT_LOG.md`, multiple experiments failed due to Modal connection issues:
- `StreamTerminatedError: Connection lost`
- `GRPCError: App state is APP_STATE_STOPPED`
- `Function call has expired`
- Experiments interrupted after 5-35 minutes despite 4-hour timeout

## Root Causes

1. **Client connection required** - Jobs fail when launcher terminates
2. **4-hour timeout** - Too short for long experiments (20+ epochs on CIFAR-10)
3. **No retry mechanism** - Single failure kills the entire job
4. **No detached execution** - Jobs depend on client connection staying open

## Proposed Changes (Post-Review)

### 1. Increase Timeout to 12 Hours

Modal supports up to 24-hour timeouts, but 12 hours is more reasonable to avoid
burning GPU time on hung jobs. Retries provide additional coverage.

```python
# Before
timeout=14400  # 4 hours

# After
timeout=43200  # 12 hours
```

### 2. Add Automatic Retries with Delay

Use Modal's retry mechanism for automatic restarts on failure/preemption.
Use a 30-second delay (not 0) to avoid hitting the same transient issue.

```python
@app.function(
    timeout=43200,  # 12 hours
    retries=modal.Retries(
        max_retries=5,           # 5 retries = up to 60 hours total
        initial_delay=30.0,      # 30 second delay before retry
        backoff_coefficient=1.0, # Fixed delay (not exponential)
    ),
)
```

### 3. Use `modal run --detach` for True Detached Execution

**CORRECTION:** `.spawn().get()` is equivalent to `.remote()` - both block.
The correct pattern is `modal run --detach` which the launcher should use.

```python
# launcher.py
def run_modal_training(extra_args, run_name, detach=True):
    detach_flag = "--detach" if detach else ""
    cmd = f"modal run {detach_flag} scripts/modal_app.py -- {cli_args}"
```

### 4. Auto-Resume via `resume=true` Flag

**SIMPLIFICATION:** Instead of complex checkpoint detection logic, always pass
`resume=true` to training. The existing `resume_if_requested()` in `helpers.py`
already handles missing checkpoints gracefully.

```python
def train_on_modal(extra_args: list[str]) -> None:
    args = list(extra_args)
    # Always enable resume - safe even if no checkpoint exists
    if not any(a.startswith("resume=") for a in args):
        args.append("resume=true")
    # ... rest of function
```

### 5. Remove Periodic Volume Commit, Keep Final Commit

Modal Volumes auto-commit every few seconds. The periodic commit thread adds
complexity and may cause issues. Keep only the final commit for immediate visibility.

```python
# Remove:
# - Background commit thread
# - Periodic training_volume.commit() calls

# Keep:
subprocess.run(cmd, check=True)
training_volume.commit()  # Final commit for immediate visibility
```

### 6. Remove `single_use_containers`

**NOTE:** This flag is irrelevant for single-invocation functions like training.
It's meant for functions called repeatedly via `.map()`. Not needed here.

## Implementation Summary

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `timeout` | 14400 (4h) | 43200 (12h) | Longer, but not wasteful |
| `retries` | None | 5 @ 30s delay | Auto-restart on failure |
| Launcher | blocking | `--detach` | Jobs survive client death |
| Resume | Manual | Auto `resume=true` | Simpler, already robust |
| Volume commits | Periodic thread | Final only | Auto-commit handles rest |

## Testing Plan

1. **Short job (1 epoch)**: Verify basic completion with new config
2. **Detach test**: Start job with `--detach`, kill terminal, verify job continues
3. **Resume test**: Start job, wait for checkpoint, cancel, restart, verify resume
4. **Long job (20 epochs)**: Full end-to-end test with `--detach`

## Rollback Plan

If issues arise, revert to previous `modal_app.py`:
```bash
git checkout HEAD~1 -- scripts/modal_app.py
```

## Success Criteria

- Jobs survive client disconnection (verified via detach test)
- Jobs automatically resume from checkpoint on retry
- 20+ epoch experiments complete without manual intervention
- `python scripts/modal_app.py sync` works during and after jobs

## Review Notes

Reviewed by agent on 2026-01-21. Key corrections:
- `.spawn().get()` doesn't provide detached execution (use `modal run --detach`)
- `initial_delay=0.0` is risky (use 30 seconds)
- `single_use_containers` is unnecessary for single-invocation functions
- Checkpoint detection overcomplicated (just use `resume=true` always)

Sources:
- https://modal.com/docs/reference/modal.Function
- https://modal.com/docs/reference/modal.Retries
- https://modal.com/docs/guide/volumes
