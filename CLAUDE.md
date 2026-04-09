# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is a **pre-implementation** Go library. The full design is in `DESIGN.md`. Read it before writing any code — most algorithmic questions are already answered there.

Module: `go.kvsh.ch/streaming-story`  
Go version: 1.22 (uses `iter.Seq` / `iter.Seq2` from the standard library range-over-func experiment)

## UNDERCOVER MODE — CRITICAL

You are operating UNDERCOVER in a PUBLIC/OPEN-SOURCE repository. Your commit messages, PR titles, and PR bodies MUST NOT contain ANY Anthropic-internal information.
Do not blow your cover. Do not reveal yourself. Do not add Claude signature to commit messages.

## Commands

```bash
go build ./...
go test ./...
go test -run TestName ./pkg/...   # single test
go vet ./...
```

No Makefile or custom tooling exists yet.

## Architecture

The library is a hybrid clustering system: a **Draft phase** (real-time, per-signal) and a **Refinement phase** (periodic batch, HDBSCAN). Read DESIGN.md in full before implementing — the sections below flag the decisions that are most likely to be misimplemented.

### Signal flow

1. `Ingest` → cosine-similarity nearest-centroid lookup → assign or outlier-bucket
2. Background goroutine fires every `BatchInterval` → HDBSCAN → cluster mapping → KV apply → emit events

### Cluster mapping (two-phase)

Phase 1 uses the Hungarian algorithm for optimal 1-to-1 continuity (cost = 1 − Jaccard). Phase 2 scans the full unmatched set for splits and merges. Both phases use Jaccard over **BatchWindow-scoped signals only** — not lifetime signals — to avoid the denominator blow-up on mature stories.

For N-way merges, the **oldest StoryID survives** (earliest creation time). If the secondary story is older than the primary, the survivor/retired labels flip.

### Sampling (two-pass)

When `len(signals) > BatchSampleCap`, sampling is two-pass:
1. **Guaranteed pass**: `MinClusterSize` signals per Active story, capped at `SampleGuaranteeMaxFraction` (0.5) × `BatchSampleCap` total. If the budget is exceeded, per-story reservations scale down proportionally (floor 1).
2. **Proportional pass**: remaining capacity distributed by signal count.

### Stability rule

Re-assignment is scoped to `BatchWindow`. Signals older than `BatchWindow` are never moved by a batch run. **Exception**: a merge is a key-space migration (all signal keys move under the surviving story's prefix, including historical ones) — this is exempt from the stability rule.

### Dormant story thresholds

Dormant stories have no live signals in the window, so `mean_distance` and `σ` are undefined. They are **frozen in metadata** on the Dormant transition and used for Draft-phase threshold calculation. On reactivation they are **cleared** and the story re-enters the cold-start period (falls back to `σ_global`).

### Outlier TTL reference point

Outliers are evicted when `At < lastBatchTimestamp − OutlierTTL`. The reference is `lastBatchTimestamp`, not wall-clock `now`, so a maintenance pause does not cause mass eviction.

### Concurrency

`bbolt` is single-writer. During the Apply phase an `applyInProgress` flag redirects `Ingest` calls into an in-memory `ingestBuffer` (bounded channel). The batch goroutine drains the buffer in a follow-up transaction. This is **at-most-once**: a crash between the Apply commit and the drain loses buffered signals.

### KV key schema

| Prefix | Content |
|---|---|
| `c:state` | `σ_global`, dimensionality, last batch timestamp |
| `s:{storyID}:m` | Story metadata (centroid, radius, state, timestamps, frozen stats) |
| `s:{storyID}:s:{signalID}` | Signal data |
| `o:{signalID}` | Outlier signal |
| `t:{unix_sec}:{storyID}` | Time index for efficient Tier 3 range scans |

### Resolved design decisions (do not re-open)

- `MinClusterSize` is a **fixed config constant** — not derived from window population.
- `StabilityWindow` is **removed** — `BatchWindow` is the sole re-assignment scope.
- Signal UUID namespace is a **fixed compile-time constant** (`TrackerNamespace`) — not derived from the store path.
- Default windows are calibrated for **news-frequency ingestion** (1–10 signals/day per topic). See the tuning note in DESIGN.md's Public API section for high-frequency alternatives.
