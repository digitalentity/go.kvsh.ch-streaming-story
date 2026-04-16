# Implementation Plan: Streaming Story Tracker

This plan outlines the steps required to finalize the `go.kvsh.ch/streaming-story` library according to the specifications in `DESIGN.md`.

## Phase 1: Core Infrastructure

- [x] **Core Types**: `Signal`, `StoryMeta`, `StoryEvent`, `StoryState` defined.
- [x] **Config**: Full configuration with validation and defaults implemented.
- [x] **UUID Namespace**: Fixed `TrackerNamespace` and configuration-based override implemented.
- [x] **Key Schema**: KV key generation helpers for all prefixes implemented.
- [x] **Persistence Models**: Internal `storyRecord` and `calibState` defined.
- [x] **Store Interface**: Generic `Store` and `Tx` interfaces defined.
- [x] **In-memory Mock Store**: `memStore` implemented for testing.
- [x] **Algorithms**: `hdbscan` and `hungarian` algorithms implemented with tests.
- [x] **Tracker Lifecycle**: `NewTracker`, `Close`, and subscriber management implemented.
- [ ] **Optimized Vector Math**: Integrate `gonum.org/v1/gonum` for SIMD-accelerated `float32` operations (via `blas32`).

## Phase 2: Draft Phase (Real-time Ingestion)

- [ ] **Distance Metrics**: Implement `cosineSimilarity(a, b []float32) float64`. (To be optimized with `gonum`)
- [ ] **Story Selection**: Implement `findNearestStory(sig Signal[T]) (StoryMeta, float64, error)`.
  - [ ] Use `t:{unix_sec}:{storyID}` index to find "Active Context" (Tier 3) stories (default 30 days).
  - [ ] Calculate distances to all Active/Dormant story centroids.
- [ ] **Thresholding**: Implement `calcThreshold(story StoryMeta) float64`.
  - [ ] Use $T_{assign}(story) = mean\_distance(story) + AssignmentK \times \sigma(story)$.
  - [ ] Handle **Cold Start**: Use `AssignmentK * sigmaGlobal` if story signals < `ColdStartMinSignals`.
  - [ ] Handle **Sigma Floor**: Floor $\sigma(story)$ at `SigmaFloor * sigmaGlobal`.
  - [ ] Handle **Dormant Stories**: Use `FrozenMeanDistance` and `FrozenSigma`.
- [ ] **Ingest Logic**:
  - [x] Establish/validate dimensionality.
  - [ ] Perform nearest story lookup.
  - [x] If `applyInProgress`, write to `ingestBuffer`.
  - [ ] Else, write to store (signal and updated story metadata).
  - [ ] Emit `EventDraftAssigned`.

## Phase 3: Refinement Phase (Batch Processing)

- [ ] **Batch Loop**: Complete `batchLoop` and `runBatch` skeleton.
- [ ] **Collection & Sampling**:
  - [ ] Collect signals from `BatchWindow`.
  - [ ] Collect outliers within `OutlierTTL`.
  - [ ] Implement two-pass stratified reservoir sampling down to `BatchSampleCap`.
- [ ] **HDBSCAN Run**: Integrate `internal/hdbscan` with collected signals.
- [ ] **Cluster Mapping (Phase 1)**:
  - [ ] Build Jaccard cost matrix (cost = 1 - Jaccard).
  - [ ] Restrict to signals within `BatchWindow`.
  - [ ] Run Hungarian algorithm (`internal/hungarian`).
- [ ] **Cluster Mapping (Phase 2)**:
  - [ ] Detect splits (N-way) for unmatched batch clusters.
  - [ ] Detect merges (N-way) for unmatched persistent stories.
  - [ ] Rule: Oldest `StoryID` survives.
- [ ] **Apply Phase**:
  - [x] Set `applyInProgress` flag during Apply.
  - [ ] Persist all updates in a single `Update` transaction.
    - [ ] Update story centroids and radii.
    - [ ] Migrate re-assigned signals (within `BatchWindow`).
    - [ ] Migrate merged story signals (key-space migration).
    - [ ] Create new stories for unmatched clusters.
    - [ ] Promote outliers to stories.
    - [ ] Evict stale outliers.
    - [ ] Transition stories to Dormant/Archived based on `SilenceWindow`/`ArchiveWindow`.
  - [ ] Update `sigmaGlobal` using EMA.
  - [x] Clear `applyInProgress` and drain `ingestBuffer`.
  - [ ] Emit `EventBatchComplete` and all change events.

## Phase 4: Iterators & Public API

- [ ] **Go 1.22 Iterators**:
  - [ ] Implement `Stories(state StoryState) iter.Seq[StoryMeta]`.
  - [ ] Implement `SignalsOf(storyID uuid.UUID) iter.Seq2[Signal[T], error]`.
- [ ] **Story Lookup**: Complete `Story(id uuid.UUID)` implementation.

## Phase 5: Verification & Testing

- [ ] **Unit Tests**:
  - [ ] Test distance metrics.
  - [ ] Test sampling logic.
  - [ ] Test cluster mapping (Hungarian + Phase 2 splits/merges).
- [ ] **Integration Tests**:
  - [ ] Full Ingest -> Batch cycle.
  - [ ] Story lifecycle transitions.
  - [ ] Signal re-assignment validation.
- [ ] **Benchmarks**:
  - [ ] Ingest latency during Batch Apply (buffer behavior).
  - [ ] Batch performance with `BatchSampleCap` signals.
