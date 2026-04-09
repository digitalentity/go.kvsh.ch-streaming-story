// Package story provides a streaming story tracker: a hybrid clustering
// system that ingests a continuous stream of signals and groups them into
// evolving stories.
//
// Ingestion is two-phase:
//
//   - Real-time Draft phase: each signal is immediately assigned to the
//     nearest story centroid via cosine similarity.
//   - Periodic Refinement phase: a background batch run applies HDBSCAN
//     to re-cluster recent signals, resolving splits, merges, and
//     misassignments from the Draft phase.
//
// Signal IDs are UUID v5. Callers should derive them using the exported
// TrackerNamespace:
//
//	id := uuid.NewSHA1(story.TrackerNamespace, []byte(domainKey))
//
// Create a Tracker by supplying a Config with at least a Store and Codec:
//
//	t, err := story.NewTracker(story.Config[MyData]{
//	    Store: myStore,
//	    Codec: myCodec,
//	})
package story
