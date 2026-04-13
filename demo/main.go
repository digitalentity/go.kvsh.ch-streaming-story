// Demo reads a news-signal JSON file, feeds signals one by one into an
// accumulation buffer, and runs HDBSCAN batch re-clustering every N signals
// (default 50). After each batch it prints the discovered clusters with a few
// representative samples per cluster (closest to the centroid) and their
// cosine distance.
//
// Usage:
//
//	go run ./demo [flags] [path-to-json]
//
// Flags:
//
//	-batch-every N        trigger a batch every N signals (default 50)
//	-min-cluster-size N   HDBSCAN minClusterSize (default 2)
//	-min-samples N        HDBSCAN minSamples; 1 works well for sparse
//	                      high-dimensional embeddings (default 1)
//	-samples-per-cluster  signals to print per cluster (default 4)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"go.kvsh.ch/streaming-story/internal/hdbscan"
)

// inputSignal mirrors the JSON structure of the test-data file.
type inputSignal struct {
	ID        string            `json:"id"`
	Timestamp time.Time         `json:"timestamp"`
	Source    string            `json:"source"`
	Author    string            `json:"author"`
	Title     string            `json:"title"`
	Content   string            `json:"content"`
	Metadata  map[string]string `json:"metadata"`
	Embedding []float32         `json:"embedding"`
}

func main() {
	batchEvery := flag.Int("batch-every", 50, "run a batch every N signals")
	minClusterSize := flag.Int("min-cluster-size", 2, "HDBSCAN minClusterSize")
	minSamples := flag.Int("min-samples", 1, "HDBSCAN minSamples (1 = use nearest-neighbour core distance; good for sparse high-dim data)")
	samplesPerCluster := flag.Int("samples-per-cluster", 4, "signals to print per cluster")
	flag.Parse()

	path := "demo/testdata-1h.json"
	if flag.NArg() > 0 {
		path = flag.Arg(0)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", path, err)
		os.Exit(1)
	}

	var signals []inputSignal
	if err := json.Unmarshal(raw, &signals); err != nil {
		fmt.Fprintf(os.Stderr, "parse: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d signals from %s\n", len(signals), path)
	fmt.Printf("Settings: batch-every=%d  min-cluster-size=%d  min-samples=%d  samples-per-cluster=%d\n",
		*batchEvery, *minClusterSize, *minSamples, *samplesPerCluster)

	var buf []inputSignal
	batchNum := 0

	for i, sig := range signals {
		buf = append(buf, sig)

		if (i+1)%*batchEvery == 0 || i == len(signals)-1 {
			batchNum++
			runBatch(batchNum, buf, *minClusterSize, *minSamples, *samplesPerCluster)
		}
	}
}

// runBatch runs HDBSCAN on all accumulated signals and prints the results.
func runBatch(batchNum int, sigs []inputSignal, minClusterSize, minSamples, samplesPerCluster int) {
	pts := make([][]float32, len(sigs))
	for i, s := range sigs {
		pts[i] = s.Embedding
	}

	labels, err := hdbscan.Cluster(pts, minClusterSize, minSamples)
	if err != nil {
		fmt.Printf("\n=== Batch %d (%d signals) — clustering error: %v ===\n", batchNum, len(sigs), err)
		return
	}

	// Group point indices by cluster label; -1 is noise.
	byCluster := map[int][]int{}
	for i, lbl := range labels {
		byCluster[lbl] = append(byCluster[lbl], i)
	}

	// Collect and sort cluster IDs (noise last).
	clusterIDs := make([]int, 0, len(byCluster))
	for lbl := range byCluster {
		clusterIDs = append(clusterIDs, lbl)
	}
	sort.Slice(clusterIDs, func(i, j int) bool {
		a, b := clusterIDs[i], clusterIDs[j]
		if a == -1 {
			return false
		}
		if b == -1 {
			return true
		}
		return a < b
	})

	noiseCount := len(byCluster[-1])
	clusterCount := len(clusterIDs)
	if noiseCount > 0 {
		clusterCount-- // don't count noise as a cluster
	}

	fmt.Printf("\n╔══════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  Batch %d — %d signals ingested, %d clusters, %d noise\n",
		batchNum, len(sigs), clusterCount, noiseCount)
	fmt.Printf("╚══════════════════════════════════════════════════════════════╝\n")

	for _, lbl := range clusterIDs {
		indices := byCluster[lbl]

		if lbl == -1 {
			fmt.Printf("\n  [noise: %d signals]\n", len(indices))
			continue
		}

		// Compute centroid.
		centroid := computeCentroid(pts, indices)

		// Sort members by cosine distance to centroid.
		type member struct {
			idx  int
			dist float64
		}
		members := make([]member, len(indices))
		for i, idx := range indices {
			members[i] = member{idx: idx, dist: cosineDist(pts[idx], centroid)}
		}
		sort.Slice(members, func(i, j int) bool { return members[i].dist < members[j].dist })

		fmt.Printf("\n  ── Cluster %d (%d signals) ──\n", lbl, len(indices))

		n := samplesPerCluster
		if n > len(members) {
			n = len(members)
		}
		for _, m := range members[:n] {
			sig := sigs[m.idx]
			ts := sig.Timestamp.Format("01-02 15:04")
			fmt.Printf("    [dist=%.4f] [%s] [%s] %s\n", m.dist, ts, sig.Source, sig.Title)
		}
	}
	fmt.Println()
}

// computeCentroid returns the mean embedding of the points at indices.
func computeCentroid(pts [][]float32, indices []int) []float32 {
	if len(indices) == 0 {
		return nil
	}
	dim := len(pts[indices[0]])
	c := make([]float32, dim)
	for _, idx := range indices {
		for d, v := range pts[idx] {
			c[d] += v
		}
	}
	n := float32(len(indices))
	for d := range c {
		c[d] /= n
	}
	return c
}

// cosineDist returns cosine distance (1 − similarity) between a and b.
func cosineDist(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		na += ai * ai
		nb += bi * bi
	}
	if na == 0 || nb == 0 {
		return 1
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if cos > 1 {
		cos = 1
	} else if cos < -1 {
		cos = -1
	}
	return 1 - cos
}
