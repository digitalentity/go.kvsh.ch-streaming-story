package hdbscan

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"testing"
)

type gf2 struct {
	MinClusterSize int         `json:"min_cluster_size"`
	MinSamples     int         `json:"min_samples"`
	Embeddings     [][]float32 `json:"embeddings"`
	Labels         []int       `json:"labels"`
}

func TestDiag2(t *testing.T) {
	files := []string{
		"golden_mcs2_ms1.json",
		"golden_mcs2_ms2.json",
		"golden_mcs5_ms1.json",
		"golden_mcs5_ms5.json",
	}
	for _, fname := range files {
		raw, _ := os.ReadFile("testdata/" + fname)
		var fix gf2
		json.Unmarshal(raw, &fix)
		pts := fix.Embeddings
		n := len(pts)

		dist := pairwiseCosine(pts)
		core := coreDistances(dist, n, fix.MinSamples)
		mst := primMST(dist, core, n)
		sort.Slice(mst, func(i, j int) bool { return mst[i].w < mst[j].w })
		dn := buildDendrogram(mst, n)
		clusters, pointFallout := condense(dn, n, fix.MinClusterSize)
		computeStability(clusters, pointFallout)
		selected := selectClusters(clusters)

		// Count what we'd label
		labels := make([]int, n)
		for i := range labels { labels[i] = -1 }
		labelID := 0
		for i := range clusters {
			if selected[i] {
				labelSubtree(clusters, i, labelID, labels, pointFallout)
				labelID++
			}
		}
		goNoise := 0
		for _, l := range labels { if l == -1 { goNoise++ } }

		// Also count if we label ALL fallout (including isNoise=true)
		labels2 := make([]int, n)
		for i := range labels2 { labels2[i] = -1 }
		labelID2 := 0
		for i := range clusters {
			if selected[i] {
				labelSubtreeAll(clusters, i, labelID2, labels2, pointFallout)
				labelID2++
			}
		}
		goNoise2 := 0
		for _, l := range labels2 { if l == -1 { goNoise2++ } }

		refNoise := 0
		for _, l := range fix.Labels { if l == -1 { refNoise++ } }

		fmt.Printf("%s: ref_noise=%d go_noise=%d go_noise_with_all=%d\n",
			fname, refNoise, goNoise, goNoise2)
	}
}

func labelSubtreeAll(clusters []cCluster, idx int, labelID int, labels []int, pointFallout []fallout) {
	for _, f := range pointFallout {
		if f.clusterIdx == idx {
			labels[f.pointIdx] = labelID
		}
	}
	for _, childIdx := range clusters[idx].children {
		labelSubtreeAll(clusters, childIdx, labelID, labels, pointFallout)
	}
}
