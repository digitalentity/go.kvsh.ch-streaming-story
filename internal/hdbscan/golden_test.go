package hdbscan_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kvsh.ch/streaming-story/internal/hdbscan"
)

// goldenFixture is the JSON structure written by gen_golden.py.
type goldenFixture struct {
	Description            string      `json:"description"`
	Source                 string      `json:"source"`
	N                      int         `json:"n"`
	MinClusterSize         int         `json:"min_cluster_size"`
	MinSamples             int         `json:"min_samples"`
	ClusterSelectionMethod string      `json:"cluster_selection_method"`
	IDs                    []string    `json:"ids"`
	Titles                 []string    `json:"titles"`
	Embeddings             [][]float32 `json:"embeddings"`
	Labels                 []int       `json:"labels"`
}

// TestGolden_ARI loads every golden_*.json fixture and verifies that our
// HDBSCAN implementation produces labels with Adjusted Rand Index ≥ 0.90
// against the reference Python output.
//
// The threshold is intentionally lenient: minor algorithmic differences
// (MST tie-breaking, floating-point order) can flip borderline points without
// meaningfully changing the partition.  An ARI of 0.90 still requires very
// strong structural agreement.
func TestGolden_ARI(t *testing.T) {
	fixtures, err := filepath.Glob(filepath.Join("testdata", "golden_*.json"))
	require.NoError(t, err)
	require.NotEmpty(t, fixtures, "no golden_*.json fixtures found in testdata/")

	for _, path := range fixtures {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			require.NoError(t, err, "read fixture")

			var fix goldenFixture
			require.NoError(t, json.Unmarshal(raw, &fix), "parse fixture")
			require.Len(t, fix.Embeddings, fix.N, "fixture embedding count")
			require.Len(t, fix.Labels, fix.N, "fixture label count")

			got, err := hdbscan.Cluster(fix.Embeddings, fix.MinClusterSize, fix.MinSamples)
			require.NoError(t, err)
			require.Len(t, got, fix.N)

			ari := adjustedRandIndex(fix.Labels, got)
			t.Logf("ARI=%.4f  (mcs=%d ms=%d, n=%d, ref clusters=%d, got clusters=%d)",
				ari,
				fix.MinClusterSize, fix.MinSamples, fix.N,
				clusterCount(fix.Labels), clusterCount(got),
			)
			assert.GreaterOrEqualf(t, ari, 0.90,
				"ARI below threshold for %s\ndescription: %s",
				filepath.Base(path), fix.Description,
			)
		})
	}
}

// clusterCount returns the number of distinct non-noise cluster labels.
func clusterCount(labels []int) int {
	seen := make(map[int]struct{})
	for _, l := range labels {
		if l >= 0 {
			seen[l] = struct{}{}
		}
	}
	return len(seen)
}

// adjustedRandIndex computes the Adjusted Rand Index between two label
// vectors a and b.  Both must have the same length.  Labels may be any
// integers; −1 is treated as noise and counted as its own singleton class
// (one per point), following the convention that noise points are never
// considered to be in the same cluster.
//
// Formula (contingency-table form):
//
//	ARI = (Σ C(n_ij,2) − [Σ C(a_i,2)·Σ C(b_j,2)] / C(n,2))
//	    / (½·[Σ C(a_i,2) + Σ C(b_j,2)] − [Σ C(a_i,2)·Σ C(b_j,2)] / C(n,2))
//
// where C(k,2) = k*(k−1)/2.
func adjustedRandIndex(a, b []int) float64 {
	n := len(a)
	if n == 0 {
		return 1.0
	}

	// Assign each noise point (−1) in a and b its own unique class so that
	// two noise points are never considered co-clustered.
	nextNoise := -2
	aa := make([]int, n)
	bb := make([]int, n)
	for i := range a {
		if a[i] == -1 {
			aa[i] = nextNoise
			nextNoise--
		} else {
			aa[i] = a[i]
		}
	}
	nextNoise = -2
	for i := range b {
		if b[i] == -1 {
			bb[i] = nextNoise
			nextNoise--
		} else {
			bb[i] = b[i]
		}
	}

	// Build a → unique index and b → unique index maps.
	aIdx := labelIndex(aa)
	bIdx := labelIndex(bb)
	ra, rb := len(aIdx), len(bIdx)

	// Contingency matrix n[i][j] = count of points with a-class i and b-class j.
	cont := make([][]int64, ra)
	for i := range cont {
		cont[i] = make([]int64, rb)
	}
	for i := range aa {
		cont[aIdx[aa[i]]][bIdx[bb[i]]]++
	}

	// Row sums (a marginals) and column sums (b marginals).
	aSum := make([]int64, ra)
	bSum := make([]int64, rb)
	for i := range cont {
		for j, v := range cont[i] {
			aSum[i] += v
			bSum[j] += v
		}
	}

	c2 := func(k int64) float64 { return float64(k * (k - 1) / 2) }

	sumNij2 := 0.0
	for i := range cont {
		for _, v := range cont[i] {
			sumNij2 += c2(v)
		}
	}
	sumAi2 := 0.0
	for _, v := range aSum {
		sumAi2 += c2(v)
	}
	sumBj2 := 0.0
	for _, v := range bSum {
		sumBj2 += c2(v)
	}
	cN2 := c2(int64(n))

	expected := sumAi2 * sumBj2 / cN2
	numerator := sumNij2 - expected
	denominator := 0.5*(sumAi2+sumBj2) - expected

	if denominator == 0 {
		if numerator == 0 {
			return 1.0
		}
		return 0.0
	}
	return numerator / denominator
}

// labelIndex builds a map from label value → dense 0-based index.
func labelIndex(labels []int) map[int]int {
	idx := make(map[int]int, len(labels))
	next := 0
	for _, l := range labels {
		if _, ok := idx[l]; !ok {
			idx[l] = next
			next++
		}
	}
	return idx
}

// TestAdjustedRandIndex verifies the ARI helper with known values.
func TestAdjustedRandIndex(t *testing.T) {
	cases := []struct {
		name string
		a, b []int
		want float64 // approximate
	}{
		{
			name: "identical partitions",
			a:    []int{0, 0, 1, 1},
			b:    []int{0, 0, 1, 1},
			want: 1.0,
		},
		{
			name: "identical partitions with permuted labels",
			a:    []int{0, 0, 1, 1},
			b:    []int{1, 1, 0, 0},
			want: 1.0,
		},
		{
			name: "all in one cluster vs all noise",
			a:    []int{0, 0, 0, 0},
			b:    []int{-1, -1, -1, -1},
			want: 0.0, // no agreement on co-cluster
		},
		{
			name: "completely independent random",
			// a: two equal halves; b: alternating — ARI < 0
			// contingency: all cells = 1, sumNij2=0, sumAi2=sumBj2=2, C(4,2)=6
			// ARI = (0 - 4/6) / (2 - 4/6) = (-2/3) / (4/3) = -0.5
			a:    []int{0, 0, 1, 1},
			b:    []int{0, 1, 0, 1},
			want: -0.5,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := adjustedRandIndex(tc.a, tc.b)
			assert.InDeltaf(t, tc.want, got, 1e-9, fmt.Sprintf("ARI(%v, %v)", tc.a, tc.b))
		})
	}
}
