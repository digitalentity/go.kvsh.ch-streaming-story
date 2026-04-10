package hdbscan_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kvsh.ch/streaming-story/internal/hdbscan"
)

// labelCounts returns a map of label → count, excluding noise (−1).
func labelCounts(labels []int) map[int]int {
	m := make(map[int]int)
	for _, l := range labels {
		if l >= 0 {
			m[l]++
		}
	}
	return m
}

func noiseCount(labels []int) int {
	n := 0
	for _, l := range labels {
		if l == -1 {
			n++
		}
	}
	return n
}

// makeDenseBlob returns n identical points (maximum density cluster).
func makeDenseBlob(n int, val float32) [][]float32 {
	pts := make([][]float32, n)
	for i := range pts {
		pts[i] = []float32{val, 0}
	}
	return pts
}

// --- parameter validation ---

func TestCluster_nil_input_returns_error(t *testing.T) {
	_, err := hdbscan.Cluster(nil, 2, 2)
	require.Error(t, err)
}

func TestCluster_empty_input_returns_error(t *testing.T) {
	_, err := hdbscan.Cluster([][]float32{}, 2, 2)
	require.Error(t, err)
}

func TestCluster_zero_minClusterSize_returns_error(t *testing.T) {
	pts := makeDenseBlob(5, 1.0)
	_, err := hdbscan.Cluster(pts, 0, 2)
	require.Error(t, err)
}

func TestCluster_zero_minSamples_returns_error(t *testing.T) {
	pts := makeDenseBlob(5, 1.0)
	_, err := hdbscan.Cluster(pts, 2, 0)
	require.Error(t, err)
}

func TestCluster_inconsistent_dimensions_returns_error(t *testing.T) {
	pts := [][]float32{{1, 2}, {3, 4, 5}}
	_, err := hdbscan.Cluster(pts, 2, 2)
	require.Error(t, err)
}

// --- result shape ---

func TestCluster_returns_one_label_per_point(t *testing.T) {
	pts := makeDenseBlob(10, 1.0)
	labels, err := hdbscan.Cluster(pts, 2, 2)
	require.NoError(t, err)
	assert.Len(t, labels, len(pts))
}

// --- core cluster detection ---

func TestCluster_two_tight_blobs_produce_two_clusters(t *testing.T) {
	// Blob A near 0, Blob B near 100 — well separated in 1D.
	pts := make([][]float32, 0, 10)
	for i := 0; i < 5; i++ {
		pts = append(pts, []float32{float32(i) * 0.01})
	}
	for i := 0; i < 5; i++ {
		pts = append(pts, []float32{100 + float32(i)*0.01})
	}
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	counts := labelCounts(labels)
	assert.Len(t, counts, 2, "expected exactly 2 clusters, got %v", counts)
	for _, c := range counts {
		assert.GreaterOrEqual(t, c, 3)
	}
}

func TestCluster_three_tight_blobs_produce_three_clusters(t *testing.T) {
	pts := make([][]float32, 0, 15)
	for _, center := range []float32{0, 100, 200} {
		for i := 0; i < 5; i++ {
			pts = append(pts, []float32{center + float32(i)*0.01})
		}
	}
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	counts := labelCounts(labels)
	assert.Len(t, counts, 3, "expected exactly 3 clusters, got %v", counts)
}

// --- noise handling ---

func TestCluster_isolated_point_is_noise(t *testing.T) {
	// 5 tight points + 1 isolated outlier far away.
	pts := [][]float32{
		{0}, {0.01}, {0.02}, {0.03}, {0.04}, // dense cluster
		{1000}, // isolated outlier
	}
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	assert.Equal(t, -1, labels[5], "isolated point should be labeled noise")
}

func TestCluster_all_noise_when_density_too_low(t *testing.T) {
	// Each point is far from every other point; no cluster can form with MinClusterSize=3.
	pts := [][]float32{{0}, {100}, {200}, {300}}
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	assert.Equal(t, len(pts), noiseCount(labels),
		"all points should be noise when no cluster meets MinClusterSize")
}

// --- edge cases ---

func TestCluster_single_point_is_noise(t *testing.T) {
	labels, err := hdbscan.Cluster([][]float32{{1, 2, 3}}, 2, 2)
	require.NoError(t, err)
	require.Len(t, labels, 1)
	assert.Equal(t, -1, labels[0])
}

func TestCluster_fewer_points_than_minClusterSize(t *testing.T) {
	pts := [][]float32{{0}, {1}}
	labels, err := hdbscan.Cluster(pts, 5, 5)
	require.NoError(t, err)
	assert.Equal(t, 2, noiseCount(labels))
}

func TestCluster_all_identical_points_form_one_cluster(t *testing.T) {
	pts := makeDenseBlob(8, 0.5)
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	assert.Zero(t, noiseCount(labels), "identical points must not be noise")
	counts := labelCounts(labels)
	assert.Len(t, counts, 1, "identical points must form exactly one cluster")
}

func TestCluster_labels_are_zero_indexed_and_contiguous(t *testing.T) {
	// Two blobs → labels should be 0 and 1 with no gaps.
	pts := make([][]float32, 0)
	for i := 0; i < 5; i++ {
		pts = append(pts, []float32{float32(i) * 0.01})
	}
	for i := 0; i < 5; i++ {
		pts = append(pts, []float32{100 + float32(i)*0.01})
	}
	labels, err := hdbscan.Cluster(pts, 3, 3)
	require.NoError(t, err)
	counts := labelCounts(labels)
	require.Len(t, counts, 2)
	_, has0 := counts[0]
	_, has1 := counts[1]
	assert.True(t, has0 && has1, "cluster labels must be 0-indexed: got %v", counts)
}
