// Package hdbscan implements the HDBSCAN* clustering algorithm.
//
// Reference: Campello et al., "Density-Based Clustering Based on Hierarchical
// Density Estimates" (2013).
//
// Input:  [][]float32 embedding matrix (each row is one point)
// Output: []int label slice; −1 denotes noise.
//
// Distance metric: cosine distance (1 − cosine_similarity). Embeddings are
// expected to be normalised; the implementation normalises internally so
// callers need not pre-normalise.
package hdbscan

import (
	"errors"
	"math"
)

// Cluster runs HDBSCAN* on pts and returns a cluster label per point.
// Labels are 0-indexed integers; −1 means noise.
//
//   - minClusterSize: minimum number of points for a persistent cluster.
//   - minSamples:     number of neighbours used to compute the core distance
//     (≥ 1; setting it equal to minClusterSize is the standard choice).
func Cluster(pts [][]float32, minClusterSize, minSamples int) ([]int, error) {
	if len(pts) == 0 {
		return nil, errors.New("hdbscan: empty input")
	}
	if minClusterSize < 1 {
		return nil, errors.New("hdbscan: minClusterSize must be ≥ 1")
	}
	if minSamples < 1 {
		return nil, errors.New("hdbscan: minSamples must be ≥ 1")
	}
	dim := len(pts[0])
	for _, p := range pts[1:] {
		if len(p) != dim {
			return nil, errors.New("hdbscan: all points must have the same dimension")
		}
	}

	n := len(pts)

	// Fast path: fewer points than minClusterSize → all noise.
	if n < minClusterSize {
		labels := make([]int, n)
		for i := range labels {
			labels[i] = -1
		}
		return labels, nil
	}

	// Step 1: pairwise cosine distances.
	dist := pairwiseCosine(pts)

	// Step 2: core distances (k-th NN distance, k = minSamples).
	core := coreDistances(dist, minSamples)

	// Step 3: mutual reachability distances.
	mrd := mutualReachability(dist, core)

	// Step 4: minimum spanning tree via Prim's algorithm on the MRD graph.
	mst := primMST(mrd, n)

	// Step 5: build condensed cluster tree from the MST.
	tree := buildCondensedTree(mst, n, minClusterSize)

	// Step 6: extract flat clusters from the condensed tree by stability.
	return extractLabels(tree, n), nil
}

// --- Step 1: pairwise cosine distances ----------------------------------------

func pairwiseCosine(pts [][]float32) [][]float64 {
	n := len(pts)
	norms := make([]float64, n)
	for i, p := range pts {
		var s float64
		for _, v := range p {
			s += float64(v) * float64(v)
		}
		norms[i] = math.Sqrt(s)
	}

	dist := make([][]float64, n)
	for i := range dist {
		dist[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			var dot float64
			for k, v := range pts[i] {
				dot += float64(v) * float64(pts[j][k])
			}
			ni, nj := norms[i], norms[j]
			var d float64
			if ni == 0 || nj == 0 {
				if ni == nj { // both zero → identical zero vectors → distance 0
					d = 0
				} else {
					d = 1
				}
			} else {
				cos := dot / (ni * nj)
				if cos > 1 {
					cos = 1
				} else if cos < -1 {
					cos = -1
				}
				d = 1 - cos
			}
			dist[i][j] = d
			dist[j][i] = d
		}
	}
	return dist
}

// --- Step 2: core distances ----------------------------------------------------

func coreDistances(dist [][]float64, minSamples int) []float64 {
	n := len(dist)
	core := make([]float64, n)
	k := minSamples // we want the minSamples-th NN (1-indexed)
	if k >= n {
		k = n - 1
	}
	for i := 0; i < n; i++ {
		// Partial sort to find the k-th smallest distance from i (excluding self).
		neighbors := make([]float64, 0, n-1)
		for j := 0; j < n; j++ {
			if j != i {
				neighbors = append(neighbors, dist[i][j])
			}
		}
		core[i] = kthSmallest(neighbors, k-1) // 0-indexed k-th
	}
	return core
}

// kthSmallest returns the k-th smallest element (0-indexed) using
// a simple O(n) selection on a copy.
func kthSmallest(a []float64, k int) float64 {
	cp := make([]float64, len(a))
	copy(cp, a)
	return quickSelect(cp, 0, len(cp)-1, k)
}

func quickSelect(a []float64, lo, hi, k int) float64 {
	if lo == hi {
		return a[lo]
	}
	pivot := a[(lo+hi)/2]
	i, j := lo, hi
	for i <= j {
		for a[i] < pivot {
			i++
		}
		for a[j] > pivot {
			j--
		}
		if i <= j {
			a[i], a[j] = a[j], a[i]
			i++
			j--
		}
	}
	if k <= j {
		return quickSelect(a, lo, j, k)
	}
	if k >= i {
		return quickSelect(a, i, hi, k)
	}
	return pivot
}

// --- Step 3: mutual reachability -----------------------------------------------

func mutualReachability(dist [][]float64, core []float64) [][]float64 {
	n := len(dist)
	mrd := make([][]float64, n)
	for i := range mrd {
		mrd[i] = make([]float64, n)
		for j := range mrd[i] {
			if i == j {
				continue
			}
			v := dist[i][j]
			if core[i] > v {
				v = core[i]
			}
			if core[j] > v {
				v = core[j]
			}
			mrd[i][j] = v
		}
	}
	return mrd
}

// --- Step 4: Prim's MST --------------------------------------------------------

type edge struct{ u, v int; w float64 }

func primMST(mrd [][]float64, n int) []edge {
	inTree := make([]bool, n)
	minW := make([]float64, n)
	parent := make([]int, n)
	for i := range minW {
		minW[i] = math.MaxFloat64
		parent[i] = -1
	}
	minW[0] = 0

	edges := make([]edge, 0, n-1)
	for iter := 0; iter < n; iter++ {
		// Find the minimum-weight vertex not yet in the tree.
		u := -1
		for v := 0; v < n; v++ {
			if !inTree[v] && (u == -1 || minW[v] < minW[u]) {
				u = v
			}
		}
		inTree[u] = true
		if parent[u] >= 0 {
			edges = append(edges, edge{u: parent[u], v: u, w: minW[u]})
		}
		// Relax neighbours.
		for v := 0; v < n; v++ {
			if !inTree[v] && mrd[u][v] < minW[v] {
				minW[v] = mrd[u][v]
				parent[v] = u
			}
		}
	}
	return edges
}

// --- Step 5: condensed cluster tree -------------------------------------------

// clusterNode represents a node in the condensed tree.
// Each node is either a cluster or a leaf (single point).
type clusterNode struct {
	parent     int
	lambda     float64 // 1/distance at which this node split from its parent
	size       int     // number of original points below this node
	isLeaf     bool
	pointIndex int     // valid only for leaves
	children   []int  // internal node children (indices into nodes slice)
	stability  float64
}

// buildCondensedTree constructs the condensed cluster hierarchy from the MST.
// The MST edges are processed in decreasing weight order (single-linkage).
func buildCondensedTree(mst []edge, n, minClusterSize int) []clusterNode {
	// Sort MST edges by descending weight.
	sortEdgesDesc(mst)

	// Union-Find for merging components.
	uf := newUF(n)

	// We build the condensed tree bottom-up.
	// Start with each point as its own component (leaf cluster).
	// Node indices 0..n-1 are the original points.
	// New internal nodes start at index n.
	nodes := make([]clusterNode, n)
	for i := range nodes {
		nodes[i] = clusterNode{parent: -1, isLeaf: true, pointIndex: i, size: 1}
	}

	clusterOf := make([]int, n) // clusterOf[point] = current cluster node index
	for i := range clusterOf {
		clusterOf[i] = i
	}

	for _, e := range mst {
		ca := uf.find(e.u)
		cb := uf.find(e.v)
		if ca == cb {
			continue
		}
		lambda := 0.0
		if e.w > 0 {
			lambda = 1.0 / e.w
		}

		nodeA := clusterOf[ca]
		nodeB := clusterOf[cb]
		sizeA := componentSize(uf, ca, n)
		sizeB := componentSize(uf, cb, n)
		merged := uf.union(ca, cb)

		// Decide: create a new cluster node or just absorb.
		if sizeA >= minClusterSize && sizeB >= minClusterSize {
			// True split: create a new internal node.
			newNode := clusterNode{
				parent:   -1,
				lambda:   lambda,
				size:     sizeA + sizeB,
				children: []int{nodeA, nodeB},
			}
			nodes[nodeA].parent = len(nodes)
			nodes[nodeA].lambda = lambda
			nodes[nodeB].parent = len(nodes)
			nodes[nodeB].lambda = lambda
			nodes = append(nodes, newNode)
			clusterOf[merged] = len(nodes) - 1
		} else {
			// One side is too small — absorb into the larger cluster.
			surviveNode := nodeA
			if sizeB > sizeA {
				surviveNode = nodeB
			}
			nodes[surviveNode].size = sizeA + sizeB
			if sizeA < minClusterSize {
				markNoise(&nodes, nodeA, lambda)
			} else {
				markNoise(&nodes, nodeB, lambda)
			}
			clusterOf[merged] = surviveNode
		}
	}

	// Compute stability for each internal node.
	computeStability(nodes)
	return nodes
}

// markNoise marks all leaves under nodeIdx as having fallen out at lambda.
func markNoise(nodes *[]clusterNode, idx int, lambda float64) {
	nd := &(*nodes)[idx]
	if nd.isLeaf {
		nd.parent = -2 // noise marker
		nd.lambda = lambda
		return
	}
	for _, c := range nd.children {
		markNoise(nodes, c, lambda)
	}
}

func computeStability(nodes []clusterNode) {
	// Stability of cluster C = sum over points p in C of (λ_p - λ_birth(C))
	// where λ_p is the lambda at which p falls out of C and λ_birth is the
	// lambda when C split from its parent.
	for i := range nodes {
		if nodes[i].isLeaf || nodes[i].size == 0 {
			continue
		}
		lambdaBirth := nodes[i].lambda
		var stab float64
		collectLeafLambdas(nodes, i, lambdaBirth, &stab)
		nodes[i].stability = stab
	}
}

func collectLeafLambdas(nodes []clusterNode, idx int, lambdaBirth float64, acc *float64) {
	nd := nodes[idx]
	if nd.isLeaf {
		*acc += nd.lambda - lambdaBirth
		return
	}
	for _, c := range nd.children {
		collectLeafLambdas(nodes, c, lambdaBirth, acc)
	}
}

// --- Step 6: extract flat clusters --------------------------------------------

func extractLabels(nodes []clusterNode, n int) []int {
	if len(nodes) == 0 {
		labels := make([]int, n)
		for i := range labels {
			labels[i] = -1
		}
		return labels
	}

	// Find the root (node with parent == -1 that is not a leaf).
	root := -1
	for i := len(nodes) - 1; i >= 0; i-- {
		if !nodes[i].isLeaf && nodes[i].parent == -1 {
			root = i
			break
		}
	}
	if root == -1 {
		// No internal nodes formed — all noise.
		labels := make([]int, n)
		for i := range labels {
			labels[i] = -1
		}
		return labels
	}

	// Select clusters using the excess-of-mass (stability-based) approach.
	selected := make([]bool, len(nodes))
	selectClusters(nodes, root, selected)

	// Assign labels.
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}
	clusterID := 0
	for i, sel := range selected {
		if !sel {
			continue
		}
		assignClusterLabel(nodes, i, clusterID, labels)
		clusterID++
	}
	return labels
}

// selectClusters runs the excess-of-mass algorithm bottom-up.
// It marks the set of clusters that maximises total stability.
func selectClusters(nodes []clusterNode, idx int, selected []bool) float64 {
	nd := nodes[idx]
	if nd.isLeaf || len(nd.children) == 0 {
		return 0
	}
	childSum := 0.0
	for _, c := range nd.children {
		childSum += selectClusters(nodes, c, selected)
	}
	if nd.stability >= childSum {
		// Select this node and deselect its descendants.
		selected[idx] = true
		deselectDescendants(nodes, idx, selected)
		return nd.stability
	}
	return childSum
}

func deselectDescendants(nodes []clusterNode, idx int, selected []bool) {
	selected[idx] = false
	for _, c := range nodes[idx].children {
		deselectDescendants(nodes, c, selected)
	}
}

func assignClusterLabel(nodes []clusterNode, idx, label int, labels []int) {
	nd := nodes[idx]
	if nd.isLeaf {
		if nd.pointIndex >= 0 && nd.pointIndex < len(labels) {
			labels[nd.pointIndex] = label
		}
		return
	}
	for _, c := range nd.children {
		assignClusterLabel(nodes, c, label, labels)
	}
}

// --- Union-Find ---------------------------------------------------------------

type uf struct {
	parent []int
	rank   []int
	size   []int
}

func newUF(n int) *uf {
	u := &uf{parent: make([]int, n), rank: make([]int, n), size: make([]int, n)}
	for i := range u.parent {
		u.parent[i] = i
		u.size[i] = 1
	}
	return u
}

func (u *uf) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

// union merges the components of a and b and returns the representative.
func (u *uf) union(a, b int) int {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return ra
	}
	if u.rank[ra] < u.rank[rb] {
		ra, rb = rb, ra
	}
	u.parent[rb] = ra
	u.size[ra] += u.size[rb]
	if u.rank[ra] == u.rank[rb] {
		u.rank[ra]++
	}
	return ra
}

func (u *uf) componentSize(root int) int {
	return u.size[u.find(root)]
}

func componentSize(u *uf, root, _ int) int {
	return u.componentSize(root)
}

// --- Sorting helpers ----------------------------------------------------------

func sortEdgesDesc(edges []edge) {
	// Simple insertion sort is fine for the MST sizes we expect (≤ BatchSampleCap).
	// For larger inputs, replace with sort.Slice.
	for i := 1; i < len(edges); i++ {
		key := edges[i]
		j := i - 1
		for j >= 0 && edges[j].w < key.w {
			edges[j+1] = edges[j]
			j--
		}
		edges[j+1] = key
	}
}
