// Package hdbscan implements the HDBSCAN* clustering algorithm.
//
// Reference: Campello et al., "Density-Based Clustering Based on Hierarchical
// Density Estimates" (2013).
//
// Input:  n×d float32 matrix (one embedding per row)
// Output: []int label slice; −1 denotes noise.
//
// Distance metric: cosine distance (1 − cosine_similarity). Points are
// normalised internally; callers need not pre-normalise.
package hdbscan

import (
	"errors"
	"math"
	"sort"
)

// Cluster runs HDBSCAN* on pts and returns a cluster label per point.
// Labels are 0-indexed integers; −1 means noise.
//
//   - minClusterSize: minimum number of points in a persistent cluster.
//   - minSamples:     neighbourhood size used to compute the core distance.
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
	labels := make([]int, n)
	for i := range labels {
		labels[i] = -1
	}

	if n < minClusterSize {
		return labels, nil
	}

	dist := pairwiseCosine(pts)
	core := coreDistances(dist, n, minSamples)
	mrd := mutualReachability(dist, core, n)
	mst := primMST(mrd, n)

	// Sort MST edges in ascending order of MRD weight: closest first.
	sort.Slice(mst, func(i, j int) bool { return mst[i].w < mst[j].w })

	dn := buildDendrogram(mst, n)
	clusters := condense(dn, n, minClusterSize)

	if len(clusters) == 0 {
		return labels, nil
	}

	computeStability(clusters)
	selected := make([]bool, len(clusters))
	selectClusters(clusters, 0, selected)

	labelID := 0
	for ci, sel := range selected {
		if !sel {
			continue
		}
		for p := range collectPoints(clusters, ci) {
			labels[p] = labelID
		}
		labelID++
	}
	return labels, nil
}

// ── Step 1: pairwise cosine distances ────────────────────────────────────────

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
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			ni, nj := norms[i], norms[j]
			if ni == 0 && nj == 0 {
				dist[i][j] = 0
				continue
			}
			if ni == 0 || nj == 0 {
				dist[i][j] = 1
				continue
			}
			var dot float64
			for k := range pts[i] {
				dot += float64(pts[i][k]) * float64(pts[j][k])
			}
			cos := dot / (ni * nj)
			if cos > 1 {
				cos = 1
			} else if cos < -1 {
				cos = -1
			}
			dist[i][j] = 1 - cos
		}
	}
	return dist
}

// ── Step 2: core distances ───────────────────────────────────────────────────

func coreDistances(dist [][]float64, n, minSamples int) []float64 {
	k := minSamples
	if k >= n {
		k = n - 1
	}
	core := make([]float64, n)
	row := make([]float64, n-1)
	for i := 0; i < n; i++ {
		idx := 0
		for j := 0; j < n; j++ {
			if j != i {
				row[idx] = dist[i][j]
				idx++
			}
		}
		core[i] = kthSmallest(row, k-1)
	}
	return core
}

func kthSmallest(a []float64, k int) float64 {
	cp := make([]float64, len(a))
	copy(cp, a)
	return qselect(cp, 0, len(cp)-1, k)
}

func qselect(a []float64, lo, hi, k int) float64 {
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
	switch {
	case k <= j:
		return qselect(a, lo, j, k)
	case k >= i:
		return qselect(a, i, hi, k)
	default:
		return pivot
	}
}

// ── Step 3: mutual reachability ──────────────────────────────────────────────

func mutualReachability(dist [][]float64, core []float64, n int) [][]float64 {
	mrd := make([][]float64, n)
	for i := range mrd {
		mrd[i] = make([]float64, n)
		for j := 0; j < n; j++ {
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

// ── Step 4: Prim's MST ───────────────────────────────────────────────────────

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
	for range n {
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
		for v := 0; v < n; v++ {
			if !inTree[v] && mrd[u][v] < minW[v] {
				minW[v] = mrd[u][v]
				parent[v] = u
			}
		}
	}
	return edges
}

// ── Step 5: single-linkage dendrogram ───────────────────────────────────────
//
// Nodes 0..n-1 are leaves (original points).
// Nodes n..2n-2 are internal (one per MST edge, processed in ascending order).

type dendro struct {
	left, right int
	lambda      float64 // 1/w at merge time; +Inf when w==0
	size        int
}

func buildDendrogram(mst []edge, n int) []dendro {
	nodes := make([]dendro, 2*n-1)
	for i := 0; i < n; i++ {
		nodes[i] = dendro{left: -1, right: -1, size: 1}
	}
	uf := newUF(n)
	repNode := make([]int, n) // component representative → dendrogram node index
	for i := range repNode {
		repNode[i] = i
	}
	next := n
	for _, e := range mst {
		ra, rb := uf.find(e.u), uf.find(e.v)
		la, lb := repNode[ra], repNode[rb]
		lam := math.Inf(1)
		if e.w > 0 {
			lam = 1.0 / e.w
		}
		nodes[next] = dendro{
			left:   la,
			right:  lb,
			lambda: lam,
			size:   nodes[la].size + nodes[lb].size,
		}
		merged := uf.union(ra, rb)
		repNode[merged] = next
		next++
	}
	return nodes[:next]
}

// ── Step 6: condensed cluster tree ──────────────────────────────────────────

type cCluster struct {
	lambdaBirth float64
	parent      int     // −1 = no parent
	children    []int
	// falls[pointIdx] = lambda at which that point left this cluster.
	// Every original point appears in exactly one cluster's falls map.
	falls     map[int]float64
	stability float64
}

// condense walks the dendrogram and returns the condensed cluster tree.
// Index 0 is always the root cluster (if any clusters form at all).
func condense(nodes []dendro, n, minClusterSize int) []cCluster {
	if len(nodes) == 0 {
		return nil
	}
	root := len(nodes) - 1 // last internal node = dendrogram root
	if nodes[root].size < minClusterSize {
		return nil
	}

	clusters := []cCluster{{lambdaBirth: 0, parent: -1, falls: make(map[int]float64)}}
	walkDendro(nodes, root, 0, &clusters, minClusterSize)
	return clusters
}

// walkDendro processes dendrogram node idx within cluster clusterIdx.
func walkDendro(nodes []dendro, idx, clusterIdx int, clusters *[]cCluster, minClusterSize int) {
	nd := nodes[idx]

	// Leaf: original point. It falls off the current cluster at the entry lambda.
	if nd.left == -1 {
		(*clusters)[clusterIdx].falls[idx] = (*clusters)[clusterIdx].lambdaBirth
		return
	}

	lambda := nd.lambda
	leftSize := nodes[nd.left].size
	rightSize := nodes[nd.right].size
	leftBig := leftSize >= minClusterSize
	rightBig := rightSize >= minClusterSize

	// When w == 0 the lambda is +Inf: the two sides are effectively
	// inseparable, so we do not create a new split — just recurse into
	// both children within the same cluster.
	if math.IsInf(lambda, 1) {
		walkDendro(nodes, nd.left, clusterIdx, clusters, minClusterSize)
		walkDendro(nodes, nd.right, clusterIdx, clusters, minClusterSize)
		return
	}

	switch {
	case leftBig && rightBig:
		// True split: the current cluster forks into two child clusters.
		li := len(*clusters)
		*clusters = append(*clusters, cCluster{lambdaBirth: lambda, parent: clusterIdx, falls: make(map[int]float64)})
		ri := len(*clusters)
		*clusters = append(*clusters, cCluster{lambdaBirth: lambda, parent: clusterIdx, falls: make(map[int]float64)})
		(*clusters)[clusterIdx].children = append((*clusters)[clusterIdx].children, li, ri)
		walkDendro(nodes, nd.left, li, clusters, minClusterSize)
		walkDendro(nodes, nd.right, ri, clusters, minClusterSize)

	case leftBig:
		// Right side is too small: those points fall off current cluster.
		collectFalloff(nodes, nd.right, lambda, clusterIdx, clusters)
		walkDendro(nodes, nd.left, clusterIdx, clusters, minClusterSize)

	case rightBig:
		collectFalloff(nodes, nd.left, lambda, clusterIdx, clusters)
		walkDendro(nodes, nd.right, clusterIdx, clusters, minClusterSize)

	default:
		// Neither side is big enough: current cluster dies here.
		// All remaining points fall off at this lambda.
		collectFalloff(nodes, nd.left, lambda, clusterIdx, clusters)
		collectFalloff(nodes, nd.right, lambda, clusterIdx, clusters)
	}
}

// collectFalloff marks all original points under node idx as having left
// cluster clusterIdx at lambda.
func collectFalloff(nodes []dendro, idx int, lambda float64, clusterIdx int, clusters *[]cCluster) {
	nd := nodes[idx]
	if nd.left == -1 {
		(*clusters)[clusterIdx].falls[idx] = lambda
		return
	}
	collectFalloff(nodes, nd.left, lambda, clusterIdx, clusters)
	collectFalloff(nodes, nd.right, lambda, clusterIdx, clusters)
}

// ── Step 7: cluster stability ────────────────────────────────────────────────

func computeStability(clusters []cCluster) {
	for i := range clusters {
		var s float64
		birth := clusters[i].lambdaBirth
		for _, fall := range clusters[i].falls {
			d := fall - birth
			if d > 0 {
				s += d
			}
		}
		clusters[i].stability = s
	}
}

// ── Step 8: excess-of-mass cluster selection ─────────────────────────────────

// selectClusters returns the total propagated stability for the subtree
// rooted at idx, and marks the selected set.
func selectClusters(clusters []cCluster, idx int, selected []bool) float64 {
	if len(clusters[idx].children) == 0 {
		selected[idx] = true
		return clusters[idx].stability
	}
	childSum := 0.0
	for _, c := range clusters[idx].children {
		childSum += selectClusters(clusters, c, selected)
	}
	if clusters[idx].stability >= childSum {
		selected[idx] = true
		deselectAll(clusters, idx, selected)
		return clusters[idx].stability
	}
	return childSum
}

func deselectAll(clusters []cCluster, idx int, selected []bool) {
	selected[idx] = false
	for _, c := range clusters[idx].children {
		deselectAll(clusters, c, selected)
	}
}

// ── Step 9: label extraction ─────────────────────────────────────────────────

// collectPoints returns all original point indices under the subtree of cluster ci.
func collectPoints(clusters []cCluster, ci int) map[int]struct{} {
	pts := make(map[int]struct{}, len(clusters[ci].falls))
	for p := range clusters[ci].falls {
		pts[p] = struct{}{}
	}
	for _, c := range clusters[ci].children {
		for p := range collectPoints(clusters, c) {
			pts[p] = struct{}{}
		}
	}
	return pts
}

// ── Union-Find ────────────────────────────────────────────────────────────────

type uf struct {
	parent, rank, size []int
}

func newUF(n int) *uf {
	u := &uf{
		parent: make([]int, n),
		rank:   make([]int, n),
		size:   make([]int, n),
	}
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
