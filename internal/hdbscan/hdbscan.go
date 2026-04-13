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
	mst := primMST(dist, core, n)

	// Sort MST edges in ascending order of MRD weight: closest first.
	sort.Slice(mst, func(i, j int) bool { return mst[i].w < mst[j].w })

	dn := buildDendrogram(mst, n)
	clusters, pointFallout := condense(dn, n, minClusterSize)

	if len(clusters) == 0 {
		return labels, nil
	}

	computeStability(clusters, pointFallout)
	selected := selectClusters(clusters)

	anySelected := false
	for _, s := range selected {
		if s {
			anySelected = true
			break
		}
	}
	if !anySelected {
		return labels, nil
	}

	labelID := 0
	for i := range clusters {
		if selected[i] {
			labelSubtree(clusters, i, labelID, labels, pointFallout)
			labelID++
		}
	}
	return labels, nil
}

func labelSubtree(clusters []cCluster, idx int, labelID int, labels []int, pointFallout []fallout) {
	for _, f := range pointFallout {
		if f.clusterIdx == idx {
			labels[f.pointIdx] = labelID
		}
	}
	for _, childIdx := range clusters[idx].children {
		labelSubtree(clusters, childIdx, labelID, labels, pointFallout)
	}
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
		core[i] = kthSmallest(row, minSamples-1)
	}
	return core
}

func kthSmallest(a []float64, k int) float64 {
	if k < 0 {
		return 0
	}
	if k >= len(a) {
		k = len(a) - 1
	}
	cp := make([]float64, len(a))
	copy(cp, a)
	sort.Float64s(cp)
	return cp[k]
}

// ── Step 3: Prim's MST ───────────────────────────────────────────────────────

type edge struct {
	u, v int
	w    float64
}

func primMST(dist [][]float64, core []float64, n int) []edge {
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
			if inTree[v] {
				continue
			}
			// mrd(u, v) = max(dist[u][v], core[u], core[v])
			mrdUV := dist[u][v]
			if core[u] > mrdUV {
				mrdUV = core[u]
			}
			if core[v] > mrdUV {
				mrdUV = core[v]
			}
			if mrdUV < minW[v] {
				minW[v] = mrdUV
				parent[v] = u
			}
		}
	}
	return edges
}

// ── Step 4: single-linkage dendrogram ───────────────────────────────────────

type dendro struct {
	left, right int
	lambda      float64
	size        int
}

func buildDendrogram(mst []edge, n int) []dendro {
	nodes := make([]dendro, 2*n-1)
	for i := 0; i < n; i++ {
		nodes[i] = dendro{left: -1, right: -1, size: 1}
	}
	uf := newUF(n)
	repNode := make([]int, n)
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

// ── Step 5: condensed cluster tree ──────────────────────────────────────────

type cCluster struct {
	lambdaBirth float64
	lambdaDeath float64
	size        int
	children    []int
	stability   float64
}

type fallout struct {
	pointIdx   int
	clusterIdx int
	lambda     float64
}

func condense(nodes []dendro, n, minClusterSize int) ([]cCluster, []fallout) {
	if len(nodes) == 0 {
		return nil, nil
	}
	root := len(nodes) - 1
	if nodes[root].size < minClusterSize {
		return nil, nil
	}

	clusters := []cCluster{{
		lambdaBirth: 0,
		lambdaDeath: 0, // Will be updated if it splits or dies
		size:        nodes[root].size,
	}}
	var pointFallout []fallout

	walkDendro(nodes, root, 0, &clusters, &pointFallout, minClusterSize)
	return clusters, pointFallout
}

func walkDendro(nodes []dendro, idx, clusterIdx int, clusters *[]cCluster, pointFallout *[]fallout, mcs int) {
	nd := nodes[idx]
	if nd.left == -1 {
		// Point reached leaf within this cluster.
		// Use lambdaDeath of the cluster as fallout lambda if it dies here.
		// Actually, if a cluster reaches a leaf, it effectively "ends" at Inf 
		// if it was a true leaf, BUT in HDBSCAN* all points must eventually 
		// fall out or split. If it's a leaf node in the dendrogram, it's a point.
		// If it stayed until the end of this cluster, we'll use Inf, 
		// and computeStability will cap it at lambdaDeath.
		*pointFallout = append(*pointFallout, fallout{pointIdx: idx, clusterIdx: clusterIdx, lambda: math.Inf(1)})
		return
	}

	left, right := nodes[nd.left], nodes[nd.right]
	leftBig := left.size >= mcs
	rightBig := right.size >= mcs

	if leftBig && rightBig {
		// True split.
		(*clusters)[clusterIdx].lambdaDeath = nd.lambda
		
		li := len(*clusters)
		*clusters = append(*clusters, cCluster{lambdaBirth: nd.lambda, lambdaDeath: math.Inf(1), size: left.size})
		ri := len(*clusters)
		*clusters = append(*clusters, cCluster{lambdaBirth: nd.lambda, lambdaDeath: math.Inf(1), size: right.size})
		
		(*clusters)[clusterIdx].children = append((*clusters)[clusterIdx].children, li, ri)
		
		walkDendro(nodes, nd.left, li, clusters, pointFallout, mcs)
		walkDendro(nodes, nd.right, ri, clusters, pointFallout, mcs)
	} else if leftBig {
		// Right is noise.
		collectFallout(nodes, nd.right, nd.lambda, clusterIdx, pointFallout)
		walkDendro(nodes, nd.left, clusterIdx, clusters, pointFallout, mcs)
	} else if rightBig {
		// Left is noise.
		collectFallout(nodes, nd.left, nd.lambda, clusterIdx, pointFallout)
		walkDendro(nodes, nd.right, clusterIdx, clusters, pointFallout, mcs)
	} else {
		// Both are noise, cluster dies.
		(*clusters)[clusterIdx].lambdaDeath = nd.lambda
		collectFallout(nodes, nd.left, nd.lambda, clusterIdx, pointFallout)
		collectFallout(nodes, nd.right, nd.lambda, clusterIdx, pointFallout)
	}
}

func collectFallout(nodes []dendro, idx int, lambda float64, clusterIdx int, pointFallout *[]fallout) {
	nd := nodes[idx]
	if nd.left == -1 {
		*pointFallout = append(*pointFallout, fallout{pointIdx: idx, clusterIdx: clusterIdx, lambda: lambda})
		return
	}
	collectFallout(nodes, nd.left, lambda, clusterIdx, pointFallout)
	collectFallout(nodes, nd.right, lambda, clusterIdx, pointFallout)
}

// ── Step 6: cluster stability ────────────────────────────────────────────────

func computeStability(clusters []cCluster, pointFallout []fallout) {
	for i := range clusters {
		birth := clusters[i].lambdaBirth
		s := 0.0
		for _, f := range pointFallout {
			if f.clusterIdx == i {
				lamP := f.lambda
				if math.IsInf(lamP, 1) {
					// This point stayed until the cluster split or died.
					// If it died, it fell out at lambdaDeath.
					// If it split, it also effectively "ended" for THIS cluster node at lambdaDeath.
					lamP = clusters[i].lambdaDeath
				}
				d := lamP - birth
				if d > 0 {
					s += d
				}
			}
		}
		// Also add contribution from points that stayed in the cluster until it split into children.
		// Reference formula often expresses this by saying points in children are also in parent.
		// S(C) = sum_{p in C_direct_fallout} (lambda_p - lambda_birth) + sum_{child in children} child.size * (child.lambda_birth - lambda_birth)
		for _, childIdx := range clusters[i].children {
			child := clusters[childIdx]
			s += float64(child.size) * (child.lambdaBirth - birth)
		}
		clusters[i].stability = s
	}
}

// ── Step 7: excess-of-mass cluster selection ─────────────────────────────────

func selectClusters(clusters []cCluster) []bool {
	n := len(clusters)
	propStability := make([]float64, n)
	for i := range propStability {
		propStability[i] = clusters[i].stability
	}

	// Bottom-up pass to propagate stability.
	for i := n - 1; i >= 0; i-- {
		if len(clusters[i].children) > 0 {
			childSum := 0.0
			for _, childIdx := range clusters[i].children {
				childSum += propStability[childIdx]
			}
			if childSum > propStability[i] {
				propStability[i] = childSum
			}
		}
	}

	selected := make([]bool, n)
	// Top-down pass to select.
	var bfs []int
	if len(clusters[0].children) > 0 {
		bfs = append(bfs, clusters[0].children...)
	} else {
		// Root has no children, so it's the only potential cluster.
		// Select it if its stability is finite and > 0.
		if !math.IsInf(clusters[0].stability, 1) && clusters[0].stability > 0 {
			selected[0] = true
		}
		return selected
	}

	for len(bfs) > 0 {
		curr := bfs[0]
		bfs = bfs[1:]

		if len(clusters[curr].children) == 0 {
			if !math.IsInf(clusters[curr].stability, 1) {
				selected[curr] = true
			}
			continue
		}

		childSum := 0.0
		for _, childIdx := range clusters[curr].children {
			childSum += propStability[childIdx]
		}

		if clusters[curr].stability >= childSum && !math.IsInf(clusters[curr].stability, 1) {
			selected[curr] = true
		} else {
			for _, childIdx := range clusters[curr].children {
				bfs = append(bfs, childIdx)
			}
		}
	}
	return selected
}

// ── Step 8: Union-Find ────────────────────────────────────────────────────────

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
