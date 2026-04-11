# HDBSCAN* (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

The `hdbscan` package provides a robust and efficient implementation of the HDBSCAN* clustering algorithm in Go, optimized for use with high-dimensional embeddings.

## Overview

HDBSCAN* (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a powerful clustering algorithm that extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using a technique to extract a flat clustering based on the stability of clusters.

### Key Features

- **Density-Based**: Finds clusters of arbitrary shapes and varying densities.
- **Hierarchical**: Builds a dendrogram of all possible clusters and selects the most stable ones.
- **Noise Awareness**: Explicitly identifies outliers (labeled as `-1`).
- **Cosine Distance**: Uses $1 - \text{cosine\_similarity}$ as the primary metric, making it ideal for semantic embeddings.
- **No Pre-normalization Required**: Input vectors are normalized internally.
- **Optimized Performance**: 
    - **Complexity**: $O(n^2)$ time and $O(n^2)$ space (for the distance matrix).
    - **MST Optimization**: Computes mutual reachability distances on the fly to reduce memory pressure.
    - **Stability**: Handles identical points and zero-distance clusters correctly.

## Installation

```bash
go get go.kvsh.ch/streaming-story/internal/hdbscan
```

## Usage

```go
package main

import (
	"fmt"
	"go.kvsh.ch/streaming-story/internal/hdbscan"
)

func main() {
	// Example 2D points (embeddings)
	pts := [][]float32{
		{1.0, 0.0}, {1.0, 0.1}, {0.9, 0.1}, // Cluster A
		{0.0, 1.0}, {0.1, 1.0}, {0.1, 0.9}, // Cluster B
		{-1.0, -1.0},                       // Outlier/Noise
	}

	// Parameters:
	// - minClusterSize: Minimum points to form a persistent cluster (default: 3)
	// - minSamples: Core point density parameter (default: same as minClusterSize)
	labels, err := hdbscan.Cluster(pts, 3, 3)
	if err != nil {
		panic(err)
	}

	// Output labels for each point (-1 denotes noise)
	// Example: [0 0 0 1 1 1 -1]
	fmt.Println(labels)
}
```

## Algorithm Steps

1. **Pairwise Cosine Distances**: Compute the distance matrix using cosine distance.
2. **Core Distances**: Calculate the distance to the $k$-th nearest neighbor for each point.
3. **Prim's MST**: Build a Minimum Spanning Tree of the Mutual Reachability Graph.
4. **Single-Linkage Dendrogram**: Construct the hierarchy of potential clusters.
5. **Condensed Cluster Tree**: Simplify the hierarchy by removing small or unstable splits.
6. **Stability Calculation**: Measure the "persistence" of each cluster across density levels.
7. **Excess-of-Mass Selection**: Select the optimal set of non-overlapping clusters that maximize total stability.

## Testing

Run tests to verify the implementation:

```bash
go test -v ./internal/hdbscan
```

## References

- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). "Density-Based Clustering Based on Hierarchical Density Estimates".
- McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering".
