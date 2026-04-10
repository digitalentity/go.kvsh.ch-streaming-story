# Hungarian Algorithm

The `hungarian` package provides a robust and efficient implementation of the Hungarian algorithm (also known as the Kuhn-Munkres algorithm) for solving the linear assignment problem in Go.

## Overview

The assignment problem involves finding a minimum-cost matching in a weighted bipartite graph. Given an $m \times n$ cost matrix, the package finds an assignment of rows to columns such that the total cost is minimized.

- **Complexity**: $O(m \cdot n \cdot \min(m, n))$, which simplifies to $O(n^3)$ for square matrices.
- **Precision**: Handles `float64` cost matrices with high precision using the shortest-path augmentation method.
- **Flexibility**: Supports rectangular matrices.
    - If $m < n$ (more columns than rows), every row is assigned to a unique column.
    - If $m > n$ (more rows than columns), only $n$ rows are assigned, and the remaining $m - n$ rows are left unassigned (indicated by `-1`).

## Installation

```bash
go get go.kvsh.ch/streaming-story/internal/hungarian
```

## Usage

```go
package main

import (
	"fmt"
	"go.kvsh.ch/streaming-story/internal/hungarian"
)

func main() {
	// Example cost matrix (3x3)
	cost := [][]float64{
		{9, 2, 7},
		{3, 6, 3},
		{5, 8, 1},
	}

	assignment, err := hungarian.Solve(cost)
	if err != nil {
		panic(err)
	}

	// Output: [1 0 2] 
	// (Row 0 -> Col 1, Row 1 -> Col 0, Row 2 -> Col 2)
	fmt.Println(assignment)
}
```

## Error Handling

The `Solve` function returns an error if:
- The cost matrix is `nil` or empty.
- The cost matrix is jagged (rows have different lengths).

## Testing

Run tests and benchmarks:

```bash
go test -v ./internal/hungarian
go test -bench . ./internal/hungarian
```
