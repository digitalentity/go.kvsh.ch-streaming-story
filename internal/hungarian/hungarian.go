package hungarian

import (
	"errors"
	"math"
)

// Solve finds the minimum-cost assignment for the given cost matrix.
// It returns an assignment slice of length m (number of rows) where
// assignment[row] = col, or −1 if the row could not be assigned (only
// possible when m > n).
//
// The cost matrix must be non-nil and non-empty. All rows must have the
// same length.
func Solve(cost [][]float64) ([]int, error) {
	if len(cost) == 0 {
		return nil, errors.New("hungarian: cost matrix must not be empty")
	}
	m := len(cost)
	n := len(cost[0])
	if n == 0 {
		return nil, errors.New("hungarian: cost matrix columns must not be empty")
	}
	for i := 1; i < m; i++ {
		if len(cost[i]) != n {
			return nil, errors.New("hungarian: cost matrix rows must have the same length")
		}
	}

	// The classic algorithm requires m ≤ n. When m > n, we transpose,
	// solve, then invert the assignment.
	if m > n {
		return solveWide(cost, m, n)
	}
	return solveTall(cost, m, n)
}

// solveTall handles the case m ≤ n via the O(m^2 * n) Hungarian algorithm.
// Returns an assignment vector of length m.
func solveTall(cost [][]float64, m, n int) ([]int, error) {
	u := make([]float64, m+1)
	v := make([]float64, n+1)
	p := make([]int, n+1)
	way := make([]int, n+1)

	for i := 1; i <= m; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		for j := 1; j <= n; j++ {
			minv[j] = math.Inf(1)
		}
		used := make([]bool, n+1)

		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0

			for j := 1; j <= n; j++ {
				if !used[j] {
					cur := cost[i0-1][j-1] - u[i0] - v[j]
					if cur < minv[j] {
						minv[j] = cur
						way[j] = j0
					}
					if minv[j] < delta {
						delta = minv[j]
						j1 = j
					}
				}
			}

			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}

		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	assignment := make([]int, m)
	for i := 0; i < m; i++ {
		assignment[i] = -1
	}
	for j := 1; j <= n; j++ {
		if p[j] > 0 {
			assignment[p[j]-1] = j - 1
		}
	}
	return assignment, nil
}

// solveWide handles m > n by transposing, solving, then inverting.
func solveWide(cost [][]float64, m, n int) ([]int, error) {
	transposed := make([][]float64, n)
	for j := range transposed {
		transposed[j] = make([]float64, m)
		for i := 0; i < m; i++ {
			transposed[j][i] = cost[i][j]
		}
	}
	colAssign, err := solveTall(transposed, n, m)
	if err != nil {
		return nil, err
	}
	// colAssign[col] = row assigned to that col.
	// Invert: rowAssign[row] = col.
	rowAssign := make([]int, m)
	for i := range rowAssign {
		rowAssign[i] = -1
	}
	for col, row := range colAssign {
		if row >= 0 {
			rowAssign[row] = col
		}
	}
	return rowAssign, nil
}
