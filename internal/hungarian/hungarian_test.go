package hungarian_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.kvsh.ch/streaming-story/internal/hungarian"
)

// totalCost returns the total cost of an assignment given a cost matrix.
func totalCost(cost [][]float64, assignment []int) float64 {
	var total float64
	for row, col := range assignment {
		if col >= 0 {
			total += cost[row][col]
		}
	}
	return total
}

func TestSolve_1x1(t *testing.T) {
	cost := [][]float64{{7.0}}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.Equal(t, []int{0}, got)
}

func TestSolve_2x2_identity(t *testing.T) {
	// Diagonal has zero cost — assignment must be 0→0, 1→1.
	cost := [][]float64{
		{0, 1},
		{1, 0},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.Equal(t, []int{0, 1}, got)
}

func TestSolve_2x2_cross(t *testing.T) {
	// Off-diagonal has zero cost — assignment must be 0→1, 1→0.
	cost := [][]float64{
		{1, 0},
		{0, 1},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.Equal(t, []int{1, 0}, got)
}

func TestSolve_3x3_known_optimum(t *testing.T) {
	// Optimal assignment is 0→1, 1→0, 2→2 with total cost 2+3+1=6.
	cost := [][]float64{
		{9, 2, 7},
		{3, 6, 3},
		{5, 8, 1},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.InDelta(t, 6.0, totalCost(cost, got), 1e-9)
	// Verify it is a valid permutation.
	seen := make(map[int]bool)
	for _, col := range got {
		require.False(t, seen[col], "column %d assigned twice", col)
		seen[col] = true
	}
}

func TestSolve_all_zeros(t *testing.T) {
	n := 4
	cost := make([][]float64, n)
	for i := range cost {
		cost[i] = make([]float64, n)
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.Len(t, got, n)
	// Any permutation is optimal; just verify it is a valid permutation.
	seen := make(map[int]bool)
	for _, col := range got {
		require.False(t, seen[col], "column %d assigned twice", col)
		seen[col] = true
	}
}

func TestSolve_all_equal(t *testing.T) {
	cost := [][]float64{
		{5, 5, 5},
		{5, 5, 5},
		{5, 5, 5},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.InDelta(t, 15.0, totalCost(cost, got), 1e-9)
}

func TestSolve_more_rows_than_cols(t *testing.T) {
	// 3 rows, 2 cols — two rows get assigned, one gets -1.
	cost := [][]float64{
		{3, 1},
		{4, 2},
		{2, 9},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	require.Len(t, got, 3)

	unassigned := 0
	usedCols := make(map[int]bool)
	for _, col := range got {
		if col == -1 {
			unassigned++
			continue
		}
		require.False(t, usedCols[col], "column %d assigned twice", col)
		usedCols[col] = true
	}
	assert.Equal(t, 1, unassigned)
}

func TestSolve_more_cols_than_rows(t *testing.T) {
	// 2 rows, 3 cols — each row gets one column, one column unused.
	cost := [][]float64{
		{1, 4, 5},
		{9, 2, 6},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	require.Len(t, got, 2)

	seen := make(map[int]bool)
	for _, col := range got {
		require.GreaterOrEqual(t, col, 0)
		require.False(t, seen[col], "column %d assigned twice", col)
		seen[col] = true
	}
	assert.InDelta(t, 3.0, totalCost(cost, got), 1e-9) // 0→0 (1) + 1→1 (2)
}

func TestSolve_nil_matrix_returns_error(t *testing.T) {
	_, err := hungarian.Solve(nil)
	require.Error(t, err)
}

func TestSolve_empty_matrix_returns_error(t *testing.T) {
	_, err := hungarian.Solve([][]float64{})
	require.Error(t, err)
}

func TestSolve_assigns_each_row_exactly_once(t *testing.T) {
	cost := [][]float64{
		{4, 1, 3},
		{2, 0, 5},
		{3, 2, 2},
	}
	got, err := hungarian.Solve(cost)
	require.NoError(t, err)
	require.Len(t, got, 3)

	seen := make(map[int]bool)
	for _, col := range got {
		require.False(t, seen[col], "column %d assigned twice", col)
		seen[col] = true
	}
}

func TestSolve_FloatPrecision(t *testing.T) {
	// A matrix crafted to test float precision issues.
	// We want to force `0.3 - 0.1 - 0.2` to happen in the algorithm.
	// We know `0.3 - 0.1 - 0.2` != 0.0 in float64.
	cost := [][]float64{
		{0.3, 0.3, 0.3},
		{0.1, 0.1, 0.1},
		{0.2, 0.2, 0.2},
	}
	assignment, err := hungarian.Solve(cost)
	require.NoError(t, err)
	assert.Equal(t, []int{0, 1, 2}, assignment)
}

func TestSolve_JaggedPanic(t *testing.T) {
	// A jagged matrix where m > n (so solveWide is called)
	cost := [][]float64{
		{0, 0},
		{0, 0},
		{0}, // jagged! length 1 instead of 2
	}
	_, err := hungarian.Solve(cost)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "hungarian: cost matrix rows must have the same length")
}
