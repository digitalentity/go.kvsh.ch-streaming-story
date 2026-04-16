package dist

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNorm(t *testing.T) {
	tests := []struct {
		name string
		v    []float32
		want float32
	}{
		{"empty", []float32{}, 0},
		{"zero", []float32{0, 0}, 0},
		{"unit_x", []float32{1, 0}, 1},
		{"unit_y", []float32{0, 1}, 1},
		{"3_4_5", []float32{3, 4}, 5},
		{"negative", []float32{-3, -4}, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, Norm(tt.v), 1e-6)
		})
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", []float32{}, []float32{}, 0},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"parallel", []float32{1, 2}, []float32{1, 2}, 5},
		{"negative", []float32{1, 2}, []float32{-1, -2}, -5},
		{"mismatch_length", []float32{1, 2}, []float32{1}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, Dot(tt.a, tt.b), 1e-6)
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", []float32{}, []float32{}, 0},
		{"zero_norm", []float32{0, 0}, []float32{1, 1}, 0},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"opposite", []float32{1, 2, 3}, []float32{-1, -2, -3}, -1},
		{"45_degrees", []float32{1, 0}, []float32{1, 1}, float32(1.0 / math.Sqrt(2.0))},
		{"mismatch_length", []float32{1, 2}, []float32{1}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, CosineSimilarity(tt.a, tt.b), 1e-6)
		})
	}
}

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 1},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.InDelta(t, tt.want, CosineDistance(tt.a, tt.b), 1e-6)
		})
	}
}
