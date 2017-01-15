package bayes

import (
	"math"

	"github.com/gonum/stat"
)

type Stats struct {
	Mean float64
	Std  float64
}

func CalculateStats(vals []float64) Stats {

	mean, std := stat.MeanStdDev(vals, nil)

	if math.IsNaN(std) {
		std = 0.0
	}

	return Stats{Mean: mean, Std: std}
}

func (s *Stats) CalculateProbability(val float64) float64 {
	exp := math.Exp(-(math.Pow(val-s.Mean, 2) / (2 * math.Pow(s.Std, 2))))
	return (1 / (math.Sqrt(2*math.Pi) * s.Std)) * exp
}
