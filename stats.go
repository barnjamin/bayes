package bayes

import "math"

type Stats struct {
	Mean float64
	Std  float64
}

func (s *Stats) CalculateProbability(val float64) float64 {
	exp := math.Exp(-(math.Pow(val-s.Mean, 2) / (2 * math.Pow(s.Std, 2))))
	return (1 / (math.Sqrt(2*math.Pi) * s.Std)) * exp
}
