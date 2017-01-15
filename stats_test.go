package bayes

import "testing"

var acceptableDelta = 0.001
var probaTestTable = []struct {
	Stats Stats
	Value float64
	Proba float64
}{
	{
		Stats: Stats{Mean: 73.0, Std: 6.2},
		Value: 71.5,
		Proba: 0.0624,
	},
}

func TestCalculateProbability(t *testing.T) {
	for _, test := range probaTestTable {
		proba := test.Stats.CalculateProbability(test.Value)
		if (proba - test.Proba) > acceptableDelta {
			t.Errorf("Probability not close enough, got: %.5f expected %.5f", proba, test.Proba)
		}
	}
}
