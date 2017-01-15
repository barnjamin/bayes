package bayes

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"

	"math"

	"github.com/gonum/stat"
)

type NaiveBayes struct {
	Stats   map[string][]Stats
	Grouped map[string][][]float64
}

func New() *NaiveBayes {
	return &NaiveBayes{}
}

func Load(r io.Reader) (*NaiveBayes, error) {
	nb := New()
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nb, err
	}

	err = json.Unmarshal(b, nb)

	return nb, err
}

func (n *NaiveBayes) Fit(data [][]float64, labels []string) error {
	if len(data) == 0 || len(data) != len(labels) {
		return errors.New("Invalid data: data and label length dont match or are 0")
	}

	columns := len(data[0])

	//Separate into groups based on data/labels
	grouped := map[string][][]float64{}
	for sampleIdx, label := range labels {

		if len(data[sampleIdx]) != columns {
			return fmt.Errorf("Invalid data: column count mismatch %d != %d", columns, data[sampleIdx])
		}

		///Initialize raw grouped
		if _, ok := grouped[label]; !ok {
			grouped[label] = make([][]float64, columns)
			for x := 0; x < columns; x++ {
				grouped[label][x] = []float64{}
			}
		}

		//Set values according to the index
		for columnIdx, val := range data[sampleIdx] {
			grouped[label][columnIdx] = append(grouped[label][columnIdx], val)
		}
	}
	n.Grouped = grouped

	//Calculate stats on each group
	stats := map[string][]Stats{}
	for label, colVals := range n.Grouped {
		stats[label] = make([]Stats, len(colVals))
		for idx, vals := range colVals {
			mean, std := stat.MeanStdDev(vals, nil)
			if math.IsNaN(std) {
				std = 0.0
			}
			stats[label][idx] = Stats{
				Mean: mean,
				Std:  std,
			}
		}
	}

	n.Stats = stats

	return nil
}

func (n *NaiveBayes) Predict(data []float64) string {
	probs := n.PredictProbability(data)

	prediction, maxProb := "", 0.0
	for label, prob := range probs {
		if prob > maxProb {
			prediction = label
			maxProb = prob
		}
	}

	return prediction
}

func (n *NaiveBayes) PredictProbability(data []float64) map[string]float64 {
	probabilities := map[string]float64{}
	for label, stats := range n.Stats {
		if _, ok := probabilities[label]; !ok {
			probabilities[label] = 0.0
		}
		for idx, stat := range stats {
			probabilities[label] += stat.CalculateProbability(data[idx])
		}
	}

	return probabilities
}

func (n *NaiveBayes) Dump(w io.Writer) error {
	b, err := json.Marshal(n)
	if err != nil {
		return err
	}

	_, err = w.Write(b)

	return err
}
