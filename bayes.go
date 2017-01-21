package bayes

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"sync"
)

type NaiveBayes struct {
	Stats     map[string][]Stats
	Grouped   map[string][][]float64
	ColumnCnt int
	SampleCnt int
	sync.Mutex
}

func New() *NaiveBayes {
	return &NaiveBayes{
		Stats:   map[string][]Stats{},
		Grouped: map[string][][]float64{},
	}
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

	n.SampleCnt = len(data)
	n.ColumnCnt = len(data[0])

	//Separate into groups based on data/labels
	for sampleIdx, label := range labels {
		n.Append(data[sampleIdx], label)
	}

	n.ComputeStats()

	return nil
}

// Add new observation dynamically
// Must call ComputeStats before calling Predict again
func (n *NaiveBayes) Append(data []float64, label string) error {
	if len(data) != n.ColumnCnt {
		return fmt.Errorf("Invalid data: column count mismatch %d != %d", n.ColumnCnt, data)
	}

	///Initialize raw grouped
	if _, ok := n.Grouped[label]; !ok {
		n.Grouped[label] = make([][]float64, n.ColumnCnt)
		for x := 0; x < n.ColumnCnt; x++ {
			n.Grouped[label][x] = []float64{}
		}
	}

	//Set values according to the index
	for columnIdx, val := range data {
		n.Grouped[label][columnIdx] = append(n.Grouped[label][columnIdx], val)
	}

	return nil
}

func (n *NaiveBayes) ComputeStats() {
	//Calculate stats on each label and column
	stats := map[string][]Stats{}
	for label, colVals := range n.Grouped {
		stats[label] = make([]Stats, len(colVals))
		for idx, vals := range colVals {
			stats[label][idx] = CalculateStats(vals)
		}
	}
	n.Stats = stats
}

func (n *NaiveBayes) Predict(data []float64) string {
	return maxVal(n.PredictProbability(data))
}

func (n *NaiveBayes) PredictProbability(data []float64) map[string]float64 {
	probabilities := map[string]float64{}
	for label, stats := range n.Stats {
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

func maxVal(probs map[string]float64) string {
	prediction, maxProb := "", 0.0
	for label, prob := range probs {
		if prob > maxProb {
			prediction = label
			maxProb = prob
		}
	}

	return prediction
}
