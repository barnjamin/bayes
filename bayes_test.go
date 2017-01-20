package bayes

import (
	"log"
	"os"
	"testing"
)

func Example() {
	data, labels := generateData()

	nb := New()
	if err := nb.Fit(data, labels); err != nil {
		log.Printf("Failed to fit data: %+v", err)
	}

	for idx, label := range labels {
		prediction := nb.Predict(data[idx])
		log.Printf("Expected: %s Got: %s", label, prediction)
	}
}

func TestNew(t *testing.T) {
	if nb := New(); nb == nil {
		t.Errorf("Failed to create new NaiveBayes")
	}
}

func TestFit(t *testing.T) {
	nb := New()
	d, l := generateData()

	if err := nb.Fit(d, l); err != nil {
		t.Errorf("Failed to fit data: %+v", err)
	}
}

func TestPredict(t *testing.T) {
	nb := New()
	d, l := generateData()

	if err := nb.Fit(d, l); err != nil {
		t.Errorf("Failed to fit data: %+v", err)
	}

	if p := nb.Predict([]float64{6.0}); p != "b" {
		t.Errorf("Failed to predict the correct result: %s", p)
	}
}

func TestDump(t *testing.T) {
	nb := New()
	d, l := generateData()

	if err := nb.Fit(d, l); err != nil {
		t.Errorf("Failed to fit data: %+v", err)
	}

	f, err := os.Create("/tmp/test-dump")
	if err != nil {
		t.Errorf("Failde to create tmp file: %+v", err)
	}

	err = nb.Dump(f)
	if err != nil {
		t.Errorf("Failde to create tmp file: %+v", err)
	}
}

func TestLoad(t *testing.T) {
	f, err := os.Open("/tmp/test-dump")
	if err != nil {
		t.Errorf("Failed to open dump file: %+v", err)
	}

	_, err = Load(f)
	if err != nil {
		t.Errorf("Failed to load from dump file: %+v", err)
	}

}

func generateData() ([][]float64, []string) {
	return [][]float64{{1.0}, {2.0}, {5.0}, {6.0}, {10.0}}, []string{"a", "a", "b", "b", "c"}
}
