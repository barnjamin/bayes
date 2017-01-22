package bayes

import "math/rand"

func CrossValidate(data [][]float64, labels []string, percentage float64) float64 {
	nb := New()
	trainData, trainLabels, testData, testLabels := Split(data, labels, percentage)
	nb.Fit(trainData, trainLabels)

	correct := 0
	for idx, label := range testLabels {
		if nb.Predict(testData[idx]) == label {
			correct++
		}
	}

	return (float64(correct) / float64(len(testLabels))) * 100
}

func Split(data [][]float64, labels []string, percentage float64) ([][]float64, []string, [][]float64, []string) {

	trainD, testD := [][]float64{}, [][]float64{}
	trainL, testL := []string{}, []string{}
	count := int(float64(len(data)) * percentage)

	idxMap := map[int]bool{}
	for len(idxMap) < count {
		idx := rand.Intn(len(data))
		if _, ok := idxMap[idx]; !ok {
			idxMap[idx] = true
		}
	}

	for idx, vals := range data {
		if _, ok := idxMap[idx]; ok {
			testD = append(testD, data[idx])
			testL = append(testL, labels[idx])
		} else {
			trainD = append(trainD, vals)
			trainL = append(trainL, labels[idx])
		}
	}

	return trainD, trainL, testD, testL
}
