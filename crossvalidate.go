package bayes

import "math/rand"

func CrossValidate(data [][]float64, labels []string, percentage float64) float64 {
	nb := New()
	trainData, trainLabels, testData, testLabels := split(data, labels, percentage)
	nb.Fit(trainData, trainLabels)

	correct := 0
	for idx, label := range testLabels {
		if nb.Predict(testData[idx]) == label {
			correct++
		}
	}

	return (float64(correct) / float64(len(testLabels))) * 100
}

func split(data [][]float64, labels []string, percentage float64) ([][]float64, []string, [][]float64, []string) {
	count := int(float64(len(data)) * percentage)
	idxMap := map[int]bool{}
	for x := 0; x < count; x++ {
		idxMap[rand.Intn(len(data))] = true
	}

	trainD, testD := [][]float64{}, [][]float64{}
	trainL, testL := []string{}, []string{}
	for idx, vals := range data {
		if _, ok := idxMap[idx]; ok {
			testD = append(testD, vals)
			testL = append(testL, labels[idx])
		} else {
			trainD = append(trainD, vals)
			trainL = append(trainL, labels[idx])
		}
	}

	return trainD, trainL, testD, testL
}
