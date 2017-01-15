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
	idxList := []int{}
	for x := 0; x < count; x++ {
		idx := rand.Intn(len(data))
		idxMap[idx] = true
		idxList = append(idxList, idx)
	}

	trainD, testD := [][]float64{}, [][]float64{}
	trainL, testL := []string{}, []string{}

	for _, idx := range idxList {
		testD = append(testD, data[idx])
		testL = append(testL, labels[idx])
	}

	for idx, vals := range data {
		if _, ok := idxMap[idx]; !ok {
			trainD = append(trainD, vals)
			trainL = append(trainL, labels[idx])
		}
	}

	return trainD, trainL, testD, testL
}
