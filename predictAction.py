import numpy as np
import matplotlib.pyplot as plt
import math
import statistics as st

def normalizedED(testMoments, trainMoments, trainLabels):
    distances = []
    variances = np.var(trainMoments, axis=0)
    for j in range(trainMoments.shape[0]):
        distance = 0
        for i in range(trainMoments.shape[1]):
            distance += ((testMoments[i] - trainMoments[j][i]) ** 2) / variances[i]
        distances.append((math.sqrt(distance), j))
    distances.sort(key=lambda x: x[0])
    temp = []
    for dist in distances:
        temp.append(dist[1])
    print("st", temp, trainLabels)
    return distances

def normalizedEDEC(testMoments, trainMoments, trainLabels):
    distances = []
    variances = np.square(np.var(trainMoments, axis=0))
    for j in range(trainMoments.shape[0]):
        distance = 0
        for i in range(trainMoments.shape[1]):
            distance += ((testMoments[i] - trainMoments[j][i]) ** 2) / variances[i]
        distances.append((math.sqrt(distance), j))
    distances.sort(key=lambda x: x[0])
    temp = []
    for dist in distances:
        temp.append(dist[1])
    print("st", temp, trainLabels)
    return distances

def predictAction(testMoments, trainMoments, trainLabels):
    return trainLabels[normalizedED(testMoments, trainMoments, trainLabels)[0][1]]

def predictActionEC(testMoments, trainMoments, trainLabels):
    return trainLabels[normalizedEDEC(testMoments, trainMoments, trainLabels)[0][1]]

def predictActionMean(testMoments, trainMoments, trainLabels):
    k = 4
    distances = normalizedED(testMoments, trainMoments, trainLabels)
    mean = 0
    labels = []
    for i in range(k):
        mean+= trainLabels[distances[i][1]]
        labels.append(trainLabels[distances[i][1]])
    print(labels)
    mean = int(round(mean/k))
    if mean > 5:
        return 5
    elif mean < 1:
        return 1
    else:
        return mean

def predictActionMode(testMoments, trainMoments, trainLabels):
    k = 4
    distances = normalizedED(testMoments, trainMoments, trainLabels)
    labels = []
    for i in range(k):
        labels.append(trainLabels[distances[i][1]])
    try:
        label = st.mode(labels)
    except st.StatisticsError:
        label = labels[0]
    return label
