import numpy as np
import matplotlib.pyplot as plt
import math
from predictAction import predictAction
from predictAction import predictActionMean
from predictAction import predictActionEC
from predictAction import predictActionMode
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
trainLabelsAll = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
means = {}
total = {}
for trainLabel in trainLabelsAll:
    means[trainLabel] = 0
    if trainLabel not in total:
        total[trainLabel] = 1
    else:
        total[trainLabel] += 1
huVectors = np.load("huVectors.npy")
# huVectors = np.load("huVectors40000.npy")
confusionMat = np.zeros((len(actions), len(actions)), dtype=int)
for i in range(huVectors.shape[0]):
    trainMoments = np.delete(huVectors, i, 0)
    trainLabels = np.delete(trainLabelsAll, i, 0)
    testMoments = huVectors[i]
    # res = predictAction(testMoments, trainMoments, trainLabels)
    # res = predictActionEC(testMoments, trainMoments, trainLabels)
    # res = predictActionMean(testMoments, trainMoments, trainLabels)
    res = predictActionMode(testMoments, trainMoments, trainLabels)
    print(res)
    confusionMat[trainLabelsAll[i]-1][res-1] += 1
    if res == trainLabelsAll[i]:
        means[res] += 1
for key in means.keys():
    means[key] /= total[key]
    print(actions[key-1], " : ", means[key])
fig, axs =plt.subplots()
axs.axis('tight')
axs.axis('off')
the_table = axs.table(cellText=confusionMat,colLabels=actions, rowLabels=actions,loc='center')
plt.show()
