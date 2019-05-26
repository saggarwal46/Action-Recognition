import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import math
from predictAction import normalizedED

def showNearestMHI(testMoments, trainMoments, trainLabels, allMHIs, k, temp):
    distances = normalizedED(testMoments, trainMoments, trainLabels)
    print(distances)
    for i in range(k):
        idx = distances[i][1]
        img = allMHIs[:,:,idx]
        plt.imshow(img, cmap='gray')
        name = "./" + str(temp) + "_" + str(i) + ".png"
        imsave(name, img)
        plt.show()

if __name__ == '__main__':
    actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
    trainLabelsAll = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    huVectors = np.load("huVectors.npy")
    overallMHIs = np.load("allMHIs.npy")
    for idx in [5, 16]:
        img = overallMHIs[:,:,idx]
        plt.imshow(img, cmap='gray')
        name = "./" + str(idx) + ".png"
        imsave(name, img)
        plt.show()
    print("starting")
    for i in [5, 16]:
        trainMoments = np.delete(huVectors, i, 0)
        trainLabels = np.delete(trainLabelsAll, i, 0)
        testMoments = huVectors[i]
        allMHIs = np.delete(overallMHIs, i, 2)
        showNearestMHI(testMoments, trainMoments, trainLabels, allMHIs, 4, i)
