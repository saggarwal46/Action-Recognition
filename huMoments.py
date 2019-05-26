import numpy as np
import matplotlib.pyplot as plt

class ImageMoment:
    def __init__(self, H):
        self.H = H
    def imageMoment(self, i, j):
        moment = 0
        for y in range(self.H.shape[0]):
            for x in range(self.H.shape[1]):
                moment += (x ** i) * (y ** j) * self.H[y][x]
        return moment
    def centralMoment(self, i, j, x_mean, y_mean):
        moment = 0
        for y in range(self.H.shape[0]):
            for x in range(self.H.shape[1]):
                moment += ((x - x_mean) ** i) * ((y - y_mean) ** j) * self.H[y][x]
        return moment
def huMoments(H):
    imgMmnts = ImageMoment(H)
    x_mean = imgMmnts.imageMoment(1,0)/imgMmnts.imageMoment(0,0)
    y_mean = imgMmnts.imageMoment(0,1)/imgMmnts.imageMoment(0,0)
    u11 = imgMmnts.centralMoment(1, 1, x_mean, y_mean)
    u12 = imgMmnts.centralMoment(1, 2, x_mean, y_mean)
    u02 = imgMmnts.centralMoment(0, 2, x_mean, y_mean)
    u20 = imgMmnts.centralMoment(2, 0, x_mean, y_mean)
    u21 = imgMmnts.centralMoment(2, 1, x_mean, y_mean)
    u03 = imgMmnts.centralMoment(0, 3, x_mean, y_mean)
    u30 = imgMmnts.centralMoment(3, 0, x_mean, y_mean)

    h1 = u20 + u02

    h2 = ((u20 - u02) ** 2)  + 4 * (u11**2)

    h3 = ((u30 - 3 * u12) ** 2) + ((3 * u21 - u03) ** 2)

    h4 = (u30 + u12) ** 2 + (u21 + u03) ** 2

    h5 = (u30 - 3 * u12) * (u30 + u12) * ((u30 + u12) ** 2 - (3 * ((u21 + u03) ** 2)))
    h5 += (3 * u21 - u03) * (u21 + u03) * ( (3 * ((u30 + u12) ** 2)) - (u21 + u03) ** 2 )

    h6 = (u20 - u02) * ( ((u30 + u12) ** 2) - ((u21 + u03) ** 2) )
    h6 += 4 * u11 * (u30 + u12) * (u21 + u03)

    h7 = ( 3 * u21 - u03) * (u30 + u12) * ((u30 + u12) ** 2 - (3 * ((u21 + u03) ** 2)))
    h7 -= (u30 - 3 * u12) * (u21 + u03) * ( (3 * ((u30 + u12) ** 2)) - (u21 + u03) ** 2 )

    h = [h1, h2, h3, h4, h5, h6, h7]

    return h

# allMHIs = np.load("allMHIs.npy")

allMHIs = np.load("allMHIs40000.npy")
huVectors = None
for i in range (allMHIs.shape[2]):
    h = np.asarray(huMoments(allMHIs[:][:][i]))
    print(h)
    if huVectors is None:
        huVectors = h
    else:
        huVectors = np.vstack((huVectors, h))
    print(huVectors.shape)
# np.save("huVectors.npy", huVectors)
np.save("huVectors40000.npy", huVectors)
