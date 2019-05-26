import numpy as np
import glob
import pdb
import os
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
def computeMHIEC(directoryName):
    depthfiles = glob.glob(directoryName + '*.pgm');
    depthfiles = np.sort(depthfiles)
    mhi = None
    maxMhi = -1
    for i in range(len(depthfiles)):
        depth = imread(depthfiles[i])
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if depth[y][x] > 39500:
                    depth[y][x] = 0
                else:
                    depth[y][x] = 1
        if mhi is None:
            mhi = np.zeros(depth.shape)
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if depth[y][x] == 1:
                    mhi[y][x] = len(depthfiles)
                else:
                    mhi[y][x] = max(0, mhi[y][x]-1)
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            maxMhi = max(maxMhi, mhi[y][x])
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            mhi[y][x] = mhi[y][x] / maxMhi
    return mhi

def computeMHI(directoryName):
    depthfiles = glob.glob(directoryName + '*.pgm');
    depthfiles = np.sort(depthfiles)
    mhi = None
    maxMhi = -1
    for i in range(len(depthfiles)):
        depth = imread(depthfiles[i])
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if x > 570:
                    depth[y][x] = 0
                elif depth[y][x] > 39800:
                    depth[y][x] = 0
                else:
                    depth[y][x] = 1
        if mhi is None:
            mhi = np.zeros(depth.shape)
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if depth[y][x] == 1:
                    mhi[y][x] = len(depthfiles)
                else:
                    mhi[y][x] = max(0, mhi[y][x]-1)
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            maxMhi = max(maxMhi, mhi[y][x])
    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):
            mhi[y][x] = mhi[y][x] / maxMhi
    return mhi

basedir = './'
actions = ['botharms', 'crouch', 'leftarmup', 'punch', 'rightkick']
allMHIs = None
for actionnum in range(len(actions)):
    subdirname = basedir + actions[actionnum] + '/'
    dirName =  basedir + 'MHI40000/' + actions[actionnum] + '/'
    try:
        os.makedirs(dirName)
    except FileExistsError:
        pass
    subdir = os.listdir(subdirname)
    for seqnum in range(len(subdir)):
        directoryName = subdirname + subdir[seqnum] + '/'
        mhi = computeMHIEC(directoryName)
        if allMHIs is None:
            allMHIs = mhi
        else:
            allMHIs = np.dstack((allMHIs, mhi))
        name = dirName + subdir[seqnum] + ".png"
        imsave(name, mhi)
np.save("allMHIs40000.npy", allMHIs)
