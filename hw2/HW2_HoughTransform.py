
import math
import glob
import numpy as np
from PIL import Image


# parameters

datadir = '../data'
resultdir='../results'

sigma=2
threshold=0.03
rhoRes=2
thetaRes=math.pi/180
nLines=20


def ConvFilter(Igs, G):
    # TODO ...

    return Iconv

def EdgeDetection(Igs, sigma):
    # TODO ...


    return Im, Io, Ix, Iy

def HoughTransform(Im,threshold, rhoRes, thetaRes):
    # TODO ...


    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    # TODO ...


    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im, threshold):
    # TODO ...


    return l

def main():

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im, threshold)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()