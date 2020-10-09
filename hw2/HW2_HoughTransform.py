
import math
import glob
import numpy as np
from PIL import Image, ImageDraw
from cv2 import line

# parameters

datadir = './data'
resultdir='./results'

sigma=2
threshold=0.03
rhoRes=2
thetaRes=math.pi/180
nLines=20

def shift_pixel(pixel_grid, axis, n):
    '''
    shift n pixel along axis

    Args:
        pixel_grid (ndarray): 2D numpy array representing an grayscale image.
        axis(int): axis to shift along
        n(int): how many pixel to shift

    Returns:
        ndarray: 2D numpy array representing an RGB image after shifting.
    '''
    sizeY, sizeX = pixel_grid.shape
    if axis == 0: #along y
        after_shift = np.tile(pixel_grid[sizeY-1],(sizeY,1)) if n>=0 else np.tile(pixel_grid[0],(sizeY,1))
        after_shift[max(0,-n):sizeY-n,:] = pixel_grid[max(0,+n):sizeY+n, :]
    elif (axis == 1): #along x
        after_shift = np.tile(pixel_grid[:,sizeX-1].reshape(-1,1),sizeX) if n>=0 else np.tile(pixel_grid[:,0].reshape(-1,1),sizeX)
        after_shift[:,max(0,-n):sizeX-n] = pixel_grid[:,max(0,+n):sizeX+n]
    else:
        raise NotImplementedError
    
    return after_shift

def getGaussianKernel(sigma):
    '''
    get gaussian kernal of std sigma
    consider up to 3*sigma, that is size of kernel is 1(center) + int(3*sigma)

    Args:
        sigma(float):std of the kernel

    Returns:
        ndarray: 2D numpy array representing an gaussian kernel.
    '''
    halfSize = int(3*sigma) # half of size-1 indeed
    dist1dSq = np.square(np.arange(-halfSize,halfSize+1))
    dist2dSq = dist1dSq.reshape(-1,1)+dist1dSq

    kernel = np.exp(-0.5*dist2dSq/np.square(sigma))

    return kernel/np.sum(kernel)    


def ConvFilter(Igs, G):
    # TODO ...
    sizeY = G.shape[0]
    sizeX = G.shape[1]
    toConv = []
    for i in range(-int(sizeY/2),sizeY-int(sizeY/2)):
        temp=[]
        for j in range(-int(sizeX/2),sizeX-int(sizeX/2)):
            temp.append(shift_pixel(shift_pixel(Igs,0,i),1,j))
        toConv.append(temp)
    toConv = np.asarray(toConv)
    #shape of toConv is (sizeY, sizeX, Igs.shape[0],Igs.shape[1])
    G = G.reshape(sizeY,sizeX,1,1)

    Iconv = np.sum(toConv*G,axis=(0,1))

    return Iconv

def EdgeDetection(Igs, sigma):
    # TODO ...
    gk = getGaussianKernel(sigma)
    Igs = ConvFilter(Igs, gk)
    sobelX = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelY = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix = ConvFilter(Igs,sobelX)
    Iy = ConvFilter(Igs,sobelY)
    Im = np.sqrt(Ix**2+Iy**2)
    Io = np.arctan(Iy/Ix)

    return Im, Io, Ix, Iy

def nonMaximumSuppresion(Im, Io = None, neighborSize = (3,10)):
    '''
    non-maxmimum suppression

    Args:
        Im(ndarray): 2D numpy array representing an edge magnitude image.
        Io(ndarray): 2D numpy array representing an edge orientation image(rad).
                    if None, consider all neighboring pixels in non-maximum suppression
        neighborSize(int, int): half of suppression kernal size, only used when Io = None

    Returns:
        ndarray: 2D numpy array representing edge image after non-maximum suppresion

    '''
    sizeY, sizeX = Im.shape
    result = Im.copy()
    if (Io is None):
        neighbor = []
        for i in range(-neighborSize[0],neighborSize[0]+1):
            for j in range(-neighborSize[1],neighborSize[1]+1):
                if(i==0 and j==0): continue
                neighbor.append(shift_pixel(shift_pixel(result,0,i),1,j))
        neighbor = np.asarray(neighbor)
        neighborMax = np.max(neighbor, axis=0)
        result[Im < neighborMax] = 0

    else:
        for i in range(sizeY):
            for j in range(sizeX):
                try:
                    angle = Io[i,j]
                    if (angle < -np.pi/4):
                        temp = math.tan(angle + np.pi/2)
                        p = (1-temp)*Im[i+1,j]+temp*Im[i+1,j+1]
                        q = (1-temp)*Im[i-1,j]+temp*Im[i-1,j-1]
                    elif (angle < 0):
                        temp = math.tan(-angle)
                        p = (1-temp)*Im[i,j+1]+temp*Im[i+1,j+1]
                        q = (1-temp)*Im[i,j-1]+temp*Im[i-1,j-1]
                    elif (angle < np.pi/4):
                        temp = math.tan(angle)
                        p = (1-temp)*Im[i,j+1]+temp*Im[i-1,j+1]
                        q = (1-temp)*Im[i,j-1]+temp*Im[i+1,j-1]
                    elif (angle <= np.pi/2):
                        temp = math.tan(np.pi/2-angle)
                        p = (1-temp)*Im[i-1,j]+temp*Im[i-1,j+1]
                        q = (1-temp)*Im[i+1,j]+temp*Im[i+1,j-1]
                    else:
                        pass
                except IndexError:
                    pass
                if(Im[i,j]<q or Im[i,j]<p):
                    result[i,j] = 0

    return result

def doubleThresholding(Im, threshold1, threshold2): # not used

    result = np.zeros_like(Im,dtype = np.uint8)
    result[Im<threshold1] = 0
    result[np.logical_and(Im>=threshold1,Im<threshold2)] = 127
    result[Im>=threshold2] = 255
    return result

def edgeTracking(Im): # not used
    result = Im.copy()
    
    while(True): #iterate until no update
        isWeak = result ==127
        neighbor = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if(i==0 and j==0): continue
                neighbor.append(shift_pixel(shift_pixel(result,0,i),1,j))
        neighbor = np.asarray(neighbor)
        continuity = np.max(neighbor,axis=0) == 255
        weakToStrong = np.logical_and(isWeak,continuity)
        if not np.any(weakToStrong):
            break
        result[weakToStrong] = 255

    result[np.logical_and(isWeak,np.invert(continuity))] = 0

    return result

def Canny(Igs, sigma, threshold1, threshold2): # not used
    '''
    Canny Edge Operator

    Args:
        Igs(ndarray): 2D numpy array representing an grayscale image.
        sigma(float): std of the gaussian kernel to blur with
        threshold1(float): if edge magnitude smaller than this, definitely not an edge.
                            else, depends on context
        threshold2(float): if edge magnitude larger than this, definitely an edge.

    Returns:
        ndarray: 2D numpy array representing an grayscale image after applying Canny Edge Operator.
    '''
    
    Im, Io, _, _ = EdgeDetection(Igs, sigma)
    result = nonMaximumSuppresion(Im, Io)
    result = doubleThresholding(result, threshold1, threshold2)
    result = edgeTracking(result)

    return result



def HoughTransform(Im,threshold, rhoRes, thetaRes):
    # TODO ...
    sizeY, sizeX = Im.shape
    rhoMax = math.sqrt(sizeY**2+sizeX**2)
    sizeRho = (int)(rhoMax/rhoRes)
    sizeTheta = (int)(2*np.pi/thetaRes)
    
    H = np.zeros((sizeRho, sizeTheta), dtype=int)

    for y, x in zip(*np.where(Im>threshold)):
        try:
            thetaAxis = np.arange(sizeTheta)
            thetas = thetaRes*thetaAxis
            rhos = x*np.cos(thetas)+y*np.sin(thetas)
            indices = np.where(rhos>0)
            H[np.rint(rhos[indices]/rhoRes).astype(int),indices] += 1
        except IndexError:
            continue

    return H

def drawLines(Igs, lRho, lTheta):
    result = Igs.copy()
    sizeY, sizeX, _ = Igs.shape

    for rho, theta in zip(lRho, lTheta):
        for i in range(sizeY):
            try:
                j = (int)(round(((rho-i*math.sin(theta))/math.cos(theta))))
                if(j in range(sizeX)):
                    result[i,j] = [255, 0, 0] 
            except OverflowError:
                break
            except ValueError:
                break
        for j in range(sizeX):
            try:
                i = (int)(round(((rho-j*math.cos(theta))/math.sin(theta))))
                if(i in range(sizeY)):
                    result[i,j] = [255, 0, 0]
            except OverflowError:
                break
            except ValueError:
                break

    return result

def HoughLines(H,rhoRes,thetaRes,nLines): #return is radian
    # TODO ...
    sizeX = H.shape[1]
    lRho = np.zeros(nLines,dtype=float)
    lTheta = np.zeros(nLines,dtype=float)
    H = nonMaximumSuppresion(H)
    for i in range(nLines):
        temp = np.argmax(H)
        rho = int(temp/sizeX)
        theta = temp % sizeX
        H[rho, theta] = -1
        lRho[i] = rho*rhoRes
        lTheta[i] = theta*thetaRes

    return lRho,lTheta

def getCluster(points, point):

    '''
    extract cluster that pixel (i,j) belongs to from given points

    Args:
        points (list of 2D int tuples): point set, extracted points will be removed from the list
        point (2D int tuple): point to get cluster

    Returns:
        list of 2D int tuples: cluster set

    '''
    i = point[0]
    j = point[1]
    cluster = []
    newPoints = [(i,j)]
    while(True):
        try:
            newPoint = newPoints.pop()
            cluster.append(newPoint)
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if(i==0 and j==0): continue
                    temp = (newPoint[0]+i, newPoint[1]+j)
                    if(temp in points):
                        newPoints.append(temp)
                        points.remove(temp)
        except IndexError:
            break

    return cluster


def HoughLineSegments(lRho, lTheta, Im, threshold):
    # TODO ...
    sizeY, sizeX = Im.shape
    
    l = []
    for rho, theta in zip(lRho, lTheta):
        label = np.zeros_like(Im, dtype=bool)
        for i in range(sizeY):
            try:
                j = (int)(round(((rho-i*np.sin(theta))/np.cos(theta))))
                if(j in range(sizeX)):
                    label[i-2:i+3,j-2:j+3] = True
            except OverflowError:
                break
            except ValueError:
                break
        for j in range(sizeX):
            try:
                i = (int)(round(((rho-j*np.cos(theta))/np.sin(theta))))
                if(i in range(sizeY)):
                    label[i-2:i+3,j-2:j+3] = True
            except OverflowError:
                break
            except ValueError:
                break
        points = list(zip(*np.where(np.logical_and(label, Im>threshold))))
        #label contains the lines for given lRho, lTheta and their adjacent pixels

        largestCluster = []
        while(True):
            try:
                point = points.pop()
                cluster = getCluster(points,point)
                if (len(cluster)>len(largestCluster)):
                    largestCluster = cluster
            except IndexError:
                break
        
        if(not largestCluster): continue
        temp = np.asarray(largestCluster)
        if(theta%np.pi/2<np.pi/4):
            yS = np.min(temp,axis=0)[0]
            yE = np.max(temp,axis=0)[0]
            xS = (int)(round(((rho-yS*math.sin(theta))/math.cos(theta))))
            xE = (int)(round(((rho-yE*math.sin(theta))/math.cos(theta))))
        else:
            xS = np.min(temp,axis=0)[1]
            xE = np.max(temp,axis=0)[1]
            yS = (int)(round(((rho-xS*math.cos(theta))/math.sin(theta))))
            yE = (int)(round(((rho-xE*math.cos(theta))/math.sin(theta))))

        xy = {'start':(xS,yS), 'end':(xE,yE)}
        l.append(xy)
        # l.append(largestCluster)
        
    # Image.fromarray(255*np.logical_and(label, Im>threshold).astype(np.uint8)).show()

    
    # return np.logical_and(label, Im>threshold)
    return l

def main():

    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")
        img1 = Image.open(img_path).convert("RGB")

        Igs = np.array(img)
        Igs1 = np.asarray(img1)
        Igs = Igs / 255.
        


        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        Im = nonMaximumSuppresion(Im, Io) #added



        H = HoughTransform(Im, threshold, rhoRes, thetaRes)
        # Image.fromarray(H.astype(float)).show()

        # H = nonMaximumSuppresion(H)
        # Image.fromarray(H.astype(float)).show()

        lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines)
        # print(lRho, lTheta)
        Igs1 = drawLines(Igs1,lRho,lTheta)
        # Igs1 = drawLines(np.stack([Im, Im, Im], axis = -1).as, lRho, lTheta, rhoRes, thetaRes)
        # print('Igs1', Igs1.dtype)
        # Image.fromarray(Igs1).show()

        # H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        # np.save('hough.npy',H)

        # H = nonMaximumSuppresion(H)
        # Image.fromarray(H).show()


        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im, threshold)
        draw = ImageDraw.Draw(img1)

        for line in l:
            draw.line([line['start'],line['end']],fill=(0,255,0))
        img1.show()
        # print('l0',l[0])
        # for line in l:
        #     for point in line:
        #         Igs1[point] = [0,255,0]
        # Image.fromarray(Igs1).show()
        # Image.fromarray(255*l.astype(np.uint8)).show()

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()