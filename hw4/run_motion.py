import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
from PIL import Image

def affine_transform_point(point, p):
    '''
    affine transform

    Args:
        point (ndarray): 2D numpy array representing indices of points shape=(2,) in yx format
        p(ndarray): for given p (p1,p2,p3,p4,p5,p6), transformation matrix is [[1+p1,p3,p5],[p2,1+p4,p6],[0,0,1]]
    Returns:
        ndarray: 2D numpy array representing index of warped points shape=(2) in yx format
    '''
    y,x = point
    p1, p2, p3, p4, p5, p6 = p
    T = np.asarray([[1+p1,p3,p5],[p2,1+p4,p6],[0,0,1]])

    return np.asarray([p2*x+(1+p4)*y+p6,(1+p1)*x+p3*y+p5])

def inv_affine_transform_point(point, p):
    '''
    affine transform

    Args:
        point (ndarray): 2D numpy array representing indices of points shape=(2,) in yx format
        p(ndarray): for given p (p1,p2,p3,p4,p5,p6), transformation matrix is [[1+p1,p3,p5],[p2,1+p4,p6],[0,0,1]]
    Returns:
        ndarray: 2D numpy array representing index of warped points shape=(2) in yx format
    '''
    y,x = point
    p1, p2, p3, p4, p5, p6 = p
    T = np.asarray([[1+p1,p3,p5],[p2,1+p4,p6],[0,0,1]])
    T_inv = np.linalg.inv(T)

    return np.asarray([T_inv[1,0]*x+T_inv[1,1]*y+T_inv[1,2],T_inv[0,0]*x+T_inv[0,1]*y+T_inv[0,2]])

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    height1, width1 = img1.shape
    height2, width2 = img2.shape


    #coordinates below are subsampled
    # y_coordinates = np.repeat(np.arange((int)(height1/4), (int)(height1/2)+(int)(height1/4)), (int)(width1/2))
    # x_coordinates = np.tile(np.arange((int)(width1/4), (int)(width1/2)+(int)(width1/4)), (int)(height1/2))
    y_coordinates = np.concatenate((np.repeat(np.arange(240), 80),np.repeat(np.arange(180,240), 240)))
    x_coordinates = np.concatenate((np.tile(np.arange(240,320), 240),np.tile(np.arange(240), 60)))
    yx_coordinates = np.concatenate((y_coordinates.reshape(-1,1),x_coordinates.reshape(-1,1)),axis=1)

    def gradient_I_times_dW_over_dp(yx_coordinate):
        y, x = yx_coordinate
        dIdx = Gx[y, x]/128
        dIdy = Gy[y, x]/128
        return np.asarray([[dIdx*x,dIdy*x,dIdx*y,dIdy*y,dIdx,dIdy]])
    
    def element_of_sigma_for_hessian(yx_coordinate):
        temp = gradient_I_times_dW_over_dp(yx_coordinate)
        return (temp.T).dot(temp)

    hessian = np.sum(np.apply_along_axis(element_of_sigma_for_hessian, 1, yx_coordinates),axis=0)

    interpolated =  RectBivariateSpline(np.arange(height2),np.arange(width2), img2)

    def element_of_sigma_for_dp(yx_coordinate):
        T = img1[yx_coordinate[0],yx_coordinate[1]]
        transformed_yx_coordinate = affine_transform_point(yx_coordinate, p)
        # I = img2[transformed_yx_coordinate[0],transformed_yx_coordinate[1]]
        I = interpolated(transformed_yx_coordinate[0],transformed_yx_coordinate[1])[0,0]
        return (T-I)*gradient_I_times_dW_over_dp(yx_coordinate).T
    
    sigma_term_in_dp = np.sum(np.apply_along_axis(element_of_sigma_for_dp, 1, yx_coordinates),axis=0)
    dp = np.linalg.solve(hessian, sigma_term_in_dp).reshape(6)
    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    height1, width1 = img1.shape
    height2, width2 = img2.shape

    moving_image = np.zeros_like(img2)
    interpolated1 =  RectBivariateSpline(np.arange(height1),np.arange(width1), img1)

    # do not consider points within 5 pixel from boundary
    y_coordinates = np.repeat(np.arange(5,height1-5), width1-10)
    x_coordinates = np.tile(np.arange(5,width1-5), height1-10)
    yx_coordinates = np.concatenate((y_coordinates.reshape(-1,1),x_coordinates.reshape(-1,1)),axis=1)

    def subtract_dominant_motion_pixel(yx_coordinate):
        y2,x2 = yx_coordinate
        y1,x1 = inv_affine_transform_point(yx_coordinate, p)
        moving_image[y2,x2] = np.abs(img2[y2,x2] - interpolated1(y1,x1)[0,0]).astype(np.uint8)

    global p

    dp = lucas_kanade_affine(img1,img2,p,Gx,Gy)

    p += dp
    
    np.apply_along_axis(subtract_dominant_motion_pixel, 1, yx_coordinates)

    # Image.fromarray(moving_image).save("{}.png".format(i))

    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this
    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

p = np.zeros(6)

data_dir = 'data'
video_path = 'motion.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path = os.path.join(data_dir, "{}.jpg".format(0))
T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
for i in range(1, 150):
    img_path = os.path.join(data_dir, "{}.jpg".format(i))
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()
    moving_img = subtract_dominant_motion(T, I)
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    out.write(clone)
    T = I
out.release()

