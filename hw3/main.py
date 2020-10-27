import math
import numpy as np
from PIL import Image

def compute_A(p1, p2): #shape = (2*n, 9)
    n = p1.shape[0]
    A = np.zeros((2*n, 9))

    A[0::2,:2] = p2
    A[0::2,2] = 1
    A[1::2,3:5] = p2
    A[1::2,5] = 1
    A[0::2,6:8] = -p1[:,0].reshape(n,1) * p2
    A[1::2,6:8] = -p1[:,1].reshape(n,1) * p2
    A[:,8] = -p1.reshape(2*n)

    return A

def compute_h(p1, p2):
    # TODO ...
    A = compute_A(p1, p2)

    u, s, vh = np.linalg.svd(A.T.dot(A), compute_uv=True)
    H = vh[-1].reshape(3,3)

    return H

def compute_h_norm(p1, p2):
    # TODO ...
    # normalize matrix
    nm_p1 = np.eye(3)
    nm_p1[[0,1],[0,1]] = 1/(p1.max(axis=0) - p1.min(axis=0))
    nm_p1[[0,1],[2,2]] = -p1.min(axis=0)/(p1.max(axis=0) - p1.min(axis=0))

    nm_p2 = np.eye(3)
    nm_p2[[0,1],[0,1]] = 1/(p2.max(axis=0) - p2.min(axis=0))
    nm_p2[[0,1],[2,2]] = -p2.min(axis=0)/(p2.max(axis=0) - p2.min(axis=0))

    nm_p1_inv = np.linalg.inv(nm_p1)

    #normalized
    p1_norm = warp_points(p1, nm_p1, return_float=True)
    p2_norm = warp_points(p2, nm_p2, return_float=True)


    H = compute_h(p1_norm, p2_norm)

    return nm_p1_inv.dot(H.dot(nm_p2))

def warp_points(points, H, return_float = False):
    '''
    warp points

    Args:
        points (ndarray): 2D numpy array representing indices of points shape=(N, 2)
        H(ndarray): 3 by 3 homography matrix
        return_flaot : If true, return normalized float index(from 0 to 1)

    Returns:
        ndarray: 3D numpy array representing index of warped points shape=(N, 2)
    '''
    n, _ = points.shape
    homogeneous = np.concatenate((points.T, np.ones((1,n))),axis = 0)

    warped_homogeneous = H.dot(homogeneous)
    scales = warped_homogeneous[2,:]

    if (return_float == True) : return (warped_homogeneous/scales)[:2].T
    else: return np.rint(warped_homogeneous/scales)[:2].astype(int).T

def warp_image(igs_in, igs_ref, H):
    # TODO ...
    height_in, width_in, _ = igs_in.shape
    height_ref, width_ref, _ = igs_ref.shape
    H_inv = np.linalg.inv(H)

    #calculate igs_warp
    igs_warp = np.zeros_like(igs_ref,dtype=np.uint8)
    y_indices = np.repeat(np.arange(height_ref), width_ref)
    x_indices = np.tile(np.arange(width_ref), height_ref)
    indices = np.concatenate((x_indices.reshape(-1,1),y_indices.reshape(-1,1)),axis=1)

    indices_before_warp = warp_points(indices, H_inv)
    for y in range(height_ref):
        for x in range(width_ref):
            index_before_warp = indices_before_warp[y*width_ref+x]
            x_before_warp = index_before_warp[0]
            y_before_warp = index_before_warp[1]
            if(0<=y_before_warp<height_in and 0<=x_before_warp<width_in):
                igs_warp[y,x] = igs_in[y_before_warp,x_before_warp]

    # calculate igs_merge
    corners = warp_points(np.asarray([[0,0],[width_in-1,0],[0,height_in-1],[width_in-1,height_in-1]]),H)
    warp_end_width, warp_end_height = corners.max(axis = 0)
    warp_start_width, warp_start_height = corners.min(axis = 0)

    pad_up = max(0, -warp_start_height)
    pad_down = max(0, warp_end_height-height_ref)
    pad_left = max(0, -warp_start_width)
    pad_right = max(0, warp_end_width-width_ref)

    igs_merge = np.zeros((height_ref+pad_up+pad_down, width_ref+pad_left+pad_right, 3),dtype=np.uint8)

    y_indices = np.repeat(np.arange(warp_start_height,warp_end_height), warp_end_width-warp_start_width)
    x_indices = np.tile(np.arange(warp_start_width,warp_end_width), warp_end_height-warp_start_height)
    indices = np.concatenate((x_indices.reshape(-1,1),y_indices.reshape(-1,1)),axis=1)

    indices_before_warp = warp_points(indices, H_inv)
    
    for y in range(warp_end_height - warp_start_height):
        for x in range(warp_end_width - warp_start_width):
            index_before_warp = indices_before_warp[y*(warp_end_width-warp_start_width)+x]
            x_before_warp = index_before_warp[0]
            y_before_warp = index_before_warp[1]
            if(0<=y_before_warp<height_in and 0<=x_before_warp<width_in):
                igs_merge[y+max(0,warp_start_height),x+max(0,warp_start_width)] = igs_in[y_before_warp,x_before_warp]

    igs_merge[pad_up:height_ref+pad_up,pad_left:width_ref+pad_left] = igs_ref

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # TODO ...
    height_in, width_in, _ = igs.shape

    H = compute_h_norm(p2, p1)

    H_inv = np.linalg.inv(H)

    igs_rec = np.zeros((400,250,3),dtype=np.uint8)
    # height_rec, width_rec, _ = igs_rec.shape
    # i_indices = np.repeat(np.arange(height_rec), width_rec)
    # j_indices = np.tile(np.arange(width_rec), height_rec)
    # indices = np.concatenate((i_indices.reshape(-1,1), j_indices.reshape(-1,1)),axis=1)

    # indices_before_warp = warp_points(indices, H_inv)
    # for i in range(height_rec):
    #     for j in range(width_rec):
    #         index_before_warp = indices_before_warp[i*width_rec+j]
    #         i_before_warp = index_before_warp[0]
    #         j_before_warp = index_before_warp[1]
    #         if(0<=i_before_warp<height_in and 0<=j_before_warp<width_in):
    #             igs_rec[i,j] = igs[i_before_warp,j_before_warp]

    return warp_image(igs, igs_rec, H)[0]


def set_cor_mosaic():
    # TODO ...
    # (x, y) format
    p_in = np.asarray([[678,668],[691,675],[710,679],[958,655],[1066,692],[1342,826],[1363,573]], dtype=int)
    p_ref = np.asarray([[52,671],[68,679],[92,683],[388,656],[499,693],[744,817],[786,579]], dtype=int)

    return p_in, p_ref

def set_cor_rec():
    # TODO ...
    # (x, y) format
    c_in = np.asarray([[162,16],[259,27],[162,259],[259,245]], dtype=int)
    c_ref = np.asarray([[50,50],[200,50],[50,350],[200,350]], dtype=int)

    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############
    
    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)

    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_merged.png')
  
    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec_output = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec_output.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
