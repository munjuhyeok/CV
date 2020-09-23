from utils import *
import numpy as np
import time

def get_pixel_at(pixel_grid, i, j):
    '''
    Get pixel values at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.

    Returns:
        ndarray: 1D numpy array representing RGB values.
    '''
    m, n, _ = pixel_grid.shape
    if(i in range(m) and j in range(n)):
        return pixel_grid[i,j]
    else:
        return np.zeros(3,dtype=np.uint8)

def get_patch_at(pixel_grid, i, j, size):
    '''
    Get an image patch at row i and column j.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        i (int): Row index.
        j (int): Column index.
        size (int): Patch size.

    Returns:
        ndarray: 3D numpy array representing an image patch.
    '''
    
    half_size = int(size/2)

    result = np.zeros((size,size,3),dtype=np.uint8)

    
    grid_row_size, grid_column_size, _ = pixel_grid.shape
    left_half= i - max(0,i-half_size)
    right_half = min(grid_row_size,i-half_size+size) - i
    down_half = j - max(0,j-half_size)
    up_half = min(grid_column_size,j-half_size+size) - j

    result[
        half_size-left_half:half_size+right_half,
        half_size-down_half:half_size+up_half,:] = \
    pixel_grid[
        i-left_half:i+right_half,
        j-down_half:j+up_half,:]

    return result



def gaussian_filter_1D(size):
    '''
    get an gaussian filter of size 'size'
    (approximate gaussian distibution as binomial distibution)

    Args:
        size (int): Kernel size.

    Returns:
        ndarray: gaussian filter of size 'size'
    '''
    filters=[np.asarray([1.0])]
    for i in range(size-1):
        new_filter = np.zeros(i+2)
        new_filter[:i+1] += filters[i]
        new_filter[1:] += filters[i]
        filters.append(new_filter)

    return filters[size-1]/np.sum(filters[size-1])

def shift_pixel(pixel_grid, axis, n):
    '''
    shift n pixel along axis

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        axis(int): axis to shift along
        n(int): how many pixel to shift

    Returns:
        ndarray: 3D numpy array representing an RGB image after shifting.
    '''
    after_shift = np.zeros_like(pixel_grid)
    x, y, _ = pixel_grid.shape
    if axis == 0:
        after_shift[max(0,-n):x-n,:,:] = pixel_grid[max(0,+n):x+n, :, :]
    elif (axis ==1):
        after_shift[:,max(0,-n):y-n,:] = pixel_grid[:,max(0,+n):y+n, :]
    
    return after_shift


def apply_gaussian_filter(pixel_grid, size):
    '''
    Apply gaussian filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''
    
    filter = gaussian_filter_1D(size)

    #time complexity is size^2
    '''
    after_filter = np.zeros_like(pixel_grid)
    for i in range(pixel_grid.shape[0]):
        for j in range(pixel_grid.shape[1]):
            temp = get_patch_at(pixel_grid,i,j,size)
            temp = filter.dot(temp)
            temp = filter.dot(temp)
            after_filter[i,j] = temp
    '''
    
    #time complexity is 2*size
    after_filter = np.asarray([shift_pixel(pixel_grid, 0, i) for i in range(-int(size/2),size-int(size/2))])
    after_filter = np.tensordot(filter, after_filter, axes = (0, 0))
    after_filter = np.asarray([shift_pixel(after_filter, 1, i) for i in range(-int(size/2),size-int(size/2))])
    after_filter = np.tensordot(filter, after_filter, axes = (0, 0))
    after_filter = after_filter.astype(np.uint8)
    

    return after_filter

def apply_median_filter(pixel_grid, size):
    '''
    Apply median filter for every pixel in pixel_grid, and return the
    resulting pixel grid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.

    Returns:
        ndarray: 3D numpy array representing an RGB image after filtering.
    '''

    #time complexity is size^2
    '''
    after_filter = np.zeros_like(pixel_grid)
    for i in range(pixel_grid.shape[0]):
        for j in range(pixel_grid.shape[1]):
            temp = get_patch_at(pixel_grid,i,j,size)
            after_filter[i,j] = np.median(temp, axis=(0,1))
    '''
    #time complexity is 2*size
    
    pixel_grid = [shift_pixel(pixel_grid, 0, i) for i in range(-int(size/2),size-int(size/2))]
    after_filter=[]
    for i in range(-int(size/2),size-int(size/2)):
        for j in pixel_grid:
            after_filter.append(shift_pixel(j, 1, i))
    after_filter = np.asarray(after_filter)
    after_filter = np.median(after_filter, axis=0)
    after_filter = after_filter.astype(np.uint8)
    
    return after_filter

def pyrdown(ndarray, size):
    return downsample(apply_gaussian_filter(ndarray, size))


def build_gaussian_pyramid(pixel_grid, size, levels=5):
    '''
    Build and return a Gaussian pyramid.

    Args:
        pixel_grid (ndarray): 3D numpy array representing an RGB image.
        size (int): Kernel size.
        levels (int): Number of levels.

    Returns:
        list of ndarray: List of 3D numpy arrays representing Gaussian
        pyramid.
    '''
    result = [pixel_grid]
    for i in range(levels-1):
        result.append(pyrdown(result[i],size))
    
    return result

def build_laplacian_pyramid(gaussian_pyramid):
    '''
    Build and return a Laplacian pyramid.

    Args:
        gaussian_pyramid (list of ndarray): Gaussian pyramid. 

    Returns:
        list of ndarray: List of 3D numpy arrays representing Laplacian
        pyramid
    '''
    result=[]
    
    levels = len(gaussian_pyramid)-1
    for i in range(levels):
        term1 = gaussian_pyramid[i].copy().astype(int)
        term2 = upsample(gaussian_pyramid[i+1])
        x1,y1,_ = term1.shape
        x2,y2,_ = term2.shape
        x=min(x1,x2)
        y=min(y1,y2)
        term1[:x,:y,:] -= term2[:x,:y,:] #in case shape of term1 and term 2 are different
        result.append(term1) #dtype is int
    result.append(gaussian_pyramid[levels]) 

    return result

def blend_images(left_image, right_image):
    '''
    Smoothly blend two images by concatenation.
    
    Tip: This function should build Laplacian pyramids for both images,
    concatenate left half of left_image and right half of right_image
    on all levels, then start reconstructing from the smallest one.

    Args:
        left_image (ndarray): 3D numpy array representing an RGB image.
        right_image (ndarray): 3D numpy array representing an RGB image.

    Returns:
        ndarray: 3D numpy array representing an RGB image after blending.
    '''
    size=3
    levels=5


    lp_l = build_laplacian_pyramid(build_gaussian_pyramid(left_image,size,levels))
    lp_r = build_laplacian_pyramid(build_gaussian_pyramid(right_image,size,levels))
    #lp_concat = [concat(left,right) for (left, right) in zip(lp_l,lp_r)]

    label = concat(np.ones_like(left_image), np.zeros_like(left_image))
    gp_label = build_gaussian_pyramid(label,size)
    lp_combined = [label*left+(1-label)*right for (left, right, label) in zip(lp_l, lp_r, gp_label)]

    result = lp_combined[levels-1]
    for i in range(levels-1,0,-1):
        term1 = lp_combined[i-1]
        term2 = upsample(result)
        x1,y1,_ = term1.shape
        x2,y2,_ = term2.shape
        x = min(x1,x2)
        y = min(y1,y2)
        result = term1.astype(int)
        result[:x,:y,:] += term2[:x,:y,:]
        result = np.minimum(result,255) #prevent overflow
        result = np.maximum(result, 0) #prevent underflow
        result = result.astype(np.uint8)

    return result
    
    




if __name__ == "__main__":
    ### Test Gaussian Filter ###
    dog_gaussian_noise = load_image('./images/dog_gaussian_noise.png')
    after_filter = apply_gaussian_filter(dog_gaussian_noise,5)
    save_image(after_filter, './dog_gaussian_noise_after.png')

    ### Test Median Filter ###
    dog_salt_and_pepper = load_image('./images/dog_salt_and_pepper.png')
    after_filter = apply_median_filter(dog_salt_and_pepper,5)
    save_image(after_filter, './dog_salt_and_pepper_after.png')

    
    ### Test Image Blending ###
    
    player1 = load_image('./images/player1.png')
    player2 = load_image('./images/player2.png')
    
    gp = build_gaussian_pyramid(player1,5)
    lp = build_laplacian_pyramid(gp)
    
    
    after_blending = blend_images(player1, player2)
    
    save_image(after_blending, './player3.png')

    # Simple concatenation for comparison.
    save_image(concat(player1, player2), './player_simple_concat.png')

    player1 = load_image('./images/player1_256x256.png')
    player2 = load_image('./images/player2_256x256.png')
    gp = build_gaussian_pyramid(player1,5)
    lp = build_laplacian_pyramid(gp)
    after_blending = blend_images(player1, player2)
    save_image(after_blending, './player4.png')

    girl1 = load_image('./images/girl1.png')
    girl2 = load_image('./images/girl2.png')
    gp = build_gaussian_pyramid(player1,1)
    lp = build_laplacian_pyramid(gp)
    after_blending = blend_images(girl1, girl2)
    save_image(after_blending, './girl3.png')