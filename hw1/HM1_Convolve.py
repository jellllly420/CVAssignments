import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    nrow, ncol = img.shape
    padding_img = np.zeros((nrow + padding_size * 2, ncol + padding_size * 2))

    if type=="zeroPadding":
        padding_img[padding_size: -padding_size, padding_size: -padding_size] = img

        return padding_img
    elif type=="replicatePadding":
        padding_img[padding_size: -padding_size, padding_size: -padding_size] = img

        #border
        padding_img[:padding_size, padding_size: -padding_size] = img[0]
        padding_img[-padding_size:, padding_size: -padding_size] = img[-1]
        padding_img[padding_size: -padding_size, :padding_size] = img[:, 0][:, np.newaxis]
        padding_img[padding_size: -padding_size, -padding_size:] = img[:, -1][:, np.newaxis]

        #corner
        padding_img[:padding_size, :padding_size] = img[0, 0]
        padding_img[:padding_size, -padding_size:] = img[0, -1]
        padding_img[-padding_size:, :padding_size] = img[-1, 0]
        padding_img[-padding_size:, -padding_size:] = img[-1, -1]

        return padding_img

def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    #no expicit padding

    #build the Toeplitz matrix and compute convolution
    #compute sizes
    nrow_i, ncol_i = img.shape
    nrow_k, ncol_k = kernel.shape
    nrow_o, ncol_o = nrow_i + nrow_k - 1, ncol_i + ncol_k - 1
    #flip and zero-padding kernel
    flipped_kernel = kernel.reshape(nrow_k * ncol_k)[::-1].reshape((nrow_k, ncol_k))
    padding_kernel = np.zeros((nrow_o, ncol_o))
    padding_kernel[:nrow_k, :ncol_k] = flipped_kernel

    #build Toeplitz blocks
    tmp = np.zeros((ncol_o + 1, ncol_o))
    tmp[:nrow_k, :ncol_k] = np.eye(ncol_k)
    tmp = np.concatenate((tmp,) * ((ncol_o * ncol_i) // (ncol_o + 1)) + (tmp[:ncol_k],))
    blocks = np.hsplit((tmp @ padding_kernel.T).T.reshape((ncol_i * nrow_o, ncol_o)).T, nrow_o)

    #build Toeplitz matrix (only for 6x6 input)
    zeros = np.zeros((ncol_o, ncol_i))
    toeplitz = np.concatenate((np.concatenate((zeros,) * 0 + tuple(blocks[:])),
                               np.concatenate((zeros,) * 1 + tuple(blocks[: -1])),
                               np.concatenate((zeros,) * 2 + tuple(blocks[: -2])),
                               np.concatenate((zeros,) * 3 + tuple(blocks[: -3])),
                               np.concatenate((zeros,) * 4 + tuple(blocks[: -4])),
                               np.concatenate((zeros,) * 5 + tuple(blocks[: -5])),
                               ), axis = 1)
    
    #convolve (not for 2x2 kernel)
    output = (toeplitz @ img.flatten()).reshape((nrow_o, ncol_o))[1:-1, 1:-1]
    
    return output

def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    #compute sizes
    nrow_i, ncol_i = img.shape
    nrow_k, ncol_k = kernel.shape
    nrow_o, ncol_o = nrow_i - nrow_k + 1, ncol_i - ncol_k + 1

    #compute indices
    x = np.repeat(np.arange(nrow_o), ncol_o)
    x_k = np.repeat(np.arange(nrow_k), ncol_k)
    x = x[:, np.newaxis] + x_k
    y = np.tile(np.arange(nrow_o), ncol_o)
    y_k = np.tile(np.arange(nrow_k), ncol_k)
    y = y[:, np.newaxis] + y_k
    
    #convolve
    matrix = img[x,y]
    flattened_kernel = kernel.flatten()
    output = (matrix @ flattened_kernel).reshape((nrow_o, ncol_o))

    return output

def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)
