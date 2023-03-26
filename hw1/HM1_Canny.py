import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, padding
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    #compute
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    direction_grad = np.arctan2(x_grad, y_grad)

    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """
    #compute step size
    x_step = np.cos(grad_dir) * grad_mag
    y_step = np.sin(grad_dir) * grad_mag

    #compute positions of neighbours
    nrow, ncol = grad_mag.shape
    x_postive_neighbour = np.arange(nrow)[:, np.newaxis] + x_step
    y_postive_neighbour = np.arange(ncol) + y_step
    x_negative_neighbour = np.arange(nrow)[:, np.newaxis] - x_step
    y_neagtive_neighbour = np.arange(ncol) - y_step

    def bilinear_interpolation(x, y):
        #compute raw border
        raw_up = np.floor(x).astype(int)
        raw_down = (raw_up + 1).astype(int)
        raw_left = np.floor(y).astype(int)
        raw_right = (raw_left + 1).astype(int)

        #compute borders normalized into the img shape
        up = np.where(raw_up >= nrow, nrow - 1, np.where(raw_up < 0, 0, raw_up)).astype(int)
        down = np.where(raw_down >= nrow, nrow - 1, np.where(raw_down < 0, 0, raw_down)).astype(int)
        left = np.where(raw_left >= ncol, ncol - 1, np.where(raw_left < 0, 0, raw_left)).astype(int)
        right = np.where(raw_right >= ncol, ncol - 1, np.where(raw_right < 0, 0, raw_right)).astype(int)

        #compute output
        output = (down - x) * ((y - left) * grad_mag[up, right] + (right - y) * grad_mag[up, left]) + \
                 (x - up) * ((y - left) * grad_mag[down, right] + (right - y) * grad_mag[down, left])
        
        #dropout the points whose neighbours cannot be reached
        output = np.where((raw_up >= 0) & (raw_down < nrow) & (raw_left >= 0) & (raw_right < ncol), output, 0)

        return output
    
    #suppression
    NMS_output = np.where((bilinear_interpolation(x_postive_neighbour, y_postive_neighbour) <= grad_mag) &
                          (bilinear_interpolation(x_negative_neighbour, y_neagtive_neighbour) <= grad_mag),
                          grad_mag, 0)

    return NMS_output 


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.8
    high_ratio = 1.0

    #compute threshold
    print(np.mean(img))
    mean = np.sum(img) / np.sum(img != 0)
    low_threshold = low_ratio * mean
    high_threshold = high_ratio * mean

    #compute size
    nrow, ncol = img.shape

    #dropout and reserve
    #print(img)
    img = np.where(img <= low_threshold, 0, img)
    #print(img)
    output = np.where(img >= high_threshold, 1, 0)

    """
    Switched to the strategy stated in 
    https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    from that in slides because of the lack of direction parameter.
    """
    #grow edges
    updated = True
    while updated:
        updated = False

        #compute dirs
        dirs = [[-1, 0], [-1, 1], [0, 1], [1, 1],
                [1, 0], [1, -1], [0, -1], [-1, -1]]
        
        #update
        for i in range(nrow):
            for j in range(ncol):
                if output[i, j] or not img[i, j]:
                    continue
                for i_dir, j_dir in dirs:
                    i_neighbour, j_neighbour = i + i_dir, j + j_dir
                    if i_neighbour < 0 or i_neighbour >= nrow or j_neighbour < 0 or j_neighbour >= ncol:
                        continue
                    if output[i_neighbour, j_neighbour]:
                        output[i, j] = 1
                        updated = True
                        break
    
    #update stable weak points to 0
    output = np.where((img > 0) & np.logical_not(output), 0, output)
    
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
