import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
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
    magnitude_grad = np.sqrt(x_grad * x_grad + y_grad * y_grad)
    direction_grad = np.argtan2(y_grad, x_grad)

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
        up = np.floor(x)
        down = up + 1
        left = np.floor(y)
        right = left + 1

        grad_mag

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
    low_ratio = 0.10
    high_ratio = 0.30
    
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
