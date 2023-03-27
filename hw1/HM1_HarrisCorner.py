import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.

    #compute gradients
    I_x, I_y = Sobel_filter_y(input_img), Sobel_filter_x(input_img)
    I_xx, I_yy, I_xy = I_x * I_x, I_y * I_y, I_x * I_y

    #compute gaussian_window_function of the given size
    sigma = 1.0
    half_window_size = window_size // 2
    x, y = np.mgrid[-half_window_size: half_window_size + 1, -half_window_size: half_window_size + 1]
    gaussian_window_function = np.exp((-(x ** 2 + y ** 2)) / 2 * sigma ** 2)

    #compute matrix M = I * w (Convolution)
    wI_xx, wI_yy, wI_xy = convolve(padding(I_xx, half_window_size, 'replicatePadding'), gaussian_window_function), \
                          convolve(padding(I_yy, half_window_size, 'replicatePadding'), gaussian_window_function), \
                          convolve(padding(I_xy, half_window_size, 'replicatePadding'), gaussian_window_function),
    
    #compute eignevalues, determinants, traces
    delta = (wI_xx + wI_yy) * (wI_xx + wI_yy) - 4 * (wI_xx * wI_yy - wI_xy * wI_xy)
    eigenvalue1, eigenvalue2 = (wI_xx + wI_yy + np.sqrt(delta)) / 2, (wI_xx + wI_yy - np.sqrt(delta)) / 2
    det = eigenvalue1 * eigenvalue2
    trace = eigenvalue1 + eigenvalue2
    
    #TODO: deal with complex eigenvalues
    #det = np.where(delta >= 0, eigenvalue1 * eigenvalue2, wI_xx * wI_yy - wI_xy * wI_xy)
    #trace = np.where(delta >= 0, eigenvalue1 + eigenvalue2, wI_xx + wI_yy)

    #compute corner response theta
    theta = det - alpha * trace * trace

    #dropout and construct output
    x_corner, y_corner = np.where(theta > threshold)
    corner_list = list(zip(x_corner, y_corner, theta[x_corner, y_corner]))

    return corner_list # the corners in corne_list: a tuple of (index of rows, index of cols, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 0.3

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
