import numpy as np
from utils import draw_save_plane_with_points


if __name__ == "__main__":


    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    #RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    
    num_per_sample = 3
    sample_time = int(np.ceil(-3 / np.log10(1 - (100 / 130) ** num_per_sample)))
    distance_threshold = 0.05

    # sample points group
    data_points = np.concatenate((noise_points, np.ones((noise_points.shape[0], 1))), axis = 1)
    samples = data_points[np.random.choice(noise_points.shape[0], size = (sample_time, num_per_sample), replace = False)]

    # estimate the plane with sampled points group
    _, _, vh = np.linalg.svd(samples)
    planes = vh[:, -1, :]

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    distance = (planes @ data_points.T) ** 2
    inliners = np.where(distance < distance_threshold, 1, 0)
    best_idx = np.argmax(np.sum(inliners, axis = 1), axis = 0)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    inliner_points = data_points[np.where(inliners[best_idx] == 1)]
    _, _, vh = np.linalg.svd(inliner_points)
    pf = vh[-1, :]


    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)

