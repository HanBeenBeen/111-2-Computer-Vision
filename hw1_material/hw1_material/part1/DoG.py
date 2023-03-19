import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
       
        gaussian_images_1st = [cv2.GaussianBlur (image, (0, 0), self.sigma**n) for n in range(1,5)]
        y,x = image.shape
        image_2st = cv2.resize(image, (int(x/2), int(y/2)))
        gaussian_images_2st = [cv2.GaussianBlur (image_2st, (0, 0), self.sigma**n) for n in range(1,5)]

        DoG1 = []
        DoG1.append(image - gaussian_images_1st[0])
        DoG1 = DoG1 + [gaussian_images_1st[n+1] - gaussian_images_1st[n] for n in range(3)]

        DoG2 = []
        DoG2.append(image_2st - gaussian_images_2st[0])
        DoG2 = DoG2 + [gaussian_images_2st[n+1] - gaussian_images_2st[n] for n in range(3)]
        np.dstack([a,a])
        for n in range(4):
            cv2.imwrite("DoG1-" + str(n+1) + ".png", DoG1[n])
            cv2.imwrite("DoG2-" + str(n+1) + ".png", DoG2[n])


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique


        # sort 2d-point by y, then by x

        #keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        #return keypoints
