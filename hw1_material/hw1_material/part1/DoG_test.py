import numpy as np
import cv2


threshold = 3
sigma = 2**(1/4)
image = cv2.imread('./testdata/1.png', 0).astype(np.float32)

       
gaussian_images_1st = [cv2.GaussianBlur (image, (0, 0), sigma**n) for n in range(1,5)]
y,x = image.shape
image_2st = cv2.resize(gaussian_images_1st[3], (int(x/2), int(y/2)), interpolation = cv2.INTER_NEAREST)
gaussian_images_2st = [cv2.GaussianBlur (image_2st, (0, 0), sigma**n) for n in range(1,5)]


gaussian_images_1st = [image] +gaussian_images_1st
DoG1 = [cv2.subtract(gaussian_images_1st[n+1] , gaussian_images_1st[n]) for n in range(4)]

gaussian_images_2st = [image_2st] +gaussian_images_2st
DoG2 = [cv2.subtract(gaussian_images_2st[n+1] , gaussian_images_2st[n]) for n in range(4)]
for n in range(4):
    cv2.imwrite("DoG1-" + str(n+1) + ".png", DoG1[n].astype(np.uint8))
    cv2.imwrite("DoG2-" + str(n+1) + ".png", DoG2[n].astype(np.uint8))


# Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
# - Function: cv2.subtract(second_image, first_image)
#DoG1 = [(DoG-DoG.min())/((DoG-DoG.min()).max())*255  for DoG in DoG1]
#DoG2 = [(DoG-DoG.min())/((DoG-DoG.min()).max())*255  for DoG in DoG2]
nb_img1 = np.array(DoG1[:3])
nb_img2 = np.array(DoG1[1:])
nb_img3 = np.array(DoG2[:3])
nb_img4 = np.array(DoG2[1:])

def get_keyp(nb_img):
    windows = np.lib.stride_tricks.sliding_window_view(nb_img,(3,3,3))[0]
    keypoints = []
    for y in range(windows.shape[0]):
        for x in range(windows.shape[1]):
            window = windows[y, x]
            center = window[ 1, 1, 1]
            if abs(center)<threshold:
                continue
            #elif (window >= center).all() or (window <= center).all():
            elif (window.max() == center) or (window.min() == center):
                keypoints.append([y+1,x+1])
    return keypoints
keypoints1 = get_keyp(nb_img1)
keypoints2 = get_keyp(nb_img2)
keypoints3 = get_keyp(nb_img3)*2
keypoints4 = get_keyp(nb_img4)*2

print(len(keypoints1))
print(len(keypoints2))
print(len(keypoints3))
print(len(keypoints4))
# Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
#         Keep local extremum as a keypoint

# Step 4: Delete duplicate keypoints
# - Function: np.unique
# sort 2d-point by y, then by x

#keypoints = np.unique(keypoints.view('c8')).view('i4').reshape((-1,2))
keypoints = np.unique(np.vstack(keypoints1 + keypoints2 + keypoints3 + keypoints4), axis=0)
#keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
print(len(keypoints))
d=[]
for k in keypoints2:
    d.append(k in b)
