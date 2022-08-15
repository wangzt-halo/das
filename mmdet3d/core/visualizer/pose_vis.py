import cv2
import matplotlib.pyplot as plt
import numpy as np

# cmu panoptic
LIMBS15 = [[0, 1],
           [0, 2],
           [0, 3],
           [3, 4],
           [4, 5],
           [0, 9],
           [9, 10],
           [10, 11],
           [2, 6],
           [2, 12],
           [6, 7],
           [7, 8],
           [12, 13],
           [13, 14]]
# coco
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
           [11, 13], [13, 15],
           [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]
LIMBS21 = (
    (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13),
    (13, 20),
    (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))


def pixel2world(x, K, R, t):
    X = x.copy()
    X[0, :] = X[0, :] - K[0, 2]
    X[1, :] = X[1, :] - K[1, 2]
    X[:2] = np.dot(np.linalg.inv(K[:2, :2]), X[:2])
    x1 = X.copy()
    X[0:2, :] = X[0:2, :] * X[2, :]
    x2 = X.copy()
    X = np.dot(np.linalg.inv(R), (X - t))
    x3 = X.copy()
    return x1, x2, x3


def get_limbs(num_joints):
    return eval('LIMBS%d' % num_joints)


# colormap for joints (right->red, left->blue)
COLOR15 = lambda x: (0, 0, 255) if x >= 9 else (255, 0, 0)
COLOR17 = lambda x: (0, 0, 255) if x % 2 == 0 else (255, 0, 0)
COLOR21 = lambda x: (0, 0, 255) if x not in [5, 6, 7, 11, 12, 13, 18, 20] else (255, 0, 0)


def get_colors(num_joints):
    return eval('COLOR%d' % num_joints)


def vis_keypoints(img, kps, num_joints, kp_thresh=0.4, alpha=1):
    kps_lines = get_limbs(num_joints)
    lr_color = get_colors(num_joints)
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    # Draw the keypoints.
    for link in range(len(kps_lines)):
        i1 = kps_lines[link][0]
        i2 = kps_lines[link][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[link], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=lr_color(i1), thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=lr_color(i2), thickness=-1, lineType=cv2.LINE_AA)
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
