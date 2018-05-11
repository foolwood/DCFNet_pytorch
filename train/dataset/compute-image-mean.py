import argparse
import numpy as np
import os
import time
import glob

from skimage import io
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meanPrefix', default='mean_img', type=str, help="Prefix of the mean file.")
    parser.add_argument('--imageDir', default='crop_125_2.0', type=str, help="Directory of images to read.")
    args = parser.parse_args()

    mean = np.zeros((1, 3, 125, 125))
    N = 0
    opencv_backend = True
    beginTime = time.time()
    files = glob.glob(os.path.join(args.imageDir, '*.jpg'))
    for file in files:
        if opencv_backend:
            img = cv2.imread(file)
        else:
            img = io.imread(file)
        if img.shape == (125, 125, 3):
            mean[0][0] += img[:, :, 0]
            mean[0][1] += img[:, :, 1]
            mean[0][2] += img[:, :, 2]
            N += 1
            if N % 1000 == 0:
                elapsed = time.time() - beginTime
                print("Processed {} images in {:.2f} seconds. "
                      "{:.2f} images/second.".format(N, elapsed, N / elapsed))
    mean[0] /= N

    meanImg = np.transpose(mean[0].astype(np.uint8), (1, 2, 0))
    if opencv_backend:
        cv2.imwrite("{}.png".format(args.meanPrefix), meanImg)
    else:
        io.imsave("{}.png".format(args.meanPrefix), meanImg)

    avg_chans = np.mean(meanImg, axis=(0, 1))
    if opencv_backend:
        print("image BGR mean: {}".format(avg_chans))
    else:
        print("image RGB mean: {}".format(avg_chans))