from os.path import join, isdir
from os import mkdir
import argparse
import numpy as np
import json
import cv2
from sample import resample
import time

parse = argparse.ArgumentParser(description='Generate training data (cropped) for DCFNet_pytorch')
parse.add_argument('-v', '--visual', dest='visual', action='store_true', help='whether visualise crop')
parse.add_argument('-o', '--output_size', dest='output_size', default=125, type=int, help='crop output size')
parse.add_argument('-p', '--padding', dest='padding', default=2, type=float, help='crop padding size')

args = parse.parse_args()

print args


def checkSize(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # accept only objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return not ok


def checkBorders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2]-bbox[0] + bbox[3]-bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and ((frame_sz[0]-bbox[2]) > dist_from_border) \
         and ((frame_sz[1]-bbox[3]) > dist_from_border)
    return not ok


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index


num_all_frame = 546315  # cat snippet.json | grep bbox |wc -l
num_val = 1000
# crop image
imdb = dict()
is_not_end = np.ones(num_all_frame)  # 1 for training 2 for val 3 not template
imdb['down_index'] = np.zeros(num_all_frame, np.int)  # buff
imdb['up_index'] = np.zeros(num_all_frame, np.int)

crop_base_path = 'crop_{:d}_{:1.1f}'.format(args.output_size, args.padding)
if not isdir(crop_base_path):
    mkdir(crop_base_path)

snaps = json.load(open('snippet.json', 'r'))
count = 0

begin_time = time.time()
for snap in snaps:
    frames = snap['frame']
    n_frames = len(frames)
    for f, frame in enumerate(frames):
        img_path = join(snap['base_path'], frame['img_path'])
        im = cv2.imread(img_path)
        avg_chans = np.mean(im, axis=(0, 1))
        bbox = frame['obj']['bbox']

        target_pos = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
        target_sz = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        window_sz = target_sz * (1 + args.padding)
        crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
        crop = resample(im, crop_bbox, [args.output_size, args.output_size], avg_chans)

        cv2.imwrite(join(crop_base_path, '{:08d}.jpg'.format(count)), np.transpose(crop[:, :, :], (1, 2, 0)))
        # cv2.imwrite('crop.jpg'.format(count), np.transpose(crop[:, :, :], (1, 2, 0)))

        is_not_end[count] = 0 if f == frames else 1
        imdb['down_index'][count] = f
        imdb['up_index'][count] = n_frames - f
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - begin_time
            print("Processed {} images in {:.2f} seconds. "
                  "{:.2f} images/second.".format(count, elapsed, count / elapsed))

template_id = np.where(is_not_end != 0)[0]  # NEVER use the last frame as template! I do not like bidirectional.
rand_split = np.random.choice(len(template_id), len(template_id))
imdb['train_set'] = rand_split[:(len(template_id)-num_val)]
imdb['val_set'] = rand_split[(len(template_id)-num_val):]
print len(imdb['train_set'])
print len(imdb['val_set'])

# to list for json
imdb['train_set'] = imdb['train_set'].tolist()
imdb['val_set'] = imdb['val_set'].tolist()
imdb['down_index'] = imdb['down_index'].tolist()
imdb['up_index'] = imdb['up_index'].tolist()

print('imdb json, please wait 5 seconds~')
json.dump(imdb, open('dataset.json', 'w'), indent=2)
print('done!')
