import argparse
import cv2
import numpy as np
from os import makedirs
from os.path import isfile, isdir, join
from util import cxy_wh_2_rect1
import torch
import json
from DCFNet import *

parser = argparse.ArgumentParser(description='Tune parameters for DCFNet tracker on OTB2015')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')

args = parser.parse_args()


def tune_otb(param):
    regions = []  # result and states[1 init / 2 lost / 0 skip]
    # save result
    benchmark_result_path = join('result', param['dataset'])
    tracker_path = join(benchmark_result_path, (param['network_name'] +
                        '_scale_step_{:.3f}'.format(param['config'].scale_step) +
                        '_scale_penalty_{:.3f}'.format(param['config'].scale_penalty) +
                        '_interp_factor_{:.3f}'.format(param['config'].interp_factor)))
    result_path = join(tracker_path, '{:s}.txt'.format(param['video']))
    if isfile(result_path):
        return
    if not isdir(tracker_path): makedirs(tracker_path)
    with open(result_path, 'w') as f:  # Occupation
        for x in regions:
            f.write('')

    ims = param['ims']
    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            init_rect = p['init_rect']
            tracker = DCFNetTraker(ims[f], init_rect, config=param['config'])
            regions.append(init_rect)
        else:  # tracking
            rect = tracker.track(ims[f])
            regions.append(rect)
        toc += cv2.getTickCount() - tic

        if args.visualization:  # visualization (skip lost frame)
            if f == 0: cv2.destroyAllWindows()
            location = [int(l) for l in location]  # int
            cv2.rectangle(im, (location[0], location[1]), (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 3)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video, im)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    print('{:2d} Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(v, video, toc, f / toc))
    regions = np.array(regions)
    regions[:,:2] += 1  # 1-index
    with open(result_path, 'w') as f:
        for x in regions:
            f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')


params = {'dataset':['OTB2013'], 'network':['param.pth'],
          'scale_step':np.arange(1.01, 1.05, 0.005, np.float32),
          'scale_penalty':np.arange(0.98, 1.0, 0.025, np.float32),
          'interp_factor':np.arange(0.001, 0.015, 0.001, np.float32)}

p = dict()
p['config'] = TrackerConfig()
for network in params['network']:
    p['network_name'] = network
    np.random.shuffle(params['dataset'])
    for dataset in params['dataset']:
        base_path = join('dataset', dataset)
        json_path = join('dataset', dataset+'.json')
        annos = json.load(open(json_path, 'r'))
        videos = annos.keys()
        p['dataset'] = dataset
        np.random.shuffle(videos)
        for v, video in enumerate(videos):
            p['v'] = v
            p['video'] = video
            video_path_name = annos[video]['name']
            init_rect = np.array(annos[video]['init_rect']).astype(np.float)
            image_files = [join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]
            target_pos = np.array([init_rect[0] + init_rect[2] / 2 -1 , init_rect[1] + init_rect[3] / 2 -1])  # 0-index
            target_sz = np.array([init_rect[2], init_rect[3]])
            ims = []
            for image_file in image_files:
                im = cv2.imread(image_file)
                if im.shape[2] == 1:
                    cv2.cvtColor(im, im, cv2.COLOR_GRAY2RGB)
                ims.append(im)
            p['ims'] = ims
            p['init_rect'] = init_rect

            np.random.shuffle(params['scale_step'])
            np.random.shuffle(params['scale_penalty'])
            np.random.shuffle(params['interp_factor'])
            for scale_step in params['scale_step']:
                for scale_penalty in params['scale_penalty']:
                    for interp_factor in params['interp_factor']:
                        p['config'].scale_step = float(scale_step)
                        p['config'].scale_penalty = float(scale_penalty)
                        p['config'].interp_factor = float(interp_factor)
                        tune_otb(p)
