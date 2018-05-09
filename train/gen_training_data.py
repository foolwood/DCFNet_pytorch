from os.path import join, isfile
from os import listdir
import argparse
import numpy as np
from random import randrange
import json
import glob
import xml.etree.ElementTree as ET
import cv2
from sample import resample
import cPickle as pickle

parse = argparse.ArgumentParser(description='Generate training data (cropped) for DCFNet_pytorch')
parse.add_argument('-v', '--visual', dest='visual', action='store_true', help='whether visualise crop')
parse.add_argument('-o', '--output_size', dest='output_size', default=125, type=int, help='crop output size')
parse.add_argument('-p', '--padding', dest='padding', default=1.5, type=float, help='crop padding size')

args = parse.parse_args()

print args

set_name = {'vid2015'}
num_all_frame = 547560
num_val = 1000


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


if 'vid2015' in set_name:
    print('VID2015 Data:')
    VID_base_path = '/data1/qwang/ILSVRC2015/'
    ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
    img_base_path = join(VID_base_path, 'Data/VID/train/')
    sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
    imbd =[]
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):
            print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
            v = dict()
            v['base_path'] = join(img_base_path, sub_set, video)
            v['frame'] = []
            video_base_path = join(sub_set_base_path, video)
            xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
            for xml in xmls:
                fram = dict()
                xmltree = ET.parse(xml)
                size = xmltree.findall('size')[0]
                frame_sz = [int(it.text) for it in size]
                objects = xmltree.findall('object')
                objs = []
                for object_iter in objects:
                    trackid = int(object_iter.find('trackid').text)
                    name = (object_iter.find('name')).text
                    bndbox = object_iter.find('bndbox')
                    occluded = int(object_iter.find('occluded').text)
                    o = dict()
                    o['c'] = name
                    o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                    o['trackid'] = trackid
                    o['occ'] = occluded
                    objs.append(o)
                    # [xmin, ymin, xmax, ymax] = [it.text for it in bndbox]
                fram['frame_sz'] = frame_sz
                fram['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                fram['objs'] = objs
                v['frame'].append(fram)
            s.append(v)
        imbd.append(s)
    print('save json, please wait 1 min~')
    json.dump(imbd, open('imdb.json', 'w'), indent=2)
    print('done!')

    # Filter out snippets
    imbd = json.load(open('imdb.json', 'r'))
    snaps = []
    n_snaps = 0
    n_videos = 0
    for subset in imbd:
        for video in subset:
            n_videos += 1
            frames = video['frame']
            id_frames = [[]] * 60
            id_set = []
            for f, frame in enumerate(frames):
                objs = frame['objs']
                frame_sz = frame['frame_sz']
                for obj in objs:
                    trackid = obj['trackid']
                    occluded = obj['occ']
                    bbox = obj['bbox']
                    if occluded:  # remove occluded objects from 2,005,418 -> 912,976
                        continue

                    if checkSize(frame_sz, bbox) or checkBorders(frame_sz, bbox):  # remove near boarder, too small and too large objects from 912,976 -> 634,825
                        continue

                    if obj['c'] in ['n01674464', 'n01726692', 'n04468005', 'n02062744']:  # 634,825 -> 578217
                        continue

                    if trackid not in id_set:
                        id_set.append(trackid)
                        id_frames[trackid] = []
                    id_frames[trackid].append(f)

            for trackid_select in id_set:
                frame_ids = id_frames[trackid_select]
                snap = dict()
                snap['base_path'] = video['base_path']
                snap['frame'] = []
                for f in frame_ids:
                    frame = frames[f]
                    fram = dict()
                    fram['frame_sz'] = frame['frame_sz']
                    fram['img_path'] = frame['img_path']
                    objs = frame['objs']
                    for obj in objs:
                        trackid = obj['trackid']
                        if trackid == trackid_select:
                            ob = obj
                            continue
                    fram['objs'] = ob
                    snap['frame'].append(fram)
                snaps.append(snap)
                n_snaps += 1
            print('video: {:d} snaps_num: {:d}'.format(n_videos, n_snaps))

    print('save json, please wait 1 min~')
    json.dump(snaps, open('snippet.json', 'w'), indent=2)
    print('done!')

    # crop image
    def cxy_wh_2_bbox(cxy, wh):
        return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index

    imdb = dict()
    rand_split = np.random.choice(num_all_frame, num_all_frame)
    imdb['train_set'] = rand_split[:(num_all_frame-num_val)]
    imdb['val_set'] = rand_split[(num_all_frame-num_val):]
    imdb['images_id'] = range(num_all_frame)

    imdb['down_index'] = np.zeros(num_all_frame, np.uint16)
    imdb['up_index'] = np.zeros(num_all_frame, np.uint16)

    snaps = json.load(open('snippet.json', 'r'))
    count = 0
    for snap in snaps:
        frames = snap['frame']
        n_frames = len(frames)
        for f, frame in enumerate(frames):
            img_path = join(snap['base_path'], frame['img_path'])
            im = cv2.imread(img_path)
            avg_chans = np.mean(im, axis=(0, 1))
            bbox = frame['objs']['bbox']

            target_pos = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
            target_sz = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
            window_sz = target_sz * (1 + args.padding)
            crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
            crop = resample(im, crop_bbox, [args.output_size, args.output_size], avg_chans)

            imdb['down_index'][count] = f
            imdb['up_index'][count] = n_frames-f
            cv2.imwrite('./crop/{:08d}.jpg'.format(count), np.transpose(crop[:, :, :], (1, 2, 0)))
            # cv2.imwrite('crop.jpg'.format(count), np.transpose(crop[:, :, :], (1, 2, 0)))
            count += 1
            print count

    with open('imdb.pkl','w') as f:
        pickle.dump(imdb, f, pickle.HIGHEST_PROTOCOL)
