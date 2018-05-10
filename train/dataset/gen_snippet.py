import numpy as np
import json


def check_size(frame_sz, bbox):
    min_ratio = 0.1
    max_ratio = 0.75
    # accept only objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return not ok


def check_borders(frame_sz, bbox):
    dist_from_border = 0.05 * (bbox[2]-bbox[0] + bbox[3]-bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and ((frame_sz[0]-bbox[2]) > dist_from_border) \
         and ((frame_sz[1]-bbox[3]) > dist_from_border)
    return not ok


# Filter out snippets
vid = json.load(open('vid.json', 'r'))
snippets = []
n_snippets = 0
n_videos = 0
for subset in vid:
    for video in subset:
        n_videos += 1
        frames = video['frame']
        id_frames = [[]] * 60  # at most 60 objects
        id_set = []
        for f, frame in enumerate(frames):
            objs = frame['objs']
            frame_sz = frame['frame_sz']
            for obj in objs:
                trackid = obj['trackid']
                occluded = obj['occ']
                bbox = obj['bbox']
                if occluded:
                    continue

                if check_size(frame_sz, bbox) or check_borders(frame_sz, bbox):
                    continue

                if obj['c'] in ['n01674464', 'n01726692', 'n04468005', 'n02062744']:
                    continue

                if trackid not in id_set:
                    id_set.append(trackid)
                    id_frames[trackid] = []
                id_frames[trackid].append(f)

        for trackid_select in id_set:
            frame_ids = id_frames[trackid_select]
            snippet = dict()
            snippet['base_path'] = video['base_path']
            snippet['frame'] = []
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
                snippet['frame'].append(fram)
            snippets.append(snippet)
            n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

print('save json (snippets), please wait 1 min~')
json.dump(snippets, open('snippet.json', 'w'), indent=2)
print('done!')