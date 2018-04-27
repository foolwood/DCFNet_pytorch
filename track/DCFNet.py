from os.path import join, isdir
from os import makedirs
import json
import numpy as np
import torch

import cv2
import time as time
from util import cxy_wh_2_rect1, rect1_2_cxy_wh, gaussian_shaped_labels, cxy_wh_2_bbox
from net import DCFNet
from sample import resample


class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = 'param.pth'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1
    interp_factor = 0.011
    num_scale = 3
    scale_step = 1.015
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    yf_ = np.fft.fft2(gaussian_shaped_labels(output_sigma, net_input_size))
    yf = torch.Tensor(1, 1, crop_sz, crop_sz, 2)
    yf_real = torch.Tensor(np.real(yf_))
    yf_imag = torch.Tensor(np.imag(yf_))
    yf[0, 0, :, :, 0] = yf_real
    yf[0, 0, :, :, 1] = yf_imag
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz)))


def DCFNet_init(im, target_pos, target_sz, use_gpu=True):
    pass


def DCFNet_track(state, im):
    pass


if __name__ == '__main__':
    dataset = 'OTB2015'
    base_path = join('dataset', dataset)
    json_path = join('dataset', dataset + '.json')
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())
    use_gpu = True
    visualization = True
    for video_id, video in enumerate(videos[30:]):  # run without resetting
        video_path_name = annos[video]['name']
        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        image_files = [join('/data1/qwang/OTB100/', video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]
        n_images = len(image_files)

        target_pos, target_sz = rect1_2_cxy_wh(init_rect)

        im = cv2.imread(image_files[0])  # HxWxC
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # init tracker
        config = TrackerConfig()
        net = DCFNet(config)
        net.load_param(config.feature_path)
        net.eval()
        min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        [im_h, im_w, _] = im.shape
        max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = resample(im, bbox, config.net_input_size, [0, 0, 0])
        # crop = np.transpose(patch, (1, 2, 0)).astype(np.float32)
        # cv2.imwrite('crop.jpg', crop)
        target = patch - config.net_average_image
        net.update(torch.Tensor(np.expand_dims(target, axis=0)))

        res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
        tic = time.time()
        patch_crop = np.zeros((3, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
        for f in range(1, n_images):
            im = cv2.imread(image_files[0])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # track
            for i in range(config.num_scale):
                window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                bbox = cxy_wh_2_bbox(target_pos, window_sz)
                patch_crop[i,:] = resample(im, bbox, config.net_input_size, [0, 0, 0])
                # crop = np.transpose(patch_crop[i], (1, 2, 0)).astype(np.float32)
                # cv2.imwrite('crop.jpg', crop)

            search = patch_crop - config.net_average_image
            response = net(torch.Tensor(search))
            response_cpu = response.data.cpu().numpy()
            cv2.imwrite('response_map.jpg', response_cpu[0,0,:])
            peak, idx = torch.max(response.view(config.num_scale, -1), 1)
            peak = peak.data.numpy() * config.scale_factor
            best_scale = np.argmax(peak)
            r_max, c_max = np.unravel_index(idx[best_scale], config.net_input_size)

            if r_max > config.net_input_size[0] / 2:
                r_max = r_max - config.net_input_size[0]
            if c_max > config.net_input_size[1] / 2:
                c_max = c_max - config.net_input_size[1]
            window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))

            target_pos = target_pos + np.array([r_max, c_max]) * window_sz / config.net_input_size
            target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

            # model update
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = resample(im, bbox, config.net_input_size, [0, 0, 0])
            target = patch - config.net_average_image
            net.update(torch.Tensor(np.expand_dims(target, axis=0)), lr=config.interp_factor)

            res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

            if visualization:
                im_show = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.rectangle(im_show, (int(target_pos[1] - target_sz[1] / 2), int(target_pos[0] - target_sz[0] / 2)),
                              (int(target_pos[1] + target_sz[1] / 2), int(target_pos[0] + target_sz[0] / 2)),
                              (0, 255, 0), 3)
                cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(video, im_show)
                cv2.waitKey(1)

        toc = time.time() - tic
        print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.2f}fps'.format(video_id, video, toc, n_images / toc))

        # save result
        test_path = './result/OTB2015/DCFNet_test/'
        if not isdir(test_path): makedirs(test_path)
        result_path = join(test_path, video + '.txt')
        with open(result_path, 'w') as f:
            for x in res:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')