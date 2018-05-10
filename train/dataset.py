import torch.utils.data as data
from os.path import join
import cv2
import json
import numpy as np


class VID(data.Dataset):
    def __init__(self, file='dataset/dataset.json', root='dataset/crop_125_1.5', train=True):
        self.imdb = json.load(open(file, 'r'))
        self.root = root
        self.train = train

    def __getitem__(self, item):
        if self.train:
            target_id = self.imdb['train_set'][item]
        else:
            target_id = self.imdb['val_set'][item]

        range_down = self.imdb['down_index'][target_id]
        range_up = self.imdb['up_index'][target_id]
        search_id = np.random.randint(-range_down, range_up) + target_id

        target = cv2.imread(join(self.root, '{:08d}.jpg'.format(target_id)))
        search = cv2.imread(join(self.root, '{:08d}.jpg'.format(search_id)))

        target = np.transpose(target, (2, 0, 1)).astype(np.float32)
        search = np.transpose(search, (2, 0, 1)).astype(np.float32)

        return target, search

    def __len__(self):
        if self.train:
            return len(self.imdb['train_set'])
        else:
            return len(self.imdb['val_set'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = VID(train=True)
    n = len(data)
    for i in range(n):
        z, x = data[i]
        z, x = np.transpose(z, (1, 2, 0)).astype(np.uint8), np.transpose(x, (1, 2, 0)).astype(np.uint8)
        zx = np.concatenate((z, x), axis=1)
        plt.imshow(cv2.cvtColor(zx, cv2.COLOR_BGR2RGB))
        plt.show()
