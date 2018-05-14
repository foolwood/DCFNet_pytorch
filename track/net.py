import torch.nn as nn
import torch  # pytorch 0.4.0! fft
import numpy as np
import cv2


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.model_alphaf = []
        self.model_xf = []
        self.config = config

    def forward(self, x):
        x = self.feature(x) * self.config.cos_window
        xf = torch.rfft(x, signal_ndim=2)
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.irfft(complex_mul(kxzf, self.model_alphaf), signal_ndim=2)
        # r_max = torch.max(response)
        # cv2.imshow('response', response[0, 0].data.cpu().numpy())
        # cv2.waitKey(0)
        return response

    def update(self, z, lr=1.):
        z = self.feature(z) * self.config.cos_window
        zf = torch.rfft(z, signal_ndim=2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict'] 
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)


if __name__ == '__main__':

    # network test
    net = DCFNetFeature()
    net.eval()
    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)
    for name, param in net.named_parameters():
        if 'bias' in name or 'weight' in name:
            print(param.size())
    from scipy import io
    import numpy as np
    p = io.loadmat('net_param.mat')
    x = p['res'][0][0][:,:,::-1].copy()
    x_out = p['res'][0][-1]
    from collections import OrderedDict
    pth_state_dict = OrderedDict()

    match_dict = dict()
    match_dict['feature.0.weight'] = 'conv1_w'
    match_dict['feature.0.bias'] = 'conv1_b'
    match_dict['feature.2.weight'] = 'conv2_w'
    match_dict['feature.2.bias'] = 'conv2_b'

    for var_name in net.state_dict().keys():
        print var_name
        key_in_model = match_dict[var_name]
        param_in_model = var_name.rsplit('.', 1)[1]
        if 'weight' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.transpose(p[key_in_model],(3,2,0,1)))
        elif 'bias' in var_name:
            pth_state_dict[var_name] = torch.Tensor(np.squeeze(p[key_in_model]))
        if var_name == 'feature.0.weight':
            weight = pth_state_dict[var_name].data.numpy()
            weight = weight[:, ::-1, :, :].copy()  # cv2 bgr input
            pth_state_dict[var_name] = torch.Tensor(weight)


    torch.save(pth_state_dict, 'param.pth')
    net.load_state_dict(torch.load('param.pth'))
    x_t = torch.Tensor(np.expand_dims(np.transpose(x,(2,0,1)), axis=0))
    x_pred = net(x_t).data.numpy()
    pred_error = np.sum(np.abs(np.transpose(x_pred,(0,2,3,1)).reshape(-1) - x_out.reshape(-1)))

    x_fft = torch.rfft(x_t, signal_ndim=2, onesided=False)


    print('model_transfer_error:{:.5f}'.format(pred_error))


