import torch.nn as nn
import torch
class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

        print(self.cols_to_crop, -self.cols_to_crop)
    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


if __name__ == '__main__':
    crop_set = [-1, 0]
    la = CropLayer(crop_set=crop_set)

    img = torch.rand([1,1,3,3])
    print('img:', img)
    out = la(img)
    print(out, out.shape)







