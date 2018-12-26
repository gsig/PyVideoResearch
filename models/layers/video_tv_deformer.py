import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoTVDeformer(nn.Module):
    def __init__(self, channels):
        super(VideoTVDeformer, self).__init__()
        self.grid = nn.Parameter(torch.Tensor(channels, 224, 224, 2))
        self.grid.data.zero_()

    def forward(self, x):
        conv3d = x.dim() == 5
        if conv3d:
            b, n, d, d, c = x.shape
            x = x.reshape(-1, d, d, c)
            x = x.permute(0, 3, 1, 2)
        x = F.grid_sample(x, self.grid, padding_mode="reflection")
        if conv3d:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(b, n, d, d, c)
        print('grid min: {} grid max: {} grid mean: {}'.format(self.grid.min(), self.grid.max(), self.grid.mean()))
        return x, self.grid
