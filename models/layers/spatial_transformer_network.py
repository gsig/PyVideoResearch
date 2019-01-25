import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            #nn.Conv2d(1, 8, kernel_size=7),  # grayscale
            nn.Conv2d(3, 8, kernel_size=7),  # color
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.Tensor([1, 0, 0, 0, 1, 0]).float())

    # Spatial transformer network forward function
    def forward(self, x):
        conv3d = x.dim() == 5
        if conv3d:
            b, n, d, d, c = x.shape
            x = x.reshape(-1, d, d, c)
            x = x.permute(0, 3, 1, 2)
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        if conv3d:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(b, n, d, d, c)

        return x
