import torch
import torch.nn.functional as F


def unroll_time(a, target, training):
    if training:
        nc = a.shape[1]

        # max over time, and add it to the batch
        a_video = a.mean(dim=2)
        a = F.upsample(a, target.shape[1], mode='linear', align_corners=True)
        target_video = target.max(dim=1)[0]

        # unroll over time
        a = a.permute(0, 2, 1).contiguous().view(-1, nc)
        target = target.permute(0, 1, 2).contiguous().view(-1, nc)

        # combine both
        a = torch.cat([a, a_video])
        target = torch.cat([target, target_video])
    else:
        a = a.mean(dim=2)
        target = target.max(dim=1)[0]
    return a, target


def winsmooth(mat, kernelsize=1):
    if kernelsize == 0:
        return mat
    print('applying smoothing with kernelsize {}'.format(kernelsize))
    mat.detach()
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[m, :] = mat[a:b + 1, :].mean(0)
    return out


