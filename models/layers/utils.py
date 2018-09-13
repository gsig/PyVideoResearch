def gtmat(sizes, target):
    # convert target to a matrix of zeros and ones
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[i, t, :] = 1
        else:
            out[i, t] = 1
    if type(target) is Variable:
        return Variable(out.cuda())
    else:
        return out.cuda()


def winsmooth(mat, kernelsize=1):
    print('applying smoothing with kernelsize {}'.format(kernelsize))
    mat.detach()
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[m, :] = mat[a:b + 1, :].mean(0)
    return out


def unit(x):
    # normalize tensor in log space to have unit sum for each row
    minx, _ = x.max(1)
    z = (x - minx[:, None]).exp().sum(1).log() + minx
    return x - z[:, None]


def lse(x, dim=None, keepdim=False):
    # log sum exp @alshedivat
    return (x - F.log_softmax(x)).sum(dim, keepdim=keepdim)


def sme(x, y, dim=None, keepdim=False):
    # Sum mul exp
    return (x * torch.exp(y)).sum(dim, keepdim=keepdim)


def axb(a, x, b):
    # a and b are batched vectors, X is batched matrix
    # returns a^t * X * b
    xb = torch.bmm(x, b[:, :, None])
    return (a * xb.squeeze()).sum(1)


def avg(iterator, weight=1.):
    # compounding weight
    item, w = next(iterator)
    total = item.clone() * w
    n = 1.
    for i, (item, w) in enumerate(iterator):
        w1 = 1. * weight**(i + 1)
        total += item * w1 * w
        n += w1
    return total / n


def nll_loss(soft_target, logdist, reduce=True):
    # @Hongyi_Zhang
    # assumes soft_target is normalized to 1 and between [0,1]
    # logdist is a (normalized) log distribution
    logdist = unit((logdist.exp() + 0.00001).log())  # for numerical stability
    if soft_target.dim() == 3:
        out = (-soft_target * logdist).sum(2).sum(1)
    else:
        out = (-soft_target * logdist).sum(1)
    if reduce:
        return out.mean()
    else:
        return out


