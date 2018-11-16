import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
import sys
import psutil


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def heatim(original, overlay):
    heat_img = plt.cm.jet(overlay/np.max(overlay))[:, :, :3]
    out = np.copy(original)
    avg = np.mean(original, axis=2)
    out[:, :, 0] = avg
    out[:, :, 1] = avg
    out[:, :, 2] = avg
    out = out*.5+heat_img*.3
    return out


def arrtoim(arr):
    return Image.fromarray(np.asarray(np.clip(arr*255., 0, 255), dtype="uint8"), "RGB")


def imtoarr(im):
    return np.asarray(np.asarray(im, dtype="int32"), dtype="float")/255.


def normalize_arr(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr


def tensor_as_image(tensor, filename='debug.png'):
    data = tensor.cpu().numpy().copy()
    data = data.swapaxes(0, 2).swapaxes(0, 1)
    data = normalize_arr(data)
    arrtoim(data).save(filename)


def mem_report():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device.index)
        except Exception as e:
            print(e)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memory_use)


def mem_report2():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    #_mem_report(host_tensors, 'CPU')
    print('='*LEN)
