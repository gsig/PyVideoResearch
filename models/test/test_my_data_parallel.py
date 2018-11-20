import unittest
from models.utils import MyDataParallel
import torch


class TestMyDataParallel(unittest.TestCase):
    def test_scatter_tensors(self):
        if torch.cuda.is_available() and torch.cuda.device_count() == 2:
            data = torch.Tensor(10, 2, 3)
            my_data_parallel = MyDataParallel(torch.nn.Module())
            out, _ = my_data_parallel.scatter((data, ), {}, [0, 1])
            # out is #gpu x #args x tensor_dims
            self.assertEqual(len(out), 2)
            self.assertSequenceEqual(out[0][0].shape, (5, 2, 3))
        else:
            print('skipping test_scatter_tensors, rerun with 2 gpus visible')

    def test_scatter_do_not_collate(self):
        if torch.cuda.is_available() and torch.cuda.device_count() == 2:
            inp = torch.Tensor(10, 2, 3)
            meta = [{'do_not_collate': True,
                    'data': torch.Tensor(3, 4)}] * 10
            data = (inp, meta)
            my_data_parallel = MyDataParallel(torch.nn.Module())
            out, _ = my_data_parallel.scatter(data, {}, [0, 1])
            self.assertEqual(len(out), 2)
            self.assertSequenceEqual(out[0][0].shape, (5, 2, 3))
            self.assertEqual(len(out[0][1]), 5)
            self.assertSequenceEqual(out[0][1][0]['data'].shape, (3, 4))
        else:
            print('skipping test_scatter_do_not_collate, rerun with 2 gpus visible')
