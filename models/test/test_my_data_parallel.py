import unittest
from models.utils import MyDataParallel
import torch


class TestMyDataParallel(unittest.TestCase):
    def test_scatter_tensors(self):
        if torch.cuda.is_available():
            data = torch.Tensor(10, 2, 3)
            #my_data_parallel = MyDataParallel(torch.nn.Module())
            my_data_parallel = torch.nn.DataParallel(torch.nn.Module())
            out = my_data_parallel.scatter((data, ), {}, [0, 1])
            self.assertEqual(len(out), 2)
            import pdb
            pdb.set_trace()
            self.assertSequenceEqual(out[0].shape, (5, 2, 3))

    #def test_scatter_do_not_collate(self):
    #    if torch.cuda.is_available():
    #        inp = torch.Tensor(10, 2, 3)
    #        meta = {'data': torch.Tensor(3, 4)}
    #        data = (inp, meta)
    #        my_data_parallel = MyDataParallel(torch.nn.Module())
    #        out = my_data_parallel.scatter(data, {}, [0, 1])
    #        self.assertEqual(len(out), 2)
    #        self.assertSequenceEqual(out[0].shape, (5, 2, 3))
