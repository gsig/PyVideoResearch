import unittest
from datasets.get import my_collate
import torch


class TestMyCollate(unittest.TestCase):
    def test_my_collate_single_tensors(self):
        # single tensor data points
        data = [torch.Tensor(2, 3)] * 10
        out = my_collate(data)
        self.assertSequenceEqual(out.shape, (10, 2, 3))

    def test_my_collate_dict(self):
        # single tensor data points and 'do_not_collate' dictionary
        inp = torch.Tensor(2, 3)
        meta = {'data': torch.Tensor(3, 4)}
        data = [(inp, meta)] * 10
        out = my_collate(data)
        self.assertSequenceEqual(out[0].shape, (10, 2, 3))
        self.assertEqual(out[1]['data'].shape, (10, 3, 4))

    def test_my_collate_do_not_collate(self):
        # single tensor data points and 'do_not_collate' dictionary
        inp = torch.Tensor(2, 3)
        meta = {'do_not_collate': True,
                'data': torch.Tensor(3, 4)}
        data = [(inp, meta)] * 10
        out = my_collate(data)
        self.assertSequenceEqual(out[0].shape, (10, 2, 3))
        self.assertEqual(len(out[1]), 10)
