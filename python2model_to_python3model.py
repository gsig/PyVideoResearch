#!/usr/bin/env python2

import sys
import torch
import collections

data = torch.load(sys.argv[1])
if isinstance(data, collections.Mapping) and 'state_dict' in data:
    data = data['state_dict']
    torch.save(data, sys.argv[2])
else:
    print('already a state_dicdt, doing nothing')
