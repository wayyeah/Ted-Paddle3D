# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional

import numpy as np

from paddle3d.apis import manager
from paddle3d.sample import Sample


class TransformABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sample: Sample):
        """
        """


@manager.TRANSFORMS.add_component
class Compose(TransformABC):
    """
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms

    def __call__(self, sample: Sample):
        """
        """
        i=0
        for t in self.transforms:
            
            sample = t(sample)##test not x_trans
            if t.__class__.__name__ != 'X_TRANS':
                sample['points']=sample['data']
               
            #np.save("/home/yw/points.npy",sample['points'][:,:])
            if 'gt_boxes' in sample:
                pass
                #np.save("/home/yw/boxes.npy",sample['gt_boxes'][:,:-1])

            i=i+1
        sample.pop('data')
        
        if sample.modality == 'image' and sample.meta.channel_order == 'hwc':
            sample.data = sample.data.transpose((2, 0, 1))
            sample.meta.channel_order = "chw"

        elif sample.modality == 'multimodal' or sample.modality == 'multiview':
            sample.img = np.stack(
                [img.transpose(2, 0, 1) for img in sample.img], axis=0)
        
        return sample
