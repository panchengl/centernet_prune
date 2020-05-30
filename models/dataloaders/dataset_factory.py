from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from models.dataloaders.ctdet import CTDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.dianli import dianli


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'dianli':dianli
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
