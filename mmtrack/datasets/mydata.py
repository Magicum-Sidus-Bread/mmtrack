from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class Mydata(DOTADataset):
    """Mydataset"""
    CLASSES = ('1', '2')