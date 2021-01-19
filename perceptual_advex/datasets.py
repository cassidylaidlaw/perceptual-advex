import torch
import tempfile
import os
import numpy as np

from torchvision.datasets import CIFAR10

from robustness.datasets import CIFAR, DATASETS, DataSet, CustomImageNet
from robustness.data_augmentation import TRAIN_TRANSFORMS_IMAGENET, \
    TEST_TRANSFORMS_IMAGENET
from robustness import data_augmentation
from torchvision.datasets.vision import VisionDataset 


class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):

        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


class ImageNet100A(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[
                [],
                [],
                [],
                [8],
                [],
                [13],
                [],
                [15],
                [],
                [20],
                [],
                [28],
                [],
                [32],
                [],
                [36],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [53],
                [],
                [64],
                [],
                [],
                [],
                [],
                [],
                [],
                [75],
                [],
                [83],
                [86],
                [],
                [],
                [],
                [94],
                [],
                [],
                [],
                [],
                [],
                [104],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [125],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [150],
                [],
                [],
                [],
                [159],
                [],
                [],
                [167],
                [],
                [170],
                [172],
                [174],
                [176],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [194],
                [],
            ],
            **kwargs,
        )


class ImageNet100C(CustomImageNet):
    """
    ImageNet-C, but restricted to the ImageNet-100 classes.
    """

    def __init__(
        self,
        data_path,
        corruption_type: str = 'gaussian_noise',
        severity: int = 1,
        **kwargs,
    ):
        # Need to create a temporary directory to act as the dataset because
        # the robustness library expects a particular directory structure.
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(os.path.join(data_path, corruption_type, str(severity)),
                   os.path.join(tmp_data_path, 'test'))

        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


class CIFAR10C(CIFAR):
    """
    CIFAR-10-C from https://github.com/hendrycks/robustness.
    """

    def __init__(
        self,
        data_path,
        corruption_type: str = 'gaussian_noise',
        severity: int = 1,
        **kwargs,
    ):
        class CustomCIFAR10(CIFAR10):
            def __init__(self, root, train=True, transform=None,
                         target_transform=None, download=False):
                VisionDataset.__init__(self, root, transform=transform,
                                       target_transform=target_transform)

                if train:
                    raise NotImplementedError(
                        'No train dataset for CIFAR-10-C')
                if download and not os.path.exists(root):
                    raise NotImplementedError(
                        'Downloading CIFAR-10-C has not been implemented')

                all_data = np.load(
                    os.path.join(root, f'{corruption_type}.npy'))
                all_labels = np.load(os.path.join(root, f'labels.npy'))

                severity_slice = slice(
                    (severity - 1) * 10000,
                    severity * 10000,
                )

                self.data = all_data[severity_slice]
                self.targets = all_labels[severity_slice]

        DataSet.__init__(
            self,
            'cifar10c',
            data_path,
            num_classes=10,
            mean=torch.tensor([0.4914, 0.4822, 0.4465]),
            std=torch.tensor([0.2023, 0.1994, 0.2010]),
            custom_class=CustomCIFAR10,
            label_mapping=None, 
            transform_train=data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32),
            transform_test=data_augmentation.TEST_TRANSFORMS_DEFAULT(32)
        )


class BirdOrBicycle(DataSet):
    """
    Bird-or-bicycle dataset.
    https://github.com/google/unrestricted-adversarial-examples/tree/master/bird-or-bicycle
    """

    def __init__(self, data_path=None, **kwargs):
        ds_name = 'bird_or_bicycle'
        import bird_or_bicycle

        # Need to create a temporary directory to act as the dataset because
        # the robustness library expects a particular directory structure.
        data_path = tempfile.mkdtemp()
        os.symlink(bird_or_bicycle.get_dataset('extras'),
                   os.path.join(data_path, 'train'))
        os.symlink(bird_or_bicycle.get_dataset('test'),
                   os.path.join(data_path, 'test'))

        ds_kwargs = {
            'num_classes': 2,
            'mean': torch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': torch.tensor([0.2600, 0.2516, 0.2575]),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': TRAIN_TRANSFORMS_IMAGENET,
            'transform_test': TEST_TRANSFORMS_IMAGENET,
        }
        super().__init__(ds_name, data_path, **ds_kwargs)


DATASETS['imagenet100'] = ImageNet100
DATASETS['imagenet100a'] = ImageNet100A
DATASETS['imagenet100c'] = ImageNet100C
DATASETS['cifar10c'] = CIFAR10C
DATASETS['bird_or_bicycle'] = BirdOrBicycle
