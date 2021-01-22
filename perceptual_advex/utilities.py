from typing import Optional, Tuple
import torch
import os
import copy
import torchvision.models as torchvision_models
from torch import nn
from robustness.datasets import DATASETS, DataSet
from robustness.model_utils import make_and_restore_model
from robustness.attacker import AttackerModel
from advex_uar.common import pyt_common as uar_common

from .models import CifarResNetFeatureModel, ImageNetResNetFeatureModel, \
    AlexNetFeatureModel, CifarAlexNet, VGG16FeatureModel, TradesWideResNet
from . import datasets


class MarginLoss(nn.Module):
    """
    Calculates the margin loss max(kappa, (max z_k (x) k != y) - z_y(x)),
    also known as the f6 loss used by the Carlini & Wagner attack.
    """

    def __init__(self, kappa=float('inf'), targeted=False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))

        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max

        if self.targeted:
            return (correct_logits - max_incorrect_logits) \
                .clamp(max=self.kappa).squeeze()
        else:
            return (max_incorrect_logits - correct_logits) \
                .clamp(max=self.kappa).squeeze()


def add_dataset_model_arguments(parser, include_checkpoint=False):
    """
    Adds the argparse arguments to the given parser necessary for calling the
    get_dataset_model command.
    """

    if include_checkpoint:
        parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    parser.add_argument('--arch', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--dataset', type=str, default='cifar',
                        help='dataset name')
    parser.add_argument('--dataset_path', type=str, default='~/datasets',
                        help='path to datasets directory')


def get_dataset_model(
    args=None,
    dataset_path: Optional[str] = None,
    arch: Optional[str] = None,
    checkpoint_fname: Optional[str] = None,
    **kwargs,
) -> Tuple[DataSet, nn.Module]:
    """
    Given an argparse namespace with certain parameters, or those parameters
    as keyword arguments, returns a tuple (dataset, model) with a robustness
    dataset and a FeatureModel.
    """

    if dataset_path is None:
        if args is None:
            dataset_path = '~/datasets'
        else:
            dataset_path = args.dataset_path
    dataset_path = os.path.expandvars(dataset_path)

    dataset_name = kwargs.get('dataset') or args.dataset
    dataset = DATASETS[dataset_name](dataset_path)

    checkpoint_is_feature_model = False

    if checkpoint_fname is None:
        checkpoint_fname = getattr(args, 'checkpoint', None)
    if arch is None:
        arch = args.arch

    if arch.startswith('rob-') or (
        dataset_name.startswith('cifar') and
        'resnet' in arch
    ):
        if arch.startswith('rob-'):
            arch = arch[4:]
        if checkpoint_fname == 'pretrained':
            pytorch_pretrained = True
            checkpoint_fname = None
        else:
            pytorch_pretrained = False
        try:
            model, _ = make_and_restore_model(
                arch=arch,
                dataset=dataset,
                resume_path=checkpoint_fname,
                pytorch_pretrained=pytorch_pretrained,
                parallel=False,
            )
        except RuntimeError as error:
            if 'state_dict' in str(error):
                model, _ = make_and_restore_model(
                    arch=arch,
                    dataset=dataset,
                    parallel=False,
                )
                try:
                    state = torch.load(checkpoint_fname)
                    model.model.load_state_dict(state['model'])
                except RuntimeError as error:
                    if 'state_dict' in str(error):
                        checkpoint_is_feature_model = True
                    else:
                        raise error
            else:
                raise error  # type: ignore
    elif arch == 'trades-wrn':
        model = TradesWideResNet()
        if checkpoint_fname is not None:
            state = torch.load(checkpoint_fname)
            model.load_state_dict(state)
    elif hasattr(torchvision_models, arch):
        if (
            arch == 'alexnet' and
            dataset_name.startswith('cifar') and
            checkpoint_fname != 'pretrained'
        ):
            model = CifarAlexNet(num_classes=dataset.num_classes)
        else:
            if checkpoint_fname == 'pretrained':
                model = getattr(torchvision_models, arch)(pretrained=True)
            else:
                model = getattr(torchvision_models, arch)(
                    num_classes=dataset.num_classes)

        if checkpoint_fname is not None and checkpoint_fname != 'pretrained':
            try:
                state = torch.load(checkpoint_fname)
                model.load_state_dict(state['model'])
            except RuntimeError as error:
                if 'state_dict' in str(error):
                    checkpoint_is_feature_model = True
                else:
                    raise error
    else:
        raise RuntimeError(f'Unsupported architecture {arch}.')

    if 'alexnet' in arch:
        model = AlexNetFeatureModel(model)
    elif 'vgg16' in arch:
        model = VGG16FeatureModel(model)
    elif 'resnet' in arch:
        if not isinstance(model, AttackerModel):
            model = AttackerModel(model, dataset)
        if dataset_name.startswith('cifar'):
            model = CifarResNetFeatureModel(model)
        elif (
            dataset_name.startswith('imagenet')
            or dataset_name == 'bird_or_bicycle'
        ):
            model = ImageNetResNetFeatureModel(model)
        else:
            raise RuntimeError('Unsupported dataset.')
    elif arch == 'trades-wrn':
        pass  # We can't use this as a FeatureModel yet.
    else:
        raise RuntimeError(f'Unsupported architecture {arch}.')

    if checkpoint_is_feature_model:
        model.load_state_dict(state['model'])

    return dataset, model


def calculate_accuracy(logits, labels):
    correct = logits.argmax(1) == labels
    return correct.float().mean()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def run_attack_with_random_targets(attack, model, inputs, labels, num_classes):
    """
    Runs an attack with targets randomly selected from all classes besides the
    correct one. The attack should be a function from (inputs, labels) to
    adversarial examples.
    """

    rand_targets = torch.randint(
        0, num_classes - 1, labels.size(),
        dtype=labels.dtype, device=labels.device,
    )
    targets = torch.remainder(labels + rand_targets + 1, num_classes)

    adv_inputs = attack(inputs, targets)
    adv_labels = model(adv_inputs).argmax(1)
    unsuccessful = adv_labels != targets
    adv_inputs[unsuccessful] = inputs[unsuccessful]

    return adv_inputs
