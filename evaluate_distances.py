import argparse
import numpy as np

import torch
import csv
from torch import nn
from typing import List, Tuple
from typing_extensions import Literal

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.distances import LPIPSDistance, LinfDistance, SSIM, \
    L2Distance
from perceptual_advex.models import FeatureModel
from perceptual_advex.perceptual_attacks import get_lpips_model
from perceptual_advex.perceptual_attacks import *
from perceptual_advex.attacks import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Distance measure analysis')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', type=str, help='output CSV')
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')

    args = parser.parse_args()

    dist_models: List[Tuple[str, nn.Module]] = [
        ('l2', L2Distance()),
        ('linf', LinfDistance()),
        ('ssim', SSIM()),
    ]

    dataset, model = get_dataset_model(args)
    if not isinstance(model, FeatureModel):
        raise TypeError('model must be a FeatureModel')
    dist_models.append(('lpips_self', LPIPSDistance(model)))

    alexnet_model_name: Literal['alexnet_cifar', 'alexnet']
    if args.dataset.startswith('cifar'):
        alexnet_model_name = 'alexnet_cifar'
    else:
        alexnet_model_name = 'alexnet'
    dist_models.append((
        'lpips_alexnet',
        LPIPSDistance(get_lpips_model(alexnet_model_name, model)),
    ))

    for _, dist_model in dist_models:
        dist_model.eval()
        if torch.cuda.is_available():
            dist_model.cuda()

    _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    attack_names: List[str] = args.attacks

    with open(args.output, 'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow([
            attack_name for attack_name in attack_names
            for _ in dist_models
        ])
        out_csv.writerow([
            dist_model_name for _ in attack_names
            for dist_model_name, _ in dist_models
        ])

        for batch_index, (inputs, labels) in enumerate(val_loader):
            if (
                args.num_batches is not None and
                batch_index >= args.num_batches
            ):
                break

            print(f'BATCH\t{batch_index:05d}')

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            batch_distances = np.zeros((
                inputs.shape[0],
                len(attack_names) * len(dist_models),
            ))

            for attack_index, attack_name in enumerate(attack_names):
                print(f'ATTACK {attack_name}')
                attack = eval(attack_name)

                adv_inputs = attack(inputs, labels)
                with torch.no_grad():
                    for dist_model_index, (_, dist_model) in \
                            enumerate(dist_models):
                        batch_distances[
                            :,
                            attack_index * len(dist_models) + dist_model_index
                        ] = dist_model(
                            inputs,
                            adv_inputs,
                        ).detach().cpu().numpy()

            for row in batch_distances:
                out_csv.writerow(row.tolist())
