"""
Scripts that generates a number of adversarial examples for each of several
attacks against a particular network.
"""

import torch
import argparse
import numpy as np
import itertools
from torchvision.utils import save_image

from perceptual_advex.attacks import *
from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model


def tile_images(images):
    """
    Given a numpy array of shape r x c x C x W x H, where r and c are rows and
    columns in a grid of images, tiles the images into a numpy array
    C x (W * c) x (H * r).
    """

    return np.concatenate(np.concatenate(images, axis=2), axis=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial example generation')

    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')

    add_dataset_model_arguments(parser, include_checkpoint=True)

    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of examples to generate '
                        'adversarial examples for')
    parser.add_argument('--batch_index', type=int, default=0,
                        help='batch index to generate adversarial examples '
                        'for')
    parser.add_argument('--shuffle', default=False, action='store_true',
                        help="Shuffle dataset before choosing a batch")
    parser.add_argument('--layout', type=str, default='vertical',
                        help='lay out the same images on the same row '
                        '(horizontal) or column (vertical)')
    parser.add_argument('--only_successful', action='store_true',
                        default=False,
                        help='only show images where adversarial example '
                        'was generated for all attacks')
    parser.add_argument('--output', type=str,
                        help='output PNG file')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='seed for the Torch RNG')

    args = parser.parse_args()

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    dataset, model = get_dataset_model(args)
    _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True,
                                         shuffle_val=args.shuffle)
    model.eval()

    inputs, labels = next(itertools.islice(
        val_loader, args.batch_index, None))
    if torch.cuda.is_available():
        model.cuda()
        inputs = inputs.cuda()
        labels = labels.cuda()
    N, C, H, W = inputs.size()

    attacks = [None] + args.attacks
    out_advs = np.ones((len(attacks), N, C, H, W))
    out_diffs = np.ones_like(out_advs)

    orig_labels = model(inputs).argmax(1)
    all_successful = np.ones(N, dtype=bool)
    all_labels = np.zeros((len(attacks), len(orig_labels)), dtype=int)
    all_labels[0] = orig_labels.cpu().detach().numpy()

    for attack_index, attack_name in enumerate(attacks):
        print(f'generating examples for {attack_name or "no"} attack')

        attack_params = None
        if attack_name is None:
            out_advs[attack_index] = inputs.cpu().numpy()
            out_diffs[attack_index] = 0
        else:
            attack = eval(attack_name)

            advs = attack(inputs, labels)
            adv_labels = model(advs).argmax(1)
            successful = (adv_labels != labels).cpu().detach().numpy() \
                .astype(bool)

            print(f'accuracy = {np.mean(1 - successful) * 100:.1f}')
            diff = (advs - inputs).cpu().detach().numpy()
            advs = advs.cpu().detach().numpy()
            out_advs[attack_index, successful] = advs[successful]
            out_diffs[attack_index, successful] = diff[successful]

            all_labels[attack_index] = adv_labels.cpu().detach().numpy()

            all_successful[(adv_labels == orig_labels).cpu().detach().numpy()
                           .astype(bool)] = False
            # mark examples that changed by less than 1/1000 as not successful
            all_successful[np.all(np.abs(diff) < 1e-3,
                                  axis=(1, 2, 3))] = False

    if args.only_successful:
        out_advs = out_advs[:, all_successful]
        out_diffs = out_diffs[:, all_successful]
        all_labels = all_labels[:, all_successful]

    for image_index in range(all_labels.shape[1]):
        print(
            f'image {image_index} labels:',
            ' '.join(map(str, all_labels[:, image_index])),
        )

    out_diffs = np.clip(out_diffs * 3 + 0.5, 0, 1)

    combined_image: np.ndarray
    if args.layout == 'vertical':
        if len(attacks) == 2:
            combined_grid = np.concatenate([
                out_advs,
                np.clip(out_diffs[1:2], 0, 1),
            ], axis=0)
        else:
            combined_grid = np.concatenate([
                out_advs,
                np.ones((len(attacks), 1, C, H, W)),
                out_diffs,
            ], axis=1)
        combined_image = tile_images(combined_grid)
    elif args.layout == 'horizontal_alternate':
        rows = []
        for i in range(out_advs.shape[1]):
            row = []
            row.append(out_advs[0, i])
            for adv, diff in zip(out_advs[1:, i], out_diffs[1:, i]):
                row.append(np.ones((C, H, W // 4)))
                row.append(adv)
                row.append(diff)
            rows.append(np.concatenate(row, axis=2))
        combined_image = np.concatenate(rows, axis=1)
    elif args.layout == 'vertical_alternate':
        rows = []
        for i in range(out_advs.shape[0]):
            row = []
            for adv, diff in zip(out_advs[i], out_diffs[i]):
                row.append(np.ones((C, H, W // 4)))
                row.append(adv)
                row.append(diff)
            rows.append(np.concatenate(row[1:], axis=2))
        combined_image = np.concatenate(rows, axis=1)
    else:
        raise ValueError(f'Unknown layout "{args.layout}"')
    save_image(torch.from_numpy(combined_image), args.output)
