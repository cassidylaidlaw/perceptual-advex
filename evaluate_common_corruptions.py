
import torch
import csv
import argparse
import copy
from typing import List

from torch.hub import load_state_dict_from_url
from torch import Tensor
from torchvision.models import AlexNet
from robustness.datasets import DATASETS

from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
from perceptual_advex.datasets import ImageNet100C


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Common corruptions evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--output', type=str,
                        help='output CSV')

    args = parser.parse_args()

    _, model = get_dataset_model(args)
    dataset_cls = DATASETS[args.dataset]

    alexnet_args = copy.deepcopy(args)
    alexnet_args.arch = 'alexnet'
    alexnet_args.checkpoint = None
    if args.dataset == 'cifar10c':
        alexnet_checkpoint_fname = 'data/checkpoints/alexnet_cifar.pt'
    elif args.dataset == 'imagenet100c':
        alexnet_checkpoint_fname = 'data/checkpoints/alexnet_imagenet100.pt'
    else:
        raise ValueError(f'Invalid dataset "{args.dataset}"')
    _, alexnet = get_dataset_model(
        alexnet_args, checkpoint_fname=alexnet_checkpoint_fname)

    model.eval()
    alexnet.eval()
    if torch.cuda.is_available():
        model.cuda()
        alexnet.cuda()

    with open(args.output, 'w') as output_file:
        output_csv = csv.writer(output_file)
        output_csv.writerow([
            'corruption_type', 'severity', 'model_error', 'alexnet_error',
        ])

        for corruption_type in [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        ]:
            model_errors: List[float] = []
            alexnet_errors: List[float] = []

            for severity in range(1, 6):
                print(f'CORRUPTION\t{corruption_type}\tseverity = {severity}')

                dataset = dataset_cls(
                    args.dataset_path, corruption_type, severity)
                _, val_loader = dataset.make_loaders(
                    4, args.batch_size, only_val=True)

                batches_correct: List[Tensor] = []
                alexnet_batches_correct: List[Tensor] = []
                for batch_index, (inputs, labels) in enumerate(val_loader):
                    if (
                        args.num_batches is not None and
                        batch_index >= args.num_batches
                    ):
                        break

                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    with torch.no_grad():
                        logits = model(inputs)
                        batches_correct.append(
                            (logits.argmax(1) == labels).detach())

                        alexnet_logits = alexnet(inputs)
                        alexnet_batches_correct.append(
                            (alexnet_logits.argmax(1) == labels).detach())

                accuracy = torch.cat(batches_correct).float().mean().item()
                alexnet_accuracy = torch.cat(
                    alexnet_batches_correct).float().mean().item()
                print('OVERALL\t',
                    f'accuracy = {accuracy * 100:.1f}',
                    f'AlexNet accuracy = {alexnet_accuracy * 100:.1f}',
                    sep='\t')

                model_errors.append(1 - accuracy)
                alexnet_errors.append(1 - alexnet_accuracy)

                output_csv.writerow([
                    corruption_type, severity,
                    1 - accuracy, 1 - alexnet_accuracy,
                ])

            ce = sum(model_errors) / sum(alexnet_errors)
            output_csv.writerow([corruption_type, 'ce', ce, 1])
