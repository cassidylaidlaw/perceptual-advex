from typing import Any, Callable, List, Optional, cast
import torch
import argparse
import numpy as np
import shutil
import glob
import time
import random
import os
from torch import nn
from tensorboardX import SummaryWriter

from perceptual_advex import evaluation
from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model, calculate_accuracy
from perceptual_advex.attacks import *
from perceptual_advex.models import FeatureModel

VAL_ITERS = 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_dataset_model_arguments(parser)

    parser.add_argument('--num_epochs', type=int, required=False,
                        help='number of epochs trained')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--val_batches', type=int, default=10,
                        help='number of batches to validate on')
    parser.add_argument('--log_dir', type=str, default='data/logs')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')

    parser.add_argument('--lpips_model', type=str, required=False,
                        help='model to use for LPIPS distance')
    parser.add_argument('--only_attack_correct', action='store_true',
                        default=False, help='only attack examples that '
                        'are classified correctly')
    parser.add_argument('--randomize_attack', action='store_true',
                        default=False,
                        help='randomly choose an attack at each step')
    parser.add_argument('--maximize_attack', action='store_true',
                        default=False,
                        help='choose the attack with maximum loss')

    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--continue', default=False, action='store_true',
                        help='continue previous training')
    parser.add_argument('--keep_every', type=int, default=1,
                        help='only keep a checkpoint every X epochs')

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, metavar='LR', required=False,
                        help='learning rate')
    parser.add_argument('--lr_schedule', type=str, required=False,
                        help='comma-separated list of epochs when learning '
                        'rate should drop')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='clip gradients to this value')

    parser.add_argument('--attack', type=str, action='append',
                        help='attack(s) to harden against')

    args = parser.parse_args()

    if args.optim == 'adam':
        if args.lr is None:
            args.lr = 1e-3
        if args.lr_schedule is None:
            args.lr_schedule = '120'
        if args.num_epochs is None:
            args.num_epochs = 100
    elif args.optim == 'sgd':
        if args.dataset.startswith('cifar'):
            if args.lr is None:
                args.lr = 1e-1
            if args.lr_schedule is None:
                args.lr_schedule = '75,90,100'
            if args.num_epochs is None:
                args.num_epochs = 100
        elif (
            args.dataset.startswith('imagenet')
            or args.dataset == 'bird_or_bicycle'
        ):
            if args.lr is None:
                args.lr = 1e-1
            if args.lr_schedule is None:
                args.lr_schedule = '30,60,80'
            if args.num_epochs is None:
                args.num_epochs = 90

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset, model = get_dataset_model(args)
    if isinstance(model, FeatureModel):
        model.allow_train()
    if torch.cuda.is_available():
        model.cuda()

    if args.lpips_model is not None:
        _, lpips_model = get_dataset_model(
            args, checkpoint_fname=args.lpips_model)
        if torch.cuda.is_available():
            lpips_model.cuda()

    train_loader, val_loader = dataset.make_loaders(
        workers=4, batch_size=args.batch_size)

    attacks = [eval(attack_str) for attack_str in args.attack]
    validation_attacks = [
        NoAttack(),
        LinfAttack(model, dataset_name=args.dataset,
                   num_iterations=VAL_ITERS),
        L2Attack(model, dataset_name=args.dataset,
                 num_iterations=VAL_ITERS),
        JPEGLinfAttack(model, dataset_name=args.dataset,
                       num_iterations=VAL_ITERS),
        FogAttack(model, dataset_name=args.dataset,
                  num_iterations=VAL_ITERS),
        StAdvAttack(model, num_iterations=VAL_ITERS),
        ReColorAdvAttack(model, num_iterations=VAL_ITERS),
        LagrangePerceptualAttack(model, num_iterations=30),
    ]

    flags = []
    if args.only_attack_correct:
        flags.append('only_attack_correct')
    if args.randomize_attack:
        flags.append('random')
    if args.maximize_attack:
        flags.append('maximum')
    if args.lpips_model:
        lpips_model_name, _ = os.path.splitext(os.path.basename(
            args.lpips_model))
        flags.append(lpips_model_name)

    experiment_path_parts = [args.dataset, args.arch]
    if args.optim != 'sgd':
        experiment_path_parts.append(args.optim)
    attacks_part = '-'.join(args.attack + flags)
    if len(attacks_part) > 255:
        attacks_part = (
            attacks_part
            .replace('model, ', '')
            .replace("'imagenet100', ", '')
            .replace("'cifar', ", '')
            .replace(", num_iterations=10", '')
        )
    experiment_path_parts.append(attacks_part)
    experiment_path = os.path.join(*experiment_path_parts)

    iteration = 0
    log_dir = os.path.join(args.log_dir, experiment_path)
    if os.path.exists(log_dir):
        print(f'The log directory {log_dir} exists, delete? (y/N) ', end='')
        if not vars(args)['continue'] and input().strip() == 'y':
            shutil.rmtree(log_dir)
            # sleep necessary to prevent weird bug where directory isn't
            # actually deleted
            time.sleep(5)
    writer = SummaryWriter(log_dir)

    # optimizer
    optimizer: optim.Optimizer
    if args.optim == 'sgd':
        weight_decay = 1e-4 if (
            args.dataset.startswith('imagenet')
            or args.dataset == 'bird_or_bicycle'
        ) else 2e-4
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters())
    else:
        raise ValueError(f'invalid optimizer {args.optim}')

    lr_drop_epochs = [int(epoch_str) for epoch_str in
                      args.lr_schedule.split(',')]

    # check for checkpoints
    def get_checkpoint_fnames():
        for checkpoint_fname in glob.glob(os.path.join(glob.escape(log_dir),
                                                       '*.ckpt.pth')):
            epoch = int(os.path.basename(checkpoint_fname).split('.')[0])
            if epoch < args.num_epochs:
                yield epoch, checkpoint_fname

    start_epoch = 0
    latest_checkpoint_epoch = -1
    latest_checkpoint_fname = None
    for epoch, checkpoint_fname in get_checkpoint_fnames():
        if epoch > latest_checkpoint_epoch:
            latest_checkpoint_epoch = epoch
            latest_checkpoint_fname = checkpoint_fname
    if latest_checkpoint_fname is not None:
        print(f'Load checkpoint {latest_checkpoint_fname}? (Y/n) ', end='')
        if vars(args)['continue'] or input().strip() != 'n':
            state = torch.load(latest_checkpoint_fname)
            if 'iteration' in state:
                iteration = state['iteration']
            if isinstance(model, FeatureModel):
                model.model.load_state_dict(state['model'])
            else:
                model.load_state_dict(state['model'])
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
            start_epoch = latest_checkpoint_epoch + 1
            adaptive_eps = state.get('adaptive_eps', {})

    # parallelize
    if torch.cuda.is_available():
        device_ids = list(range(args.parallel))
        model = nn.DataParallel(model, device_ids)
        attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]
        validation_attacks = [nn.DataParallel(attack, device_ids)
                              for attack in validation_attacks]

    # necessary to put training loop in a function because otherwise we get
    # huge memory leaks
    def run_iter(
        inputs: torch.Tensor,
        labels: torch.Tensor,
        iteration: int,
        train: bool = True,
        log_fn: Optional[Callable[[str, Any], Any]] = None,
    ):
        prefix = 'train' if train else 'val'
        if log_fn is None:
            log_fn = lambda tag, value: writer.add_scalar(
                f'{prefix}/{tag}', value, iteration)

        model.eval()  # set model to eval to generate adversarial examples

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        if args.only_attack_correct:
            with torch.no_grad():
                orig_logits = model(inputs)
                to_attack = orig_logits.argmax(1) == labels
        else:
            to_attack = torch.ones_like(labels).bool()

        if args.randomize_attack:
            step_attacks = [random.choice(attacks)]
        else:
            step_attacks = attacks

        adv_inputs_list: List[torch.Tensor] = []
        for attack in step_attacks:
            attack_adv_inputs = inputs.clone()
            if to_attack.sum() > 0:
                attack_adv_inputs[to_attack] = attack(inputs[to_attack],
                                                      labels[to_attack])
            adv_inputs_list.append(attack_adv_inputs)
        adv_inputs: torch.Tensor = torch.cat(adv_inputs_list)

        all_labels = torch.cat([labels for attack in step_attacks])

        # FORWARD PASS
        if train:
            optimizer.zero_grad()
            model.train()  # now we set the model to train mode

        logits = model(adv_inputs)

        # CONSTRUCT LOSS
        loss = F.cross_entropy(logits, all_labels, reduction='none')
        if args.maximize_attack:
            loss, _ = loss.resize(len(step_attacks), inputs.size()[0]).max(0)
        loss = loss.mean()

        # LOGGING
        accuracy = calculate_accuracy(logits, all_labels)
        log_fn('loss', loss.item())
        log_fn('accuracy', accuracy.item())

        with torch.no_grad():
            for attack_index, attack in enumerate(step_attacks):
                if isinstance(attack, nn.DataParallel):
                    attack_name = attack.module.__class__.__name__
                else:
                    attack_name = attack.__class__.__name__
                attack_logits = logits[
                    attack_index * inputs.size()[0]:
                    (attack_index + 1) * inputs.size()[0]
                ]
                log_fn(f'loss/{attack_name}',
                       F.cross_entropy(attack_logits, labels).item())
                log_fn(f'accuracy/{attack_name}',
                       calculate_accuracy(attack_logits, labels).item())

        if train:
            print(f'ITER {iteration:06d}',
                  f'accuracy: {accuracy.item() * 100:5.1f}%',
                  f'loss: {loss.item():.2f}',
                  sep='\t')

        # OPTIMIZATION
        if train:
            loss.backward()

            # clip gradients and optimize
            nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
            optimizer.step()

    for epoch in range(start_epoch, args.num_epochs):
        lr = args.lr
        for lr_drop_epoch in lr_drop_epochs:
            if epoch >= lr_drop_epoch:
                lr *= 0.1

        print(f'START EPOCH {epoch:04d} (lr={lr:.0e})')
        for batch_index, (inputs, labels) in enumerate(train_loader):
            # ramp-up learning rate for SGD
            if epoch < 5 and args.optim == 'sgd' and args.lr >= 0.1:
                lr = (iteration + 1) / (5 * len(train_loader)) * args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            run_iter(inputs, labels, iteration)
            iteration += 1
        print(f'END EPOCH {epoch:04d}')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # VALIDATION
        print('BEGIN VALIDATION')
        model.eval()

        evaluation.evaluate_against_attacks(
            model, validation_attacks, val_loader, parallel=args.parallel,
            writer=writer, iteration=iteration, num_batches=args.val_batches,
        )

        checkpoint_fname = os.path.join(log_dir, f'{epoch:04d}.ckpt.pth')
        print(f'CHECKPOINT {checkpoint_fname}')
        checkpoint_model = model
        if isinstance(checkpoint_model, nn.DataParallel):
            checkpoint_model = checkpoint_model.module
        if isinstance(checkpoint_model, FeatureModel):
            checkpoint_model = checkpoint_model.model
        state = {
            'model': checkpoint_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration,
            'arch': args.arch,
        }
        torch.save(state, checkpoint_fname)

        # delete extraneous checkpoints
        last_keep_checkpoint = (epoch // args.keep_every) * args.keep_every
        for epoch, checkpoint_fname in get_checkpoint_fnames():
            if epoch < last_keep_checkpoint and epoch % args.keep_every != 0:
                print(f'  remove {checkpoint_fname}')
                os.remove(checkpoint_fname)

    print('BEGIN EVALUATION')
    model.eval()

    evaluation.evaluate_against_attacks(
        model, validation_attacks, val_loader, parallel=args.parallel,
    )
    print('END EVALUATION')
