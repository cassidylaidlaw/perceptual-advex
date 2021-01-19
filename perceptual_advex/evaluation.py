
import torch
import random
import copy
from torch import nn

from .distances import LPIPSDistance


def evaluate_against_attacks(model, attacks, val_loader, parallel=1,
                             writer=None, iteration=None, num_batches=None):
    """
    Evaluates a model against the given attacks, printing the output and
    optionally writing it to a tensorboardX summary writer.
    """

    model_lpips_model: nn.Module = LPIPSDistance(model)
    alexnet_lpips_model: nn.Module = LPIPSDistance()

    if torch.cuda.is_available():
        model_lpips_model.cuda()
        alexnet_lpips_model.cuda()

        device_ids = list(range(parallel))
        model_lpips_model = nn.DataParallel(model_lpips_model, device_ids)
        alexnet_lpips_model = nn.DataParallel(alexnet_lpips_model, device_ids)

    model_state_dict = copy.deepcopy(model.state_dict())

    for attack in attacks:
        if isinstance(attack, nn.DataParallel):
            attack_name = attack.module.__class__.__name__
        else:
            attack_name = attack.__class__.__name__

        batches_correct = []
        successful_attacks = []
        successful_model_lpips = []
        successful_alexnet_lpips = []
        for batch_index, (inputs, labels) in enumerate(val_loader):
            if num_batches is not None and batch_index >= num_batches:
                break

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            adv_inputs = attack(inputs, labels)

            with torch.no_grad():
                logits = model(inputs)
                adv_logits = model(adv_inputs)
            batches_correct.append((adv_logits.argmax(1) == labels).detach())

            success = (
                (logits.argmax(1) == labels) &  # was classified correctly
                (adv_logits.argmax(1) != labels)  # and now is not
            )

            inputs_success = inputs[success]
            adv_inputs_success = adv_inputs[success]
            num_samples = min(len(inputs_success), 1)
            adv_indices = random.sample(range(len(inputs_success)),
                                        num_samples)
            for adv_index in adv_indices:
                successful_attacks.append(torch.cat([
                    inputs_success[adv_index],
                    adv_inputs_success[adv_index],
                    torch.clamp((adv_inputs_success[adv_index] -
                                 inputs_success[adv_index]) * 3 + 0.5,
                                0, 1),
                ], dim=1).detach())

            if success.sum() > 0:
                successful_model_lpips.extend(model_lpips_model(
                    inputs_success,
                    adv_inputs_success,
                ).detach())
                successful_alexnet_lpips.extend(alexnet_lpips_model(
                    inputs_success,
                    adv_inputs_success,
                ).detach())
        print_cols = [f'ATTACK {attack_name}']

        correct = torch.cat(batches_correct)
        accuracy = correct.float().mean()
        if writer is not None:
            writer.add_scalar(f'val/{attack_name}/accuracy',
                              accuracy.item(),
                              iteration)
        print_cols.append(f'accuracy: {accuracy.item() * 100:.1f}%')

        print(*print_cols, sep='\t')

        for lpips_name, successful_lpips in [
            ('alexnet', successful_alexnet_lpips),
            ('model', successful_model_lpips),
        ]:
            if len(successful_lpips) > 0 and writer is not None:
                writer.add_histogram(f'val/{attack_name}/lpips/{lpips_name}',
                                     torch.stack(successful_lpips)
                                     .cpu().detach().numpy(),
                                     iteration)

        if len(successful_attacks) > 0 and writer is not None:
            writer.add_image(f'val/{attack_name}/images',
                             torch.cat(successful_attacks, dim=2),
                             iteration)

        new_model_state_dict = copy.deepcopy(model.state_dict())
        for key in model_state_dict:
            old_tensor = model_state_dict[key]
            new_tensor = new_model_state_dict[key]
            max_diff = (old_tensor - new_tensor).abs().max().item()
            if max_diff > 1e-8:
                print(f'max difference for {key} = {max_diff}')
