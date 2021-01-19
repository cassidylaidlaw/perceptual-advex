import sys
import os
import functools
from torch import nn
from operator import mul
from torch import optim
from advex_uar.common.pyt_common import get_attack as get_uar_attack
from advex_uar.attacks.attacks import InverseImagenetTransform

from .perceptual_attacks import *
from .utilities import LambdaLayer
from . import utilities

# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap 
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs


PGD_ITERS = 20
DATASET_NUM_CLASSES = {
    'cifar': 10,
    'imagenet100': 100,
    'imagenet': 1000,
    'bird_or_bicycle': 2,
}


class NoAttack(nn.Module):
    """
    Attack that does nothing.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return inputs


class MisterEdAttack(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, randomize=False,
                 perturbation_norm_loss=False, lr=0.001, random_targets=False,
                 num_classes=None, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.randomize = randomize
        self.perturbation_norm_loss = perturbation_norm_loss
        self.attack_kwargs = kwargs
        self.lr = lr
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.attack = None

    def _setup_attack(self):
        cw_loss = lf.CWLossF6(self.model, self.normalizer, kappa=float('inf'))
        if self.random_targets:
            cw_loss.forward = functools.partial(cw_loss.forward, targeted=True)
        perturbation_loss = lf.PerturbationNormLoss(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = lf.RegularizedLoss({
            'cw': cw_loss,
            'pert': perturbation_loss,
        }, {
            'cw': 1.0,
            'pert': pert_factor,
        }, negate=True)

        self.pgd_attack = aa.PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'signed': False,
            'verbose': False,
            'num_iterations': 0 if self.randomize else PGD_ITERS,
            'random_init': self.randomize,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = advtrain.AdversarialAttackParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                lambda inputs, labels: self.attack.attack(inputs, labels)[0],
                self.model,
                inputs,
                labels,
                num_classes=self.num_classes,
            )
        else:
            return self.attack.attack(inputs, labels)[0]


class UARModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        inverse_transform = InverseImagenetTransform(x.size()[-1])
        return self.model(inverse_transform(x) / 255)


class UARAttack(nn.Module):
    """
    One of the attacks from the paper "Testing Robustness Against Unforeseen
    Adversaries".
    """

    def __init__(self, model, dataset_name, attack_name, bound,
                 num_iterations=PGD_ITERS, step=None, random_targets=False,
                 randomize=False):
        super().__init__()

        assert randomize is False

        if step is None:
            step = bound / (num_iterations ** 0.5)

        self.random_targets = random_targets
        self.num_classes = DATASET_NUM_CLASSES[dataset_name]
        if (
           dataset_name.startswith('imagenet')
           or dataset_name == 'bird_or_bicycle'
        ):
            dataset_name = 'imagenet'
        elif dataset_name == 'cifar':
            dataset_name = 'cifar-10'

        self.model = model
        self.uar_model = UARModel(model)
        self.attack_name = attack_name
        self.bound = bound
        self.attack_fn = get_uar_attack(dataset_name, attack_name, eps=bound,
                                        n_iters=num_iterations,
                                        step_size=step, scale_each=1)
        self.attack = None

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        if self.attack_name == 'pgd_linf':
            dist = (inputs - adv_inputs).reshape(inputs.size()[0], -1) \
                .abs().max(1)[0] * 255
        elif self.attack_name == 'pgd_l2':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                ** 2
            ).sum(1).sqrt() * 255
        elif self.attack_name == 'fw_l1':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                .abs().sum(1)
                * 255 / functools.reduce(mul, inputs.size()[1:])
            )
        else:
            raise NotImplementedError()

        return dist <= self.bound

    def forward(self, inputs, labels):
        self.uar_model.training = self.model.training

        if self.attack is None:
            self.attack = self.attack_fn()
            self.attack.transform = LambdaLayer(lambda x: x / 255)
            self.attack.inverse_transform = LambdaLayer(lambda x: x * 255)

        if self.random_targets:
            attack = lambda inputs, targets: self.attack(
                self.uar_model,
                inputs,
                targets,
                avoid_target=False,
                scale_eps=False,
            )
            adv_examples = utilities.run_attack_with_random_targets(
                attack, self.model, inputs, labels, self.num_classes,
            )
        else:
            adv_examples = self.attack(self.uar_model, inputs, labels,
                                       scale_eps=False, avoid_target=True)

        # Some UAR attacks produce NaNs, so try to get rid of them here.
        perturbations = adv_examples - inputs
        perturbations[torch.isnan(perturbations)] = 0
        return (inputs + perturbations).detach()


class LinfAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8,
                'imagenet100': 8,
                'imagenet': 8,
                'bird_or_bicycle': 16,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_linf',
            bound=bound,
            **kwargs,
        )


class L2Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 255,
                'imagenet100': 3 * 255,
                'imagenet': 3 * 255,
                'bird_or_bicycle': 10 * 255,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_l2',
            bound=bound,
            **kwargs,
        )


class L1Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 0.5078125,
                'imagenet100': 1.016422,
                'imagenet': 1.016422,
                'bird_or_bicycle': 1.016422,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='fw_l1',
            bound=bound,
            **kwargs,
        )


class JPEGLinfAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 0.25,
                'imagenet100': 0.5,
                'imagenet': 0.5,
                'bird_or_bicycle': 0.5,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='jpeg_linf',
            bound=bound,
            **kwargs,
        )


class FogAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=512, **kwargs):
        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='fog',
            bound=bound,
            **kwargs,
        )


class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )


class ReColorAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.06, **kwargs):
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(pt.ReColorAdv, {
                'xform_class': ct.FullSpatial,
                'cspace': cs.CIELUVColorSpace(),
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_params': {
                  'resolution_x': 16,
                  'resolution_y': 32,
                  'resolution_z': 32,
                },
                'use_smooth_loss': True,
            }),
            perturbation_norm_loss=0.0036 / bound,
            **kwargs,
        )


class AutoAttack(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        self.model = model
        self.kwargs = kwargs
        self.attack = None

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            try:
                import autoattack
            except ImportError:
                raise RuntimeError(
                    'Error: unable to import autoattack. Please install the '
                    'package by running '
                    '"pip install git+git://github.com/fra31/auto-attack#egg=autoattack".'
                )
            self.attack = autoattack.AutoAttack(
                self.model, device=inputs.device, **self.kwargs)

        return self.attack.run_standard_evaluation(inputs, labels)


class AutoLinfAttack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 8/255,
                'imagenet100': 8/255,
                'imagenet': 8/255,
                'bird_or_bicycle': 16/255,
            }[dataset_name]

        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            **kwargs,
        )


class AutoL2Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'cifar': 1,
                'imagenet100': 3,
                'imagenet': 3,
                'bird_or_bicycle': 10,
            }[dataset_name]

        super().__init__(
            model,
            norm='L2',
            eps=bound,
            **kwargs,
        )