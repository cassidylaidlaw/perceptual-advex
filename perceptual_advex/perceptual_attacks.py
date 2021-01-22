from typing import Optional, Union
import torch
import torchvision.models as torchvision_models
from torchvision.models.utils import load_state_dict_from_url
import math
from torch import nn
from torch.nn import functional as F
from typing_extensions import Literal

from .distances import normalize_flatten_features, LPIPSDistance
from .utilities import MarginLoss
from .models import AlexNetFeatureModel, CifarAlexNet, FeatureModel
from . import utilities


_cached_alexnet: Optional[AlexNetFeatureModel] = None
_cached_alexnet_cifar: Optional[AlexNetFeatureModel] = None


def get_lpips_model(
    lpips_model_spec: Union[
        Literal['self', 'alexnet', 'alexnet_cifar'],
        FeatureModel,
    ],
    model: FeatureModel,
) -> FeatureModel:
    global _cached_alexnet, _cached_alexnet_cifar

    lpips_model: FeatureModel

    if lpips_model_spec == 'self':
        return model
    elif lpips_model_spec == 'alexnet':
        if _cached_alexnet is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            _cached_alexnet = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model_spec == 'alexnet_cifar':
        if _cached_alexnet_cifar is None:
            alexnet_model = CifarAlexNet()
            _cached_alexnet_cifar = AlexNetFeatureModel(alexnet_model)
        lpips_model = _cached_alexnet_cifar
        if torch.cuda.is_available():
            lpips_model.cuda()
        try:
            state = torch.load('data/checkpoints/alexnet_cifar.pt')
        except FileNotFoundError:
            state = load_state_dict_from_url(
                'https://perceptual-advex.s3.us-east-2.amazonaws.com/'
                'alexnet_cifar.pt',
                progress=True,
            )
        lpips_model.load_state_dict(state['model'])
    elif isinstance(lpips_model_spec, str):
        raise ValueError(f'Invalid LPIPS model "{lpips_model_spec}"')
    else:
        lpips_model = lpips_model_spec

    lpips_model.eval()
    return lpips_model


class FastLagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 lam=10, h=1e-1, lpips_model='self', decay_step_size=True,
                 increase_lambda=True, projection='none', kappa=math.inf,
                 include_image_as_activation=False, randomize=False):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.

        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        if step is None:
            self.step = self.bound
        else:
            self.step = step
        self.num_iterations = num_iterations
        self.lam = lam
        self.h = h
        self.decay_step_size = decay_step_size
        self.increase_lambda = increase_lambda

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)
        self.loss = MarginLoss(kappa=kappa)

    def forward(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)

        perturbations.requires_grad = True

        step_size = self.step
        lam = self.lam

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        for attack_iter in range(self.num_iterations):
            # Decay step size, but increase lambda over time.
            if self.decay_step_size:
                step_size = \
                    self.step * 0.1 ** (attack_iter / self.num_iterations)
            if self.increase_lambda:
                lam = \
                    self.lam * 0.1 ** (1 - attack_iter / self.num_iterations)

            if perturbations.grad is not None:
                perturbations.grad.data.zero_()

            adv_inputs = inputs + perturbations

            if self.model == self.lpips_model:
                adv_features, adv_logits = \
                    self.model.features_logits(adv_inputs)
            else:
                adv_features = self.lpips_model.features(adv_inputs)
                adv_logits = self.model(adv_inputs)

            adv_loss = self.loss(adv_logits, labels)

            adv_features = normalize_flatten_features(adv_features)
            lpips_dists = (adv_features - input_features).norm(dim=1)

            loss = -adv_loss + lam * F.relu(lpips_dists - self.bound)
            loss.sum().backward()

            grad = perturbations.grad.data
            grad_normed = grad / \
                (grad.reshape(grad.size()[0], -1).norm(dim=1)
                 [:, None, None, None] + 1e-8)

            dist_grads = (
                adv_features -
                normalize_flatten_features(self.lpips_model.features(
                    inputs + perturbations + grad_normed * self.h))
            ).norm(dim=1) / 0.1

            perturbation_updates = -grad_normed * (
                step_size / (dist_grads + 1e-4)
            )[:, None, None, None]

            perturbations.data = (
                (inputs + perturbations + perturbation_updates).clamp(0, 1) -
                inputs
            ).detach()

        adv_inputs = (inputs + perturbations).detach()
        return self.projection(inputs, adv_inputs, input_features).detach()


class NoProjection(nn.Module):
    def __init__(self, bound, lpips_model):
        super().__init__()

    def forward(self, inputs, adv_inputs, input_features=None):
        return adv_inputs


class BisectionPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, num_steps=10):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.num_steps = num_steps

    def forward(self, inputs, adv_inputs, input_features=None):
        batch_size = inputs.shape[0]
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        lam_min = torch.zeros(batch_size, device=inputs.device)
        lam_max = torch.ones(batch_size, device=inputs.device)
        lam = 0.5 * torch.ones(batch_size, device=inputs.device)

        for _ in range(self.num_steps):
            projected_adv_inputs = (
                inputs * (1 - lam[:, None, None, None]) +
                adv_inputs * lam[:, None, None, None]
            )
            adv_features = self.lpips_model.features(projected_adv_inputs)
            adv_features = normalize_flatten_features(adv_features).detach()
            diff_features = adv_features - input_features
            norm_diff_features = torch.norm(diff_features, dim=1)

            lam_max[norm_diff_features > self.bound] = \
                lam[norm_diff_features > self.bound]
            lam_min[norm_diff_features <= self.bound] = \
                lam[norm_diff_features <= self.bound]
            lam = 0.5*(lam_min + lam_max)
        return projected_adv_inputs.detach()


class NewtonsPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, projection_overshoot=1e-1,
                 max_iterations=10):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.projection_overshoot = projection_overshoot
        self.max_iterations = max_iterations
        self.bisection_projection = BisectionPerceptualProjection(
            bound, lpips_model)

    def forward(self, inputs, adv_inputs, input_features=None):
        original_adv_inputs = adv_inputs
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        needs_projection = torch.ones_like(adv_inputs[:, 0, 0, 0]) \
            .bool()

        needs_projection.requires_grad = False
        iteration = 0
        while needs_projection.sum() > 0 and iteration < self.max_iterations:
            adv_inputs.requires_grad = True
            adv_features = normalize_flatten_features(
                self.lpips_model.features(adv_inputs[needs_projection]))
            adv_lpips = (input_features[needs_projection] -
                         adv_features).norm(dim=1)
            adv_lpips.sum().backward()

            projection_step_size = (adv_lpips - self.bound) \
                .clamp(min=0)
            projection_step_size[projection_step_size > 0] += \
                self.projection_overshoot

            grad_norm = adv_inputs.grad.data[needs_projection] \
                .view(needs_projection.sum(), -1).norm(dim=1)
            inverse_grad = adv_inputs.grad.data[needs_projection] / \
                grad_norm[:, None, None, None] ** 2

            adv_inputs.data[needs_projection] = (
                adv_inputs.data[needs_projection] -
                projection_step_size[:, None, None, None] *
                (1 + self.projection_overshoot) *
                inverse_grad
            ).clamp(0, 1).detach()

            needs_projection[needs_projection] = \
                projection_step_size > 0
            iteration += 1

        if needs_projection.sum() > 0:
            # If we still haven't projected all inputs after max_iterations,
            # just use the bisection method.
            adv_inputs = self.bisection_projection(
                inputs, original_adv_inputs, input_features)

        return adv_inputs.detach()


PROJECTIONS = {
    'none': NoProjection,
    'linesearch': BisectionPerceptualProjection,
    'bisection': BisectionPerceptualProjection,
    'gradient': NewtonsPerceptualProjection,
    'newtons': NewtonsPerceptualProjection,
}


class FirstOrderStepPerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, num_iterations=5,
                 h=1e-3, kappa=1, lpips_model='self',
                 targeted=False, randomize=False,
                 include_image_as_activation=False):
        """
        Perceptual attack using conjugate gradient to solve the constrained
        optimization problem.

        bound is the (approximate) bound on the LPIPS distance.
        num_iterations is the number of CG iterations to take.
        h is the step size to use for finite-difference calculation.
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.h = h

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = MarginLoss(kappa=kappa, targeted=targeted)

    def _multiply_matrix(self, v):
        """
        If (D phi) is the Jacobian of the features function for the model
        at inputs, then approximately calculates
            (D phi)T (D phi) v
        """

        self.inputs.grad.data.zero_()

        with torch.no_grad():
            v_features = self.lpips_model.features(self.inputs.detach() +
                                                   self.h * v)
            D_phi_v = (
                normalize_flatten_features(v_features) -
                self.input_features
            ) / self.h

        torch.sum(self.input_features * D_phi_v).backward(retain_graph=True)

        return self.inputs.grad.data.clone()

    def forward(self, inputs, labels):
        self.inputs = inputs

        inputs.requires_grad = True
        if self.model == self.lpips_model:
            input_features, orig_logits = self.model.features_logits(inputs)
        else:
            input_features = self.lpips_model.features(inputs)
            orig_logits = self.model(inputs)
        self.input_features = normalize_flatten_features(input_features)

        loss = self.loss(orig_logits, labels)
        loss.sum().backward(retain_graph=True)

        inputs_grad = inputs.grad.data.clone()
        if inputs_grad.abs().max() < 1e-4:
            return inputs

        # Variable names are from
        # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
        x = torch.zeros_like(inputs)
        r = inputs_grad - self._multiply_matrix(x)
        p = r

        for cg_iter in range(self.num_iterations):
            r_last = r
            p_last = p
            x_last = x
            del r, p, x

            r_T_r = (r_last ** 2).sum(dim=[1, 2, 3])
            if r_T_r.max() < 1e-1 and cg_iter > 0:
                # If the residual is small enough, just stop the algorithm.
                x = x_last
                break

            A_p_last = self._multiply_matrix(p_last)

            # print('|r|^2 =', ' '.join(f'{z:.2f}' for z in r_T_r))
            alpha = (
                r_T_r /
                (p_last * A_p_last).sum(dim=[1, 2, 3])
            )[:, None, None, None]
            x = x_last + alpha * p_last

            # These calculations aren't necessary on the last iteration.
            if cg_iter < self.num_iterations - 1:
                r = r_last - alpha * A_p_last

                beta = (
                    (r ** 2).sum(dim=[1, 2, 3]) /
                    r_T_r
                )[:, None, None, None]
                p = r + beta * p_last

        x_features = self.lpips_model.features(self.inputs.detach() +
                                               self.h * x)
        D_phi_x = (
            normalize_flatten_features(x_features) -
            self.input_features
        ) / self.h

        lam = (self.bound / D_phi_x.norm(dim=1))[:, None, None, None]

        inputs_grad_norm = inputs_grad.reshape(
            inputs_grad.size()[0], -1).norm(dim=1)
        # If the grad is basically 0, don't perturb that input. It's likely
        # already misclassified, and trying to perturb it further leads to
        # numerical instability.
        lam[inputs_grad_norm < 1e-4] = 0
        x[inputs_grad_norm < 1e-4] = 0

        # print('LPIPS', self.lpips_distance(
        #    inputs,
        #    inputs + lam * x,
        # ))

        return (inputs + lam * x).clamp(0, 1).detach()


class PerceptualPGDAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=5,
                 cg_iterations=5, h=1e-3, lpips_model='self',
                 decay_step_size=False, kappa=1,
                 projection='newtons', randomize=False,
                 random_targets=False, num_classes=None,
                 include_image_as_activation=False):
        """
        Iterated version of the conjugate gradient attack.

        step_size is the step size in LPIPS distance.
        num_iterations is the number of steps to take.
        cg_iterations is the conjugate gradient iterations per step.
        h is the step size to use for finite-difference calculation.
        project is whether or not to project the perturbation into the LPIPS
            ball after each step.
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.decay_step_size = decay_step_size
        self.step = step
        self.random_targets = random_targets
        self.num_classes = num_classes

        if self.step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = 2 * self.bound / self.num_iterations

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.first_order_step = FirstOrderStepPerceptualAttack(
            model, bound=self.step, num_iterations=cg_iterations, h=h,
            kappa=kappa, lpips_model=self.lpips_model,
            include_image_as_activation=include_image_as_activation,
            targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def _attack(self, inputs, labels):
        with torch.no_grad():
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        start_perturbations = torch.zeros_like(inputs)
        start_perturbations.normal_(0, 0.01)
        adv_inputs = inputs + start_perturbations
        for attack_iter in range(self.num_iterations):
            if self.decay_step_size:
                step_size = self.step * \
                    0.1 ** (attack_iter / self.num_iterations)
                self.first_order_step.bound = step_size
            adv_inputs = self.first_order_step(adv_inputs, labels)
            adv_inputs = self.projection(inputs, adv_inputs, input_features)

        # print('LPIPS', self.first_order_step.lpips_distance(
        #    inputs,
        #    adv_inputs,
        # ))

        return adv_inputs

    def forward(self, inputs, labels):
        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                self._attack,
                self.model,
                inputs,
                labels,
                self.num_classes,
            )
        else:
            return self._attack(inputs, labels)

class LagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 binary_steps=5, h=0.1, kappa=1, lpips_model='self',
                 projection='newtons', decay_step_size=True,
                 num_classes=None,
                 include_image_as_activation=False,
                 randomize=False, random_targets=False):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.
        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.decay_step_size = decay_step_size
        self.num_iterations = num_iterations
        if step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = self.bound * 2 / self.num_iterations
        else:
            self.step = step
        self.binary_steps = binary_steps
        self.h = h
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(
            self.lpips_model,
            include_image_as_activation=include_image_as_activation,
        )
        self.loss = MarginLoss(kappa=kappa, targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        return self.lpips_distance(inputs, adv_inputs) <= self.bound

    def _attack(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)
        perturbations.requires_grad = True

        batch_size = inputs.shape[0]
        step_size = self.step

        lam = 0.01 * torch.ones(batch_size, device=inputs.device)

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        live = torch.ones(batch_size, device=inputs.device, dtype=torch.bool)

        for binary_iter in range(self.binary_steps):
            for attack_iter in range(self.num_iterations):
                if self.decay_step_size:
                    step_size = self.step * \
                        (0.1 ** (attack_iter / self.num_iterations))
                else:
                    step_size = self.step

                if perturbations.grad is not None:
                    perturbations.grad.data.zero_()

                adv_inputs = (inputs + perturbations)[live]

                if self.model == self.lpips_model:
                    adv_features, adv_logits = \
                        self.model.features_logits(adv_inputs)
                else:
                    adv_features = self.lpips_model.features(adv_inputs)
                    adv_logits = self.model(adv_inputs)

                adv_labels = adv_logits.argmax(1)
                adv_loss = self.loss(adv_logits, labels[live])
                adv_features = normalize_flatten_features(adv_features)
                lpips_dists = (adv_features - input_features[live]).norm(dim=1)
                all_lpips_dists = torch.zeros(batch_size, device=inputs.device)
                all_lpips_dists[live] = lpips_dists

                loss = -adv_loss + lam[live] * F.relu(lpips_dists - self.bound)
                loss.sum().backward()

                grad = perturbations.grad.data[live]
                grad_normed = grad / \
                    (grad.reshape(grad.size()[0], -1).norm(dim=1)
                     [:, None, None, None] + 1e-8)

                dist_grads = (
                    adv_features -
                    normalize_flatten_features(self.lpips_model.features(
                        adv_inputs + grad_normed * self.h))
                ).norm(dim=1) / self.h

                updates = -grad_normed * (
                    step_size / (dist_grads + 1e-8)
                )[:, None, None, None]

                perturbations.data[live] = (
                    (inputs[live] + perturbations[live] +
                     updates).clamp(0, 1) -
                    inputs[live]
                ).detach()

                if self.random_targets:
                    live[live] = (adv_labels != labels[live]) | (lpips_dists > self.bound)
                else:
                    live[live] = (adv_labels == labels[live]) | (lpips_dists > self.bound)
                if live.sum() == 0:
                    break

            lam[all_lpips_dists >= self.bound] *= 10
            if live.sum() == 0:
                break

        adv_inputs = (inputs + perturbations).detach()
        adv_inputs = self.projection(inputs, adv_inputs, input_features)
        return adv_inputs

    def forward(self, inputs, labels):
        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                self._attack,
                self.model,
                inputs,
                labels,
                self.num_classes,
            )
        else:
            return self._attack(inputs, labels)
