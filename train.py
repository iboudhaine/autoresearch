"""
Model architecture, Muon optimizer, training loop.
"""

import time
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from prepare import get_train_loader, evaluate_model, TIME_BUDGET


#############################################
#               Muon optimizer              #
#############################################


def _zeropower_via_newtonschulz5(
    gradients_4d: list[torch.half],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    progress: float,
) -> list[torch.half]:
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-05
    eps_gms = 1e-05

    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = initial_target_mag * (1 - progress) + final_target_mag * progress

    # Use stack instead of pre-allocated tensor for better performance
    if not filter_meta_data:
        return gradients_4d

    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        grad_to_orthogonalize = gradients_4d[list_idx]
        g_reshaped = grad_to_orthogonalize.reshape(reshaped_D, reshaped_K)
        padding_dims = (0, max_K - reshaped_K, 0, max_D - reshaped_D)
        g_padded = F.pad(g_reshaped, padding_dims, "constant", 0)
        grad_list.append(g_padded)

    if not grad_list:
        return gradients_4d

    X = torch.stack(grad_list)

    # Fuse normalization operations for better performance
    current_batch_mags = X.norm(dim=(1, 2), keepdim=True)
    scale_factor = target_magnitude / (current_batch_mags + eps_gms)
    X = X * scale_factor

    X_norm = X.norm(dim=(1, 2), keepdim=True)
    X = X / (X_norm + eps_stable)

    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    # Unroll the loop for better performance
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    if transposed:
        X = X.transpose(1, 2)

    final_orthogonalized_grads_list = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        orthogonalized_g_padded = X[i]
        orthogonalized_g_reshaped = orthogonalized_g_padded[:reshaped_D, :reshaped_K]
        final_orthogonalized_grads_list[list_idx] = orthogonalized_g_reshaped.view(
            original_shape
        )
    return final_orthogonalized_grads_list


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.08,
        momentum=0.88,
        nesterov=True,
        norm_freq=1,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.progress = 0.0
        self.filter_params_meta = []
        self.max_D, self.max_K = (0, 0)
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // p.shape[0]
                    self.filter_params_meta.append(
                        {
                            "param": p,
                            "original_shape": p.data.shape,
                            "reshaped_dims": (reshaped_D, reshaped_K),
                        }
                    )
                    self.max_D = max(self.max_D, reshaped_D)
                    self.max_K = max(self.max_K, reshaped_K)
        self.max_D = max(1, self.max_D)
        self.max_K = (max(1, self.max_K) + 15) // 16 * 16
        self.current_grad_norms = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        group["norm_freq"] = 2 + int(15 * self.progress)
        # Prepare momentum buffers and track meta data
        filter_params_with_grad = []
        filter_meta_for_current_step = []
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None

        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is not None:
                filter_params_with_grad.append(p)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad,
                        dtype=group["momentum_buffer_dtype"],
                        memory_format=torch.preserve_format,
                    )
                if momentum_buffers is not None:
                    momentum_buffers.append(state["momentum_buffer"])
                filter_meta_for_current_step.append(
                    (
                        p_meta["original_shape"],
                        p_meta["reshaped_dims"][0],
                        p_meta["reshaped_dims"][1],
                        len(filter_params_with_grad)
                        - 1,  # Index in filter_params_with_grad
                    )
                )

        if not filter_params_with_grad:
            return

        # Apply momentum and add gradients
        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [
                g.to(mb.dtype)
                for g, mb in zip(
                    [p.grad for p in filter_params_with_grad], momentum_buffers
                )
            ]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]

        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad],
                momentum_buffers,
                alpha=group["momentum"],
            )
        else:
            nesterov_grads = momentum_buffers

        do_norm_scaling = self.step_count - self.last_norm_step >= group["norm_freq"]
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            self.current_grad_norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [
                (len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                for p, n in zip(filter_params_with_grad, self.current_grad_norms)
            ]

        final_orthogonalized_grads = _zeropower_via_newtonschulz5(
            nesterov_grads,
            filter_meta_for_current_step,
            self.max_D,
            self.max_K,
            self.progress,
        )

        # Apply updates in a single fused operation when possible
        if do_norm_scaling:
            # Scale gradients first
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
            # Then apply the orthogonalized updates
            torch._foreach_add_(
                filter_params_with_grad, final_orthogonalized_grads, alpha=-group["lr"]
            )
        else:
            # Apply optimizer step directly
            torch._foreach_add_(
                filter_params_with_grad, final_orthogonalized_grads, alpha=-group["lr"]
            )

        # Apply weight decay in a fused operation
        weight_decay_factor = 1 - group["lr"] * group["weight_decay"]
        if weight_decay_factor != 1.0:
            torch._foreach_mul_(filter_params_with_grad, weight_decay_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


#############################################
#            Network Definition             #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.half()
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        w = self.head.weight.data
        w.mul_(1.0 / w.std())

    def init_whiten(self, train_images, eps=0.0005):
        c, (h, w) = (train_images.shape[1], self.whiten.weight.shape[2:])
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        # Use more efficient covariance computation with SVD for better numerical stability
        est_patch_covariance = torch.mm(patches_flat.t(), patches_flat) / len(
            patches_flat
        )
        U, S, V = torch.svd(est_patch_covariance)
        # More stable inverse square root computation
        inv_sqrt_S = torch.rsqrt(S + eps)
        eigenvectors_scaled = (U * inv_sqrt_S.unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1).contiguous()
        return self.head(x) / x.size(-1)


############################################
#                Training                  #
############################################


def train():
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model = torch.compile(model)
    torch.cuda.reset_peak_memory_stats()
    training_batch_size = 1536
    bias_lr = 0.0573
    head_lr = 0.5415
    wd = 1.0418e-06 * training_batch_size

    train_loader = get_train_loader(training_batch_size)

    whiten_bias_train_steps = ceil(0.2 * len(train_loader))
    model.reset()

    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.825, nesterov=True, fused=True
    )
    optimizer2 = Muon(
        filter_params,
        lr=0.205,
        momentum=0.655,
        nesterov=True,
        norm_freq=4,
        weight_decay=wd,
    )
    optimizer2.param_groups[0]["momentum_buffer_dtype"] = torch.half
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    wall_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal wall_seconds
        wall_seconds += 1e-3 * starter.elapsed_time(ender)

    step = 0
    start_timer()
    with torch.no_grad():
        train_images = train_loader.normalize(train_loader.images[:960])
        model.init_whiten(train_images)

    # Precompute LR factors to reduce computation in training loop
    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)

    # Precompute some values to reduce computation in training loop
    lr_factor1_initial = optimizer1.param_groups[0]["initial_lr"]
    lr_factors2_initial = [
        group["initial_lr"]
        for group in optimizer1.param_groups[1:] + optimizer2.param_groups
    ]

    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(outputs, labels, label_smoothing=0.09, reduction="sum")
        return loss

    training_seconds = 0.0
    t_prev = None
    done = False
    while not done:
        ####################
        #     Training     #
        ####################
        model.train()
        for inputs, labels in train_loader:
            # Determine if we should train whiten bias
            whiten_bias_grad = step < whiten_bias_train_steps

            # Execute training step
            loss = forward_step(inputs, labels, whiten_bias_grad)
            loss.backward()

            # Time-based progress drives the main LR schedule and Muon ramps
            progress = min(training_seconds / TIME_BUDGET, 1.0)
            lr_factor1 = max(0.0, 1 - step * lr_factor1_base)
            lr_factor2 = 1 - progress

            # Apply learning rates in a fused way
            optimizer1.param_groups[0]["lr"] = lr_factor1_initial * lr_factor1
            for i, group in enumerate(
                optimizer1.param_groups[1:] + optimizer2.param_groups
            ):
                group["lr"] = lr_factors2_initial[i] * lr_factor2

            # Muon reads progress from this attribute
            optimizer2.progress = progress

            # Optimizer steps
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            # Accumulate wall-clock for the budget, skipping the first 10 steps
            # to exclude compile/autotune overhead.
            torch.cuda.synchronize()
            t_now = time.perf_counter()
            if t_prev is not None and step > 10:
                training_seconds += t_now - t_prev
            t_prev = t_now

            step += 1
            if step % 10 == 0:
                print(
                    f"step {step:4d} | training_seconds {training_seconds:6.1f} "
                    f"| progress {progress:7.2%} | lr2 {lr_factor2:.3f} "
                    f"| loss {loss.item() / len(inputs):.3f}"
                )
            if step > 10 and training_seconds >= TIME_BUDGET:
                done = True
                break

    stop_timer()
    return model, wall_seconds, training_seconds, step


if __name__ == "__main__":
    model, wall_seconds, training_seconds, num_steps = train()
    acc = evaluate_model(model)
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print("---")
    print(f"training_seconds: {training_seconds:.4f}")
    print(f"wall_seconds: {wall_seconds:.4f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"num_steps: {num_steps}")
    print(f"num_params_M: {num_params_m:.2f}")
    print(f"tta_val_acc: {acc:.6f}")
