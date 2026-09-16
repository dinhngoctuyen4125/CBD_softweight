"""
CBD-DFB Trainer
Project gradients onto discriminative subspace Q (generalized eigenvectors).
"""

import math
import torch
from typing import Callable, Dict, Optional
from .hf_trainers import ForgetTrainer


class CBDDFBForgetTrainer(ForgetTrainer):
    """
    Use CBD-DFB subspace Q to project gradients during training:
        g <- Q Q^T g
    """

    def __init__(
        self,
        model,
        train_loss_function: Callable,
        cbd_dfb_basis_path: str,
        enable_cbd_dfb: bool = True,
        use_eigval_weight: bool = False,
        trust_region: bool = False,
        trust_region_epsilon: float = 1e-3,
        trust_region_delta: float = 1e-12,
        project_forget_only: bool = False,
        **kwargs,
    ):
        super().__init__(model=model, train_loss_function=train_loss_function, **kwargs)
        self.enable_cbd_dfb = enable_cbd_dfb
        self.use_eigval_weight = bool(use_eigval_weight)
        self.trust_region = bool(trust_region)
        self.trust_region_epsilon = float(trust_region_epsilon)
        self.trust_region_delta = float(trust_region_delta)
        self.project_forget_only = bool(project_forget_only)
        self._basis_device_cache = {}
        self._basis_key_cache = {}
        self._project_param_bindings = None
        if self.enable_cbd_dfb:
            print(f"\U0001f9ed 启用 CBD-DFB 梯度投影，基底路径: {cbd_dfb_basis_path}")
            self.csm_basis = self._load_csm_basis(cbd_dfb_basis_path)
            print(f"\u2705 成功加载 {len(self.csm_basis)} 层 CBD-DFB 基底")
        else:
            self.csm_basis = None

    def _load_csm_basis(self, basis_path: str) -> Dict:
        import os
        import pickle

        if not os.path.isabs(basis_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            possible_paths = [
                basis_path,
                os.path.join(os.getcwd(), basis_path),
                os.path.join(repo_root, basis_path),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    basis_path = path
                    break
            else:
                raise FileNotFoundError(f"无法找到 CBD-DFB 基底文件: {possible_paths}")

        print(f"\U0001f50d 从路径加载 CBD-DFB 基底: {os.path.abspath(basis_path)}")
        with open(basis_path, 'rb') as f:
            basis_dict = pickle.load(f)

        processed = {}
        for layer_name, info in basis_dict.items():
            raw = info.get('components')
            if torch.is_tensor(raw):
                components = raw.detach().to(dtype=torch.float32, device='cpu').contiguous()
            else:
                components = torch.as_tensor(raw, dtype=torch.float32).contiguous()

            eigvals_raw = info.get('eigvals')
            if eigvals_raw is None:
                eigvals = None
            elif torch.is_tensor(eigvals_raw):
                eigvals = eigvals_raw.detach().to(dtype=torch.float32, device='cpu').contiguous()
            else:
                eigvals = torch.as_tensor(eigvals_raw, dtype=torch.float32).contiguous()
            processed[layer_name] = {
                'components': components,  # [k, feature_dim]
                'n_components': info.get('n_components', components.shape[0]),
                'feature_dim': components.shape[1],
                'eigvals': eigvals,  # [k] or None
                # Optional trust-region helpers
                'retain_proj': (
                    info.get('retain_proj').detach().to(dtype=torch.float32, device='cpu').contiguous()
                    if torch.is_tensor(info.get('retain_proj'))
                    else (
                        torch.as_tensor(info.get('retain_proj'), dtype=torch.float32).contiguous()
                        if info.get('retain_proj') is not None
                        else None
                    )
                ),  # [n_r, k] or None
                'n_retain': int(info.get('n_retain')) if info.get('n_retain') is not None else None,
            }
        return processed

    def _resolve_basis_key(self, layer_name: str) -> Optional[str]:
        cached = self._basis_key_cache.get(layer_name)
        if cached is not None:
            return cached
        candidates = [layer_name]
        if layer_name.startswith("base_model.model."):
            candidates.append(layer_name.replace("base_model.model.", "", 1))
        if layer_name.startswith("base_model.model.model."):
            candidates.append(layer_name.replace("base_model.model.model.", "model.", 1))
        if ".lora_B.default" in layer_name:
            candidates.append(layer_name.replace(".lora_B.default", ""))
        if ".lora_B" in layer_name:
            candidates.append(layer_name.replace(".lora_B", ""))

        for key in candidates:
            if key in self.csm_basis:
                self._basis_key_cache[layer_name] = key
                return key
        self._basis_key_cache[layer_name] = None
        return None

    def _iter_projectable_params(self):
        if self._project_param_bindings is None:
            bindings = []
            for name, param in self.model.named_parameters():
                if 'lora_B' not in name or 'up_proj' not in name or 'default' not in name:
                    continue
                if param.dim() != 2:
                    continue
                basis_key = self._resolve_basis_key(name)
                if basis_key is None:
                    continue
                bindings.append((name, param, basis_key))
            self._project_param_bindings = tuple(bindings)
        return self._project_param_bindings

    def _project_gradient_csm(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enable_cbd_dfb or self.csm_basis is None:
            return gradients

        projected = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            if 'lora_B' not in name or 'up_proj' not in name or 'default' not in name:
                projected[name] = grad
                continue

            basis_key = self._resolve_basis_key(name)
            if basis_key is None:
                projected[name] = grad
                continue

            Q_T, Q, w, _, _ = self._get_cached_basis(basis_key, device=grad.device)

            if grad.dim() != 2:
                projected[name] = grad
                continue

            # Flatten [out_dim, r] -> [d], project, then reshape back.
            g = grad
            gvec = g.reshape(-1)
            coeff = torch.matmul(Q_T, gvec)  # [k]
            if w is not None:
                coeff = coeff * w
            proj_vec = torch.matmul(Q, coeff)  # [d]
            projected[name] = proj_vec.reshape_as(g)

        return projected

    def _get_cached_basis(self, basis_key: str, device: torch.device):
        cache_key = (basis_key, str(device))
        cached = self._basis_device_cache.get(cache_key)
        if cached is not None:
            return cached

        basis_info = self.csm_basis[basis_key]
        Q_T_cpu = basis_info['components']  # [k, d] on CPU float32
        Q_T = Q_T_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        Q = Q_T.t().contiguous()  # [d, k] float32

        w = None
        if self.use_eigval_weight:
            eigvals_cpu = basis_info.get('eigvals')
            if eigvals_cpu is not None and eigvals_cpu.numel() == Q_T_cpu.shape[0]:
                # Normalize weights to keep average magnitude ~1.
                w_cpu = eigvals_cpu.clamp(min=0).sqrt()
                w_cpu = w_cpu / (w_cpu.mean() + 1e-12)
                w = w_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

        retain_proj = None
        n_retain = None
        if self.trust_region:
            retain_proj_cpu = basis_info.get('retain_proj')
            if retain_proj_cpu is not None:
                retain_proj = retain_proj_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                n_retain = basis_info.get('n_retain')

        self._basis_device_cache[cache_key] = (Q_T, Q, w, retain_proj, n_retain)
        return Q_T, Q, w, retain_proj, n_retain

    def _apply_csm_projection(self):
        if not self.enable_cbd_dfb or self.csm_basis is None:
            return
        lr = float(getattr(self.args, "learning_rate", 0.0) or 0.0)
        trust_term = None  # torch scalar on device
        trust_n = None
        grads_to_scale = []
        for name, param, basis_key in self._iter_projectable_params():
            grad = param.grad
            if grad is None:
                continue
            if grad.dim() != 2:
                continue
            Q_T, Q, w, retain_proj, n_retain = self._get_cached_basis(basis_key, device=grad.device)
            gvec = grad.reshape(-1).to(dtype=torch.float32)
            coeff = torch.matmul(Q_T, gvec)  # [k] float32
            if w is not None:
                coeff = coeff * w
            proj_vec = torch.matmul(Q, coeff)  # [d]
            if self.trust_region and retain_proj is not None and n_retain:
                # ||G_r^T g_proj||^2 where g_proj = Q coeff and G_r^T Q = retain_proj
                rc = torch.matmul(retain_proj, coeff)  # [n_r]
                term = (rc * rc).sum()
                trust_term = term if trust_term is None else (trust_term + term)
                trust_n = int(n_retain) if trust_n is None else min(trust_n, int(n_retain))

            grad.copy_(proj_vec.to(dtype=grad.dtype).reshape_as(grad))
            grads_to_scale.append(grad)

        scale, term_val = self._compute_trust_scale(trust_term, trust_n, lr)
        if scale < 1.0:
            for g in grads_to_scale:
                g.mul_(scale)
        self._maybe_log_trust_stats(scale, term_val, trust_n)

    def _compute_trust_scale(self, trust_term, trust_n, lr: float) -> tuple[float, Optional[float]]:
        # Optional: retain trust-region step (方法描述.md §6.3)
        if (
            not self.trust_region
            or trust_term is None
            or trust_n is None
            or trust_n <= 0
            or lr <= 0
        ):
            return 1.0, None
        term_val = float(trust_term.detach().cpu().item())
        denom = (lr * lr) * (term_val / float(trust_n))
        scale = 1.0
        if denom > 0:
            scale = min(
                1.0,
                math.sqrt((2.0 * self.trust_region_epsilon) / (denom + self.trust_region_delta)),
            )
        return scale, term_val

    def _maybe_log_trust_stats(self, scale: float, term_val: Optional[float], trust_n):
        if (
            term_val is None
            or trust_n is None
            or getattr(self, "state", None) is None
            or getattr(self.state, "global_step", 0) % 50 != 0
        ):
            return
        try:
            self.log({
                "cbd_dfb/trust_scale": float(scale),
                "cbd_dfb/trust_term": float(term_val),
                "cbd_dfb/trust_n": float(trust_n),
            })
        except Exception:
            pass

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Default: project the total gradient (as described in 方法描述.md §6.2).
        # Optional variant (project_forget_only): only project forget gradient, keep retain KL gradient unprojected.
        # This helps retain KL "pull back" using full parameter space when QQ^T would remove it.
        if self.project_forget_only and hasattr(self.train_loss_function, "__call__"):
            losses = self.train_loss_function(model, inputs, self.oracle_model)
            loss = losses["loss"]
            forget_loss = losses.get("forget_loss")
            retain_loss = losses.get("retain_loss")

            forget_weight = float(getattr(self.train_loss_function, "forget_weight", 1.0) or 1.0)
            retain_weight = float(getattr(self.train_loss_function, "retain_weight", 1.0) or 1.0)

            forget_obj = (forget_loss * forget_weight) if forget_loss is not None else None
            retain_obj = (retain_loss * retain_weight) if retain_loss is not None else None

            # Mirror ForgetTrainer.compute_loss logging, but avoid per-microbatch log spam.
            if self._should_log_trainloss():
                try:
                    logitems = {"trainloss/loss": float(loss.detach().cpu().item())}
                    if forget_loss is not None:
                        logitems["trainloss/forgetloss"] = float(forget_loss.detach().cpu().item())
                    if retain_loss is not None:
                        logitems["trainloss/retainloss"] = float(retain_loss.detach().cpu().item())
                    self.log(logitems)
                except Exception:
                    pass
        else:
            loss = self.compute_loss(model, inputs)
            forget_obj = None
            retain_obj = None

        if self.args.n_gpu > 1:
            loss = loss.mean()

        deepspeed = getattr(self, "deepspeed", None)
        if self.args.gradient_accumulation_steps > 1 and not deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
            if forget_obj is not None:
                forget_obj = forget_obj / self.args.gradient_accumulation_steps
            if retain_obj is not None:
                retain_obj = retain_obj / self.args.gradient_accumulation_steps

        if self.project_forget_only and forget_obj is not None and retain_obj is not None and not deepspeed:
            lr = float(getattr(self.args, "learning_rate", 0.0) or 0.0)
            do_forget = bool(getattr(forget_obj, "requires_grad", False))
            do_retain = bool(getattr(retain_obj, "requires_grad", False))

            # Some micro-batches can have no supervised tokens (e.g. all labels=-100),
            # which makes the corresponding loss a constant tensor without grad_fn.
            # Skip the backward for that component instead of crashing.
            if do_forget:
                # 1) Backward forget loss
                # Save current accumulated grads for projected params to avoid re-projecting previous micro-steps.
                saved = {}
                for name, param, _ in self._iter_projectable_params():
                    if param.grad is None:
                        continue
                    saved[name] = param.grad.detach().clone()

                forget_obj.backward()

                # Project only the newly-added forget gradients (delta = grad - saved)
                projected_deltas = []
                trust_term = None
                trust_n = None
                for name, param, basis_key in self._iter_projectable_params():
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.dim() != 2:
                        continue

                    old = saved.get(name)
                    delta = grad if old is None else (grad - old)

                    Q_T, Q, w, retain_proj, n_retain = self._get_cached_basis(basis_key, device=delta.device)
                    gvec = delta.reshape(-1).to(dtype=torch.float32)
                    coeff = torch.matmul(Q_T, gvec)
                    if w is not None:
                        coeff = coeff * w
                    proj_vec = torch.matmul(Q, coeff).to(dtype=delta.dtype).reshape_as(delta)

                    if self.trust_region and retain_proj is not None and n_retain:
                        rc = torch.matmul(retain_proj, coeff)
                        term = (rc * rc).sum()
                        trust_term = term if trust_term is None else (trust_term + term)
                        trust_n = int(n_retain) if trust_n is None else min(trust_n, int(n_retain))

                    projected_deltas.append((grad, old, proj_vec))

                scale, term_val = self._compute_trust_scale(trust_term, trust_n, lr)
                self._maybe_log_trust_stats(scale, term_val, trust_n)
                for grad, old, proj_vec in projected_deltas:
                    if scale < 1.0:
                        proj_vec = proj_vec * scale
                    if old is None:
                        grad.copy_(proj_vec)
                    else:
                        grad.copy_(old + proj_vec)

            # 2) Backward retain loss (no projection)
            if do_retain:
                retain_obj.backward()
            if not do_forget and not do_retain and bool(getattr(loss, "requires_grad", False)):
                loss.backward()
        else:
            # Fallback: standard backward then project all LoRA-B gradients
            loss.backward()
            self._apply_csm_projection()

        return loss.detach()
