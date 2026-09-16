"""
GPM (Gradient Projection Memory) Trainer
实现梯度正交投影以防止 forget 训练时对 retain 性能的干扰。
"""

import os
import torch
import pickle
from typing import Callable, Dict, Optional
from .hf_trainers import ForgetTrainer

# 检查apex是否可用
try:
    from apex import amp
    _apex_available = True
except ImportError:
    _apex_available = False


class GPMForgetTrainer(ForgetTrainer):
    """
    基于GPM方法的遗忘训练器
    在梯度更新时减去与retain基底的投影，防止对retain性能的干扰
    """
    
    def __init__(self,
                 model,
                 train_loss_function: Callable,
                 gmp_basis_path: str = "./gmp_basis/retain99_pca_basis.pkl",
                 enable_gmp: bool = True,
                 project_forget_only: bool = False,
                 **kwargs):
        """
        初始化GPM训练器

        Args:
            model: 要训练的模型
            train_loss_function: 训练损失函数
            gmp_basis_path: retain基底文件路径
            enable_gmp: 是否启用GPM投影
            project_forget_only: 是否仅投影 forget 侧梯度
            **kwargs: 其他参数传递给父类
        """
        super().__init__(model=model, train_loss_function=train_loss_function, **kwargs)

        self.enable_gmp = enable_gmp
        self.project_forget_only = bool(project_forget_only)
        self._project_param_bindings = None
        
        if self.enable_gmp:
            print(f"🔧 启用GPM梯度正交投影（按照论文标准实现）")
            print(f"📁 加载retain基底: {gmp_basis_path}")
            self.retain_basis = self._load_retain_basis(gmp_basis_path)
            print(f"✅ 成功加载 {len(self.retain_basis)} 层的retain基底")
        else:
            print("❌ GPM投影已禁用")
            self.retain_basis = None
    
    def _load_retain_basis(self, basis_path: str) -> Dict:
        """加载retain基底"""
        try:
            if not os.path.isabs(basis_path):
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                # 如果是相对路径，尝试多个可能的位置
                possible_paths = [
                    basis_path,  # 当前目录
                    os.path.join('..', basis_path),  # 上级目录
                    os.path.join(repo_root, basis_path),  # 项目根目录
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        basis_path = path
                        break
                else:
                    raise FileNotFoundError(f"无法在以下路径找到基底文件: {possible_paths}")

            print(f"🔍 从路径加载基底: {os.path.abspath(basis_path)}")
            with open(basis_path, 'rb') as f:
                basis_dict = pickle.load(f)
            
            # 转换为torch tensor并移动到正确设备
            processed_basis = {}
            for layer_name, basis_info in basis_dict.items():
                raw = basis_info['components']
                if torch.is_tensor(raw):
                    components = raw.detach().to(dtype=torch.float32, device='cpu').contiguous()
                else:
                    components = torch.as_tensor(raw, dtype=torch.float32).contiguous()
                processed_basis[layer_name] = {
                    'components': components,  # [n_components, n_features]
                    'n_components': basis_info['n_components'],
                    'feature_dim': components.shape[1]  # 从components形状推断feature_dim
                }
            
            return processed_basis
        except Exception as e:
            print(f"❌ 加载retain基底失败: {e}")
            raise e
    
    def _get_parameter_layer_mapping(self) -> Dict[str, str]:
        """
        建立模型参数名到层名的映射
        用于将梯度映射到对应的retain基底
        """
        param_to_layer = {}
        
        for name, param in self.model.named_parameters():
            if 'lora_B' in name and 'up_proj' in name and 'default' in name:
                # 提取层名，例如从 'base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight'
                # 提取到 'base_model.model.model.layers.0.mlp.up_proj.lora_B.default'
                layer_name = '.'.join(name.split('.')[:-1])  # 移除最后的 '.weight'
                param_to_layer[name] = layer_name
        
        return param_to_layer

    def _resolve_basis_key(self, layer_name: str) -> Optional[str]:
        """
        兼容LoRA参数名与基底名的对齐问题，优先匹配完整名，再回退到up_proj基名
        """
        candidates = []

        def _add(candidate: Optional[str]):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        _add(layer_name)
        if layer_name.startswith("base_model.model."):
            _add(layer_name.replace("base_model.model.", "", 1))
        if layer_name.startswith("base_model.model.model."):
            _add(layer_name.replace("base_model.model.model.", "model.", 1))

        expanded = list(candidates)
        for candidate in expanded:
            if ".lora_B.default" in candidate:
                _add(candidate.replace(".lora_B.default", ""))
            if ".lora_B" in candidate:
                _add(candidate.replace(".lora_B", ""))

        for key in candidates:
            if key in self.retain_basis:
                return key
        return None

    def _iter_projectable_params(self):
        if self._project_param_bindings is None:
            bindings = []
            for name, param in self.model.named_parameters():
                if 'lora_B' not in name or 'up_proj' not in name or 'default' not in name:
                    continue
                if param.dim() != 2:
                    continue
                basis_key = self._resolve_basis_key(name.rsplit(".weight", 1)[0])
                if basis_key is None:
                    continue
                bindings.append((name, param, basis_key))
            self._project_param_bindings = tuple(bindings)
        return self._project_param_bindings

    def _project_grad_tensor(self, grad: torch.Tensor, basis_key: str) -> torch.Tensor:
        if grad.dim() != 2:
            return grad
        basis_info = self.retain_basis[basis_key]
        compute_dtype = torch.float32
        M_T = basis_info['components'].to(device=grad.device, dtype=compute_dtype)
        M = M_T.T

        projected_grad = torch.zeros_like(grad)
        output_dim, rank = grad.shape
        if M.shape[0] != output_dim:
            return grad

        for r in range(rank):
            grad_vector = grad[:, r].to(dtype=compute_dtype)
            original_norm = torch.norm(grad_vector)
            if original_norm <= 1e-8:
                projected_grad[:, r] = grad[:, r]
                continue
            grad_M = torch.matmul(grad_vector.unsqueeze(0), M)
            projection = torch.matmul(grad_M, M_T).squeeze(0)
            projected_grad[:, r] = (grad_vector - projection).to(dtype=grad.dtype)
        return projected_grad
    
    def _project_gradient_orthogonal(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        按照GPM论文公式进行梯度正交投影

        论文公式6 (FC层): ∇W_l L_2 = ∇W_l L_2 - (∇W_l L_2)M^l(M^l)^T
        论文公式7 (Conv层): ∇W_l L_2 = ∇W_l L_2 - M^l(M^l)^T(∇W_l L_2)

        对于LoRA B矩阵，我们使用FC层的公式6

        Args:
            gradients: 参数名到梯度的映射

        Returns:
            投影后的梯度字典
        """
        if not self.enable_gmp or self.retain_basis is None:
            return gradients

        projected_gradients = {}

        # 诊断统计
        total_similarity = 0.0
        total_projection_ratio = 0.0
        projection_count = 0

        for param_name, grad in gradients.items():
            if 'lora_B' in param_name and 'up_proj' in param_name and 'default' in param_name:
                layer_name = param_name.rsplit(".weight", 1)[0]
                basis_key = self._resolve_basis_key(layer_name)
                if basis_key is not None:
                    projected_grad = self._project_grad_tensor(grad, basis_key)
                    if grad.dim() == 2:
                        for r in range(grad.shape[1]):
                            grad_vector = grad[:, r]
                            projected_vector = projected_grad[:, r]
                            original_norm = torch.norm(grad_vector)
                            if original_norm > 1e-8:
                                projection = grad_vector - projected_vector
                                projection_norm = torch.norm(projection)
                                similarity = projection_norm / original_norm
                                total_similarity += similarity.item()
                                projected_norm = torch.norm(projected_vector)
                                projection_ratio = projected_norm / original_norm
                                total_projection_ratio += projection_ratio.item()
                                projection_count += 1
                    projected_gradients[param_name] = projected_grad
                else:
                    projected_gradients[param_name] = grad
            else:
                projected_gradients[param_name] = grad

        # 🔍 记录诊断信息
        if projection_count > 0:
            avg_similarity = total_similarity / projection_count
            avg_projection_ratio = total_projection_ratio / projection_count
            if hasattr(self, '_gmp_step_count'):
                self._gmp_step_count += 1
            else:
                self._gmp_step_count = 1

            # 每10步记录一次
            if self._gmp_step_count % 10 == 0:
                print(f"GPM诊断 [Step {self._gmp_step_count}]: "
                      f"梯度-基底相似度={avg_similarity:.4f}, "
                      f"投影后梯度比例={avg_projection_ratio:.4f}")

        return projected_gradients
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法，在其中加入GPM梯度投影
        """
        # 调用父类的compute_loss方法
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # 在损失计算后，梯度更新前应用GPM投影
        # 注意：实际的梯度投影会在optimizer.step()之前的backward()之后进行

        return loss

    def _apply_gmp_projection(self):
        """
        应用GPM梯度投影到当前模型的梯度
        """
        if not self.enable_gmp or self.retain_basis is None:
            return

        # 收集当前梯度
        current_gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                current_gradients[name] = param.grad.clone()

        # 应用正交投影
        projected_gradients = self._project_gradient_orthogonal(current_gradients)

        # 更新模型梯度
        for name, param in self.model.named_parameters():
            if name in projected_gradients:
                param.grad = projected_gradients[name]
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写训练步骤，在梯度计算后应用GPM投影
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 检查是否使用apex
        use_apex = hasattr(self, 'use_apex') and self.use_apex
        use_cuda_amp = hasattr(self, 'use_cuda_amp') and self.use_cuda_amp
        deepspeed = getattr(self, 'deepspeed', None)

        if self.project_forget_only and hasattr(self.train_loss_function, "__call__"):
            losses = self.train_loss_function(model, inputs, self.oracle_model)
            loss = losses["loss"]
            forget_loss = losses.get("forget_loss")
            retain_loss = losses.get("retain_loss")

            forget_weight = float(getattr(self.train_loss_function, "forget_weight", 1.0) or 1.0)
            retain_weight = float(getattr(self.train_loss_function, "retain_weight", 1.0) or 1.0)

            forget_obj = (forget_loss * forget_weight) if forget_loss is not None else None
            retain_obj = (retain_loss * retain_weight) if retain_loss is not None else None

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
            if use_apex and _apex_available:
                with amp.autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
            forget_obj = None
            retain_obj = None

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            if forget_obj is not None:
                forget_obj = forget_obj / self.args.gradient_accumulation_steps
            if retain_obj is not None:
                retain_obj = retain_obj / self.args.gradient_accumulation_steps

        if self.project_forget_only and forget_obj is not None and retain_obj is not None and not deepspeed:
            do_forget = bool(getattr(forget_obj, "requires_grad", False))
            do_retain = bool(getattr(retain_obj, "requires_grad", False))

            if do_forget:
                saved = {}
                for name, param, _ in self._iter_projectable_params():
                    if param.grad is None:
                        continue
                    saved[name] = param.grad.detach().clone()

                forget_obj.backward()

                for name, param, basis_key in self._iter_projectable_params():
                    grad = param.grad
                    if grad is None or grad.dim() != 2:
                        continue
                    old = saved.get(name)
                    delta = grad if old is None else (grad - old)
                    projected_delta = self._project_grad_tensor(delta, basis_key)
                    if old is None:
                        grad.copy_(projected_delta)
                    else:
                        grad.copy_(old + projected_delta)

            if do_retain:
                retain_obj.backward()
        else:
            if use_apex and _apex_available:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif use_cuda_amp and hasattr(self, 'scaler'):
                self.scaler.scale(loss).backward()
            elif deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = deepspeed.backward(loss)
            else:
                loss.backward()

            # 在梯度计算完成后应用GPM投影
            self._apply_gmp_projection()

        return loss.detach()

    def log_gmp_info(self, step: int):
        """记录GPM相关信息"""
        if self.enable_gmp and step % 10 == 0:  # 每10步记录一次
            log_items = {
                'gmp/enabled': 1.0,
                'gmp/num_basis_layers': len(self.retain_basis) if self.retain_basis else 0
            }
            self.log(log_items)
