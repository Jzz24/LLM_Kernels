import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple

from group_gemm_kernel import (
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)

from utils import ResultCollector
from all2all_moe import MoEGate, Config


class GroupedGemmRunner(torch.nn.Module):
    """简化的GroupedGemmRunner"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.use_flashinfer = False

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
    ):
        c = grouped_gemm_triton(
            a, b, c, batch_size, weight_column_major, 
            seg_indptr, weight_indices, use_fp8_w8a8,
            scale_a, scale_b, block_shape=block_shape
        )
        return c


class EPMoE(torch.nn.Module):
    """简化版EPMoE，使用all_gather通信模式"""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = True,
        num_expert_group: Optional[int] = 8,
        topk_group: Optional[int] = 4,
        tp_size: Optional[int] = None,
        activation: str = "silu",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = tp_size or 1
        if self.tp_size > 1:
            # 初始化分布式环境
            if not dist.is_initialized():
                raise RuntimeError("分布式环境未初始化")
            self.tp_rank = dist.get_rank()
        else:
            self.tp_rank = 0

        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0, "专家数量必须被GPU数量整除"
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.activation = activation

        # 简化的量化设置
        self.use_fp8_w8a8 = False
        self.use_block_quant = False
        self.block_shape = None

        # 初始化权重
        # w13_weight: 融合的gate和up投影
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        
        # w2_weight: down投影
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        
        # 初始化缩放因子
        ones_tensor = torch.ones(self.num_experts_per_partition, dtype=torch.float32)
        self.w13_input_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)
        self.w2_input_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)
        self.w13_weight_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)
        self.w2_weight_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)
        
        # 为方便对比，使用MoEGate
        # self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.config = Config()
        self.gate = MoEGate(self.config)
        
        # 初始化GroupedGemmRunner
        self.grouped_gemm_runner = None

    def forward(self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None):
        ResultCollector.save("hidden_states", hidden_states, prefix="epmoe", rank=self.tp_rank)
        # 初始化GroupedGemmRunner(如果需要)
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(hidden_states.device)
            
        # 获取batch_size和seq_len
        batch_size, seq_len, _ = hidden_states.shape
        # 外围测试框架已经保证每个rank，可以看见所有tokens,
        # 否则在此处需要进行hidden_states的all_gather
        
        # 计算路由概率和topk专家, 使用deepseekv3 hf版本的实现
        topk_ids, topk_weights = self.gate(hidden_states)

        ResultCollector.save("topk_weights", topk_weights, prefix="epmoe", rank=self.tp_rank)
        ResultCollector.save("topk_ids", topk_ids.float(), prefix="epmoe", rank=self.tp_rank)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        topk_ids = topk_ids.reshape(-1, self.top_k)
        topk_weights = topk_weights.reshape(-1, self.top_k)


        # 运行MoE的预处理步骤
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )
        
        # 准备gateup_input
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # TODO: dynamic quantize hidden_states, to init self.w13_input_scale
        
        # 按照expert顺序重排每个rank的tokens
        # 申请[num_tokens * topk, hidden_size]形状的gateup_input
        # 在当前非all_to_all的通信模式下
        # 重排后的gateup_input,每个rank只处理 约1/num_rank 的专家
        # 因此大约有(num_rank-1)/num_rank的空间被浪费
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        
        # 准备当前rank的seg_indptr和weight_indices
        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        
        # 第一次分组矩阵乘法
        gateup_output = torch.empty(
            gateup_input.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        gateup_output = self.grouped_gemm_runner(
            a=gateup_input, # [num_tokens * topk, hidden_size]
            b=self.w13_weight, # [num_experts_per_partition, 2 * intermediate_size, hidden_size]
            c=gateup_output, # [num_tokens * topk, 2 * intermediate_size]
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=self.w13_weight_scale,
            block_shape=self.block_shape,
        )
        
        # 准备down_input
        down_input = torch.empty(
            gateup_output.shape[0], # num_tokens * topk
            gateup_output.shape[1] // 2, # intermediate_size
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        
        # 应用激活函数
        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        # 第二次分组矩阵乘法
        # w2 shape: [num_experts_per_partition, hidden_size, intermediate_size]
        down_output = torch.empty(
            down_input.shape[0], # num_tokens * topk
            self.w2_weight.shape[1], # hidden_size
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=self.w2_weight_scale,
            block_shape=self.block_shape,
        )
        
        # 后重排序输出
        output = torch.empty(
            (batch_size, seq_len, self.hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        post_reorder_triton_kernel[(batch_size * seq_len,)](
            down_output,
            output.reshape(-1, self.hidden_size),
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            self.hidden_size,
            BLOCK_SIZE=512,
        )
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output