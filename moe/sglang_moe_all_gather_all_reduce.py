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
from all2all_moe import MoEGate
from config import Config

PREFIX = "epmoe"


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
        config: Config,
    ):
        super().__init__()

        self.config = config
        # if params_dtype is None:
        #     params_dtype = torch.get_default_dtype()

        self.tp_size = self.config.ep_size or 1
        if self.tp_size > 1:
            # 初始化分布式环境
            if not dist.is_initialized():
                raise RuntimeError("分布式环境未初始化")
            self.tp_rank = dist.get_rank()
        else:
            self.tp_rank = 0

        self.num_experts = self.config.n_routed_experts
        assert self.num_experts % self.tp_size == 0, "专家数量必须被GPU数量整除"
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = self.config.num_experts_per_tok
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.moe_intermediate_size
        self.renormalize = True
        self.use_grouped_topk = True
        self.num_expert_group = self.config.n_group
        self.topk_group = self.config.topk_group
        self.activation = self.config.hidden_act

        # 为方便对比，使用MoEGate
        # self.router = nn.Linear(hidden_size, num_experts, bias=False)
        # self.config = Config()
        self.gate = MoEGate(self.config)
        # 初始化GroupedGemmRunner
        self.grouped_gemm_runner = None
        self._init_weight_and_scale()

    def _init_weight_and_scale(self,):
        params_dtype = self.config.w_dtype if self.config.quantize_method else torch.get_default_dtype()

        self.w13_weight = nn.Parameter(
            torch.ones(
                self.num_experts_per_partition,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        self.w2_weight = nn.Parameter(
            torch.ones(
                self.num_experts_per_partition,
                self.hidden_size,
                self.intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        ones_tensor = torch.ones(self.num_experts_per_partition, dtype=torch.float16)
        self.w13_input_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)
        self.w2_input_scale = nn.Parameter(ones_tensor.clone(), requires_grad=False)

        # 在group_gemm内部初始化
        self.w13_input_scale = None
        self.w2_input_scale = None

        if self.config.use_block_quant and self.config.quantize_method is not None:
            block_n, block_k = self.config.w_quant_block_size
            self.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    self.num_experts_per_partition,
                    2 * ((self.intermediate_size + block_n - 1) // block_n),
                    (self.hidden_size + block_k - 1) // block_k,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
            self.w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    self.num_experts_per_partition,
                    (self.hidden_size + block_n - 1) // block_n,
                    (self.intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
        else:
            self.w13_weight_scale = None
            self.w2_weight_scale = None

        #######
        # 简化的量化设置
        self.use_fp8_w8a8 = False
        self.use_block_quant = self.config.use_block_quant
        self.block_shape = self.config.w_quant_block_size if self.config.quantize_method \
                                                        and self.use_block_quant else None


    def forward(self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None):
        ResultCollector.save("hidden_states", hidden_states, prefix=PREFIX, rank=self.tp_rank)
        # 初始化GroupedGemmRunner(如果需要)
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(hidden_states.device)
            
        # 获取batch_size和seq_len
        batch_size, seq_len, _ = hidden_states.shape
        # 外围测试框架已经保证每个rank，可以看见所有tokens,
        # 否则在此处需要进行hidden_states的all_gather
        
        # 计算路由概率和topk专家, 使用deepseekv3 hf版本的实现
        topk_ids, topk_weights = self.gate(hidden_states)

        ResultCollector.save("topk_weights", topk_weights, prefix=PREFIX, rank=self.tp_rank)
        ResultCollector.save("topk_ids", topk_ids.float(), prefix=PREFIX, rank=self.tp_rank)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        topk_ids = topk_ids.reshape(-1, self.top_k)
        topk_weights = topk_weights.reshape(-1, self.top_k)


        # 运行MoE的预处理步骤
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )
        
        # 准备gateup_input
        # TODO: 将hidden_states量化和reorder组合在一个kernel里面，
        # sglang默认采用的是expert的量化，不符合我们的需求，但为保持接口一致，暂时不修改
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # 按照expert顺序重排每个rank的tokens
        # 输入fp16 hidden_states, 输出量化的gateup_input
        # 申请[num_tokens * topk, hidden_size]形状的gateup_input
        # 在当前非all_to_all的通信模式下
        # 重排后的gateup_input,每个rank只处理 约1/num_rank 的tokens
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

        # TODO: 接收已经量化的gateup_in
        # 当前gateup_in的量化在group_gemm_kernel中完成,待优化
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

        assert gateup_output.isnan().sum() == 0, "gateup_output has NaN"
        assert gateup_output.isinf().sum() == 0, "gateup_output has Inf"
        
        # 准备down_input
        # TODO:将down_input的量化和激活函数组合在一个kernel里面
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
        # TODO：接收已经量化的down_input
        # 当前down_input的量化在group_gemm_kernel中完成,待优化
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

        ResultCollector.save("output", output, prefix=PREFIX, rank=self.tp_rank)
        return output