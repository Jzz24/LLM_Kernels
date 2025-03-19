import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import numpy as np

from transformers.activations import ACT2FN

from utils import ResultCollector


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape #[1, 4, 7168]
        ### compute gating score
        hidden_states = hidden_states.view(-1, h) #[bsz*seq_len, h] [1*4, 7168]
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            # [n, n_group] -> [1*4, 8, 32]experts分8组,
            # 每组32个experts,选择每组中的top2 expertsd的分数进行加和，作为该组的group_score
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )
            # [n, top_k_group] -> [1*4, 4],选择top4的group
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            
            #将group_mask中被选中的top4 group的位置置为1
            group_mask = torch.zeros_like(group_scores)  # [n, n_group] [4, 8]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]

            # group_mask从[4, 8]转换为[1*4, 8, 32]，再到[1*4, 256]，被选中的group中，所有32个experts的位置置为1
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]，其中4个group中的所有 32个experts的位置置为1,即4*32=128个experts的位置置为1

            # 将最初的scores_for_choice[1*4, 256]中，被选中的128个experts的位置的分数保留，其余的位置的分数置为0
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            # 至此，通过group_scores的筛选，得到了top4的group，并保留其中的experts的分数

            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            ) # [1*4, 8]，所有被选中的group中，挑选出top8的experts, 返回索引
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor
        return topk_idx, topk_weight

class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV3MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV3MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        ResultCollector.save("hidden_states", hidden_states, prefix="all2all")
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)

        ResultCollector.save("topk_weights", topk_weight, prefix="all2all")
        ResultCollector.save("topk_ids", topk_idx.float(), prefix="all2all")

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training and not self.config.use_triton_group_gemm:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        elif not self.training and self.config.use_triton_group_gemm:
            y = self.triton_group_gemm_moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts))) # [bs*seq_len, e]
        cnts.scatter_(1, topk_ids, 1) # [bs*seq_len, e]被选中的experts的位置置为1

        # [e], 每个expert被选中的次数,即分配的tokens数
        tokens_per_expert = cnts.sum(dim=0)

        # topk_ids表示被选中的experts的索引，
        # 然后对experts的索引进行排序（因为后续按照experts的顺序遍历），返回排序后的索引
        idxs = topk_ids.view(-1).argsort()

        # [bs*seq_len*topk_experts, h] [1*4*8, 7168], 按照experts的顺序，重组tokens
        # 在分布式场景通信前，sorted_tokens包含了许多非当前rank负责的专家的tokens,需要进行all_to_all通信
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            # 在分布式数据并行中，由于每个rank的输入tokens不同，每个rank的tokens_per_expert也不同

            # 1.分享每个rank的专家要处理的token数量
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()

            # 2.根据这些信息交换不同rank的token数据
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )

            # 3.重新排序，确保每个专家的token连续存放
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            # 通信后，sorted_tokens只包含当前rank负责的专家需要处理的tokens
            sorted_tokens = gathered_tokens[gatherd_idxs]

            # 4.更新每个进程上专家的负载情况
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs #将按照expert的顺序计算的结果，重新按照tokens的顺序排列
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    @torch.no_grad()
    def triton_group_gemm_moe_infer(self, x, topk_ids, topk_weight):
        from group_gemm_kernel import run_moe_ep_preproess
        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.config.n_routed_experts)
        


class Config:
    def __init__(self):
        self.hidden_size = 7168
        self.intermediate_size = 18432
        self.num_experts_per_tok = 8
        self.n_routed_experts = 256
        self.routed_scaling_factor = 2.5
        self.scoring_func = "sigmoid"
        self.seq_aux = True
        self.topk_method = "noaux_tc"
        self.n_group = 8
        self.topk_group = 4
        self.norm_topk_prob = True
        self.moe_intermediate_size = 2048
        self.n_shared_experts = 0 #1
        self.ep_size = 1
        self.hidden_act = "silu"
        self.use_triton_group_gemm = False

def test_deepseekv3_moe():
    # Set default tensor type to float16 and device to cuda
    torch.set_default_dtype(torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config()
    model = DeepseekV3MoE(config).eval().to(device)
    
    # Create some random input data
    batch_size = 1
    seq_len = 4
    hidden_size = config.hidden_size
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Run the forward method
    output = model(input_data)
    
    # Print the output
    print("Output shape:", output.shape)
    print("Output:", output)

if __name__ == "__main__":
    test_deepseekv3_moe()

    """
    实现MoE (Mixture of Experts)的推理逻辑，支持专家并行(Expert Parallelism)场景
    
    在多GPU环境下的数据流动图解:
    ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
    │              GPU 0 (Rank 0)             │  │              GPU 1 (Rank 1)             │
    │                                         │  │                                         │
    │ 输入tokens: 一批序列                     │  │ 输入tokens: 另一批序列                  │
    │                                         │  │                                         │
    │ ┌─────────────────────────────────────┐ │  │ ┌─────────────────────────────────────┐ │
    │ │            Gate网络计算             │ │  │ │            Gate网络计算             │ │
    │ └─────────────────────────────────────┘ │  │ └─────────────────────────────────────┘ │
    │                   ↓                     │  │                   ↓                     │
    │ tokens_per_expert = [2, 3, 1, 2]        │  │ tokens_per_expert = [4, 1, 2, 3]        │
    └───────────────────┬─────────────────────┘  └───────────────────┬─────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                               all_to_all_single 通信                                │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
    │ tokens_per_expert_group = [2,3,4,1]     │  │ tokens_per_expert_group = [1,2,2,3]     │
    │                                         │  │                                         │
    │ output_splits = [5, 5]                  │  │ output_splits = [3, 5]                  │
    │ input_split_sizes = [5, 3]              │  │ input_split_sizes = [5, 5]              │
    └───────────────────┬─────────────────────┘  └───────────────────┬─────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                  all_to_all 通信                                    │
    │                             (交换tokens到对应专家)                                  │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
    │ • 收到需要由expert0和expert1处理的tokens  │  │ • 收到需要由expert2和expert3处理的tokens  │
    │ • 按专家重新排序tokens                   │  │ • 按专家重新排序tokens                   │
    │ tokens_per_expert = [6, 4]              │  │ tokens_per_expert = [3, 5]              │
    │                                         │  │                                         │
    │ ┌─────────────────────────────────────┐ │  │ ┌─────────────────────────────────────┐ │
    │ │  执行专家计算                       │ │  │ │  执行专家计算                       │ │
    │ │  • expert0处理6个tokens             │ │  │ │  • expert2处理3个tokens             │ │
    │ │  • expert1处理4个tokens             │ │  │ │  • expert3处理5个tokens             │ │
    │ └─────────────────────────────────────┘ │  │ └─────────────────────────────────────┘ │
    └───────────────────┬─────────────────────┘  └───────────────────┬─────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────────────────────────────────────────────────┐
    │                                  all_to_all 通信                                    │
    │                             (返回结果到原始rank)                                    │
    └─────────────────────────────────────────────────────────────────────────────────────┘
                        │                                            │
                        ▼                                            ▼
    ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐
    │ • 接收来自所有专家的计算结果             │  │ • 接收来自所有专家的计算结果             │
    │ • 按原始token顺序重新排序结果           │  │ • 按原始token顺序重新排序结果           │
    │ • 应用专家权重                          │  │ • 应用专家权重                          │
    │                                         │  │                                         │
    │ ┌─────────────────────────────────────┐ │  │ ┌─────────────────────────────────────┐ │
    │ │     最终输出整合结果                 │ │  │ │     最终输出整合结果                 │ │
    │ └─────────────────────────────────────┘ │  │ └─────────────────────────────────────┘ │
    └─────────────────────────────────────────┘  └─────────────────────────────────────────┘
    

    ## 详细说明

    ### 第一阶段：Token分配与专家数量统计

    1. **初始分配**
    - 每个rank有自己的输入token序列
    - Gate网络为每个token选择top-k个专家
    - 计算`tokens_per_expert`：统计每个专家被分配的token数量

    2. **通信1: all_to_all_single**
    - 交换`tokens_per_expert`信息，让每个rank了解每个专家的全局工作负载
    - rank0获取`[2,3,4,1]`（expert0,1的全局信息）
    - rank1获取`[1,2,2,3]`（expert2,3的全局信息）

    ### 第二阶段：Token重组与专家计算

    3. **通信2: 第一次all_to_all**
    - 将需要不同专家处理的token发送到对应的负责rank
    - rank0收到了需要expert0,1处理的所有tokens
    - rank1收到了需要expert2,3处理的所有tokens

    4. **专家计算**
    - 每个rank使用本地专家处理收到的tokens
    - 专家计算是并行的，不同rank上的专家同时工作

    ### 第三阶段：结果收集与整合

    5. **通信3: 第二次all_to_all**
    - 将专家处理结果发回原始rank
    - 确保每个token的计算结果回到它原始所在的rank

    6. **最终整合**
    - 对于每个token，结合它的top-k个专家的结果
    - 根据专家权重进行加权平均
    - 得到最终输出

    这种专家并行架构的优势在于它可以支持大量专家，同时保持计算和通信的高效性，是大规模MoE模型性能提升的关键技术。
    """