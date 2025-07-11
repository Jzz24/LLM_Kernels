import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import numpy as np

from transformers.activations import ACT2FN

from utils import ResultCollector
from config import Config

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
        # ResultCollector.save("hidden_states", hidden_states, prefix="all2all")
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)


        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training and not self.config.use_triton_group_gemm:
            y = self.moe_infer_group_gemm(hidden_states, topk_idx, topk_weight).view(*orig_shape)


        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    
    # @torch.no_grad()
    # def moe_infer_group_gemm(self, x, topk_ids, topk_weight):
    #         # x: [num_tokens, hidden_size]
    #         # topk_ids: [num_tokens, top_k]
    #         # topk_weight: [num_tokens, top_k]

    #         num_tokens, hidden_size = x.shape
    #         output = torch.zeros_like(x)
            
    #         start_expert_id = self.ep_rank * self.experts_per_rank
    #         end_expert_id = start_expert_id + self.experts_per_rank

    #         # 1. 计算每个专家的token数并重排token
    #         cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
    #         cnts.scatter_(1, topk_ids, 1)
    #         tokens_per_expert = cnts.sum(dim=0)

    #         expert_idxs = topk_ids.view(-1).argsort()
    #         resorted_token_idxs = expert_idxs // topk_ids.shape[1]
    #         resorted_token_weights = topk_weight.view(-1)[expert_idxs]
    #         tokens_per_expert_offsets = torch.cumsum(tokens_per_expert, dim=0)
            
    #         # 2. Shuffle: 根据重排索引收集tokens
    #         shuffled_tokens = x[resorted_token_idxs]

    #         # 3. Group GEMM: 对每个专家进行分段批量计算
    #         # expert_outputs = torch.empty_like(shuffled_tokens)
    #         expert_outputs = torch.zeros_like(shuffled_tokens)

    #         for i in range(start_expert_id, end_expert_id):
                
    #             start = tokens_per_expert_offsets[i-1] if i > 0 else 0
    #             end = tokens_per_expert_offsets[i]
    #             if end - start == 0: continue
                
    #             expert = self.experts[i]
    #             if expert is None: continue # EP模式下，其他rank的专家为None
                
    #             expert_outputs[start:end] = expert(shuffled_tokens[start:end])

    #         # 4. Unshuffle: 将加权后的结果加回到原始token位置
    #         weighted_outputs = (expert_outputs.type(topk_weight.dtype) * \
    #             resorted_token_weights.unsqueeze(-1)).type(x.dtype)
    #         output.scatter_add_(0, resorted_token_idxs.unsqueeze(-1).expand(-1, hidden_size), weighted_outputs)

    #         # 5. All-Reduce: 在EP模式下聚合所有rank的结果
    #         if self.ep_size > 1:
    #             dist.all_reduce(output, op=dist.ReduceOp.SUM)

    #         return output

    @torch.no_grad()
    def moe_infer_group_gemm(self, x, topk_ids, topk_weight):
            # x: [num_tokens, hidden_size]
            # topk_ids: [num_tokens, top_k]
            # topk_weight: [num_tokens, top_k]

            num_tokens, hidden_size = x.shape
            output = torch.zeros_like(x)
            
            start_expert_id = self.ep_rank * self.experts_per_rank
            end_expert_id = start_expert_id + self.experts_per_rank

            # 1. 计算全局的token-expert映射关系 (这部分逻辑不变)
            cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
            cnts.scatter_(1, topk_ids, 1)
            tokens_per_expert = cnts.sum(dim=0)
            tokens_per_expert_offsets = torch.cumsum(tokens_per_expert, dim=0)

            expert_idxs = topk_ids.view(-1).argsort()
            resorted_token_idxs = expert_idxs // topk_ids.shape[1]
            resorted_token_weights = topk_weight.view(-1)[expert_idxs]
            
            # 2. 【优化】不再创建shuffled_tokens和expert_outputs中间变量
            #    在循环内按需读取输入，并直接将结果累加到output中

            for i in range(start_expert_id, end_expert_id):
                start = tokens_per_expert_offsets[i-1] if i > 0 else 0
                end = tokens_per_expert_offsets[i]
                if end - start == 0: continue
                
                expert = self.experts[i]
                if expert is None: continue
                
                # 提取当前专家对应的索引和权重
                current_indices = resorted_token_idxs[start:end]
                current_weights = resorted_token_weights[start:end]
                
                # 按需从x中获取输入
                expert_input = x[current_indices]
                
                # 计算专家输出
                expert_output = expert(expert_input)
                
                # 加权
                weighted_output = (expert_output.type(topk_weight.dtype) * \
                    current_weights.unsqueeze(-1)).type(x.dtype)
                
                # 直接将结果加回到output中，避免创建大型中间张量
                output.scatter_add_(0, current_indices.unsqueeze(-1).expand(-1, hidden_size), weighted_output)

            # 3. All-Reduce: 在EP模式下聚合所有rank的结果
            if self.ep_size > 1:
                dist.all_reduce(output, op=dist.ReduceOp.SUM)

            return output


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