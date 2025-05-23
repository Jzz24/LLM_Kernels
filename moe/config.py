import torch

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

        # quantization
        # self.quantize_method = None
        self.quantize_method = "blockwise_w8a8_int8_gemm" # None for no quantization
        self.use_block_quant = True
        self.w_quant_block_size = (128, 128)
        self.a_quant_block_size = (1, 128)
        self.w_bit = 8
        self.a_bit = 8
        self.w_dtype = torch.int8
        self.a_dtype = torch.int8