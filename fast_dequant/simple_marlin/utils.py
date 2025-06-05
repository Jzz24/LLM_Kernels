# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()


class QuantUtils():

    @staticmethod
    def pack(weight: torch.Tensor, scale: torch.Tensor, groupsize: int = -1) -> tuple:
        """
        marlin offline int4 pack function
        @weight: weight tensor of shape (in_features, out_features), (K, N), different from torch.linear.weight layout
        @scale: scale tensor of shape (in_features // groupsize, groups), (K // groupsize, N)
        @groupsize: quantization groupsize, along K dimension

        @return: packed weight tensor of shape (K // 16, N * 16 // 8)
        @return: scale tensor of shape (K // groupsize, N)
        """

        assert weight.dtype == torch.int, f"weight dtype {weight.dtype} != torch.int"
        assert weight.shape[0] % groupsize == 0, "Weight shape must be divisible by groupsize"

        tile = 16
        k, n = weight.shape

        scale = scale.reshape(1, -1)
        if groupsize != -1:
            # weight = weight.reshape((groupsize, -1, n))
            # weight = weight.permute(1, 0, 2)
            # weight = weight.reshape((k, n)).contiguous()
            scale = scale.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            scale = scale.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        scale = scale.reshape((-1, n)).contiguous()

        weight = weight.reshape((k // tile, tile, n // tile, tile))
        weight = weight.permute((0, 2, 1, 3))
        weight = weight.reshape((k // tile, n * tile))

        res = weight
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)

        # torch >= 2.3 supports uint32, https://github.com/pytorch/pytorch/pull/116594
        packed_weight = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            packed_weight |= res[:, i::8] << 4 * i
        packed_weight = torch.from_numpy(packed_weight.astype(np.int32)).to(weight.device)

        return packed_weight, scale
    

    @staticmethod
    def fake_quantize(weight: torch.Tensor, groupsize: int = -1) -> tuple:
        """
        marlin offline int4 sym quantization function
        @weight: weight tensor of shape (in_features, out_features), (K, N), different from torch.linear.weight layout
        @groupsize: quantization groupsize, along K dimension

        @return: dequantized fp16 weight tensor of shape (K, N) as reference
        @return: quantized weight tensor of shape (K, N), dtype int32
        @return: scale tensor of shape (K // groupsize, N)
        """

        assert weight.dtype == torch.half, f"weight dtype {weight.dtype} != torch.half"
        assert weight.shape[0] % groupsize == 0, "Weight shape must be divisible by groupsize"

        maxq = 2 ** 4 - 1
        k, n = weight.shape
        if groupsize != -1:
            weight = weight.reshape((-1, groupsize, n))
            weight = weight.permute(1, 0, 2)
            weight = weight.reshape((groupsize, -1))

        scale = torch.max(torch.abs(weight), 0, keepdim=True)[0]
        scale *= 2 / maxq
        weight = torch.round(weight / scale).int()
        weight += (maxq + 1) // 2
        weight = torch.clamp(weight, 0, maxq)
        ref = (weight - (maxq + 1) // 2).half() * scale

        if groupsize != -1:
            def reshape(w):
                w = w.reshape((groupsize, -1, n))
                w = w.permute(1, 0, 2)
                w = w.reshape((k, n)).contiguous()
                return w
            ref = reshape(ref)
            weight = reshape(weight)
        scale = scale.reshape((-1, n)).contiguous()

        return ref, weight, scale