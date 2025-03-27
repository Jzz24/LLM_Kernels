import torch

class Int4QuantUtils():

    @staticmethod
    def pack(imatrix: torch.Tensor, storage_bits: int=32,
             q_bits: int=4, direction: str = "row"):
        """
        Packs a 4-bit integer matrix into a packed 16/32 bit integer matrix.
        Args:
            imatrix (torch.Tensor): matrix of integers
            storage_bits (int): number of bits to storage qmatrix
            q_bits (int): quantize bits
            direction (str): direction of packing, either "column" or "row"

        Returns:
            qmatrix (torch.Tensor): packed matrix of integers
        """
        shifts = torch.arange(0, storage_bits, q_bits, device=imatrix.device)

        imatrix = imatrix.to(torch.int8) & 0x0F
        pack_num = storage_bits // q_bits

        if direction == "column":
            imatrix = imatrix.view(-1, imatrix.shape[1] // pack_num, pack_num)
            qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

        elif direction == "row":
            imatrix = imatrix.view(imatrix.shape[0] // pack_num, pack_num, -1)
            qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

        qmatrix = qmatrix.to(torch.int16) if storage_bits == 16  else qmatrix.to(torch.int32)

        return qmatrix

    @staticmethod
    def unpack(qmatrix: torch.Tensor, storage_bits: int=32,
               q_bits: int=4, direction: str = "row"):
        """
        Unpacks a 16/32 bit packed integer matrix into a 4-bit integer matrix.

        Args:
            qmatrix (torch.Tensor): matrix of packed integers
            storage_bits (int): number of bits to storage qmatrix
            q_bits (int): quantize bits
            direction (str): direction of unpacking, either "column" or "row"

        Returns:
            imatrix (torch.Tensor): matrix of integers
        """
        shifts = torch.arange(0, storage_bits, q_bits, device=qmatrix.device)

        if direction == "column":
            imatrix = torch.bitwise_right_shift(
                qmatrix[:, :, None], shifts[None, None, :]
            ).view(qmatrix.shape[0], -1)

        elif direction == "row":
            imatrix = torch.bitwise_right_shift(
                qmatrix[:, None, :], shifts[None, :, None]
            ).view(-1, qmatrix.shape[-1])

        imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

        return imatrix