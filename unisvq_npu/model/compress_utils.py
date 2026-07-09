import torch


def pack_2bit(tensor):
    """
    Pack fp16/int values 0,1,2,3 into uint8.
    Assumes the last dimension of the input tensor is divisible by 4.
    """
    # Convert to integer type
    tensor = tensor.to(torch.uint8)
    # Shape: [..., N] -> [..., N//4, 4]
    original_shape = tensor.shape
    flat_tensor = tensor.view(-1, 4)

    # Pack: v0 << 6 | v1 << 4 | v2 << 2 | v3
    packed = (flat_tensor[:, 0] << 6) | \
             (flat_tensor[:, 1] << 4) | \
             (flat_tensor[:, 2] << 2) | \
             (flat_tensor[:, 3])

    # Restore shape and return
    new_shape = list(original_shape[:-1]) + [original_shape[-1] // 4]
    return packed.view(new_shape)

def unpack_2bit(packed_tensor, original_shape, dtype=torch.float16):
    """
    Recover original values from uint8.
    """
    # Flatten and prepare to extract bits
    flat_packed = packed_tensor.view(-1, 1)

    v0 = (flat_packed >> 6) & 0b11
    v1 = (flat_packed >> 4) & 0b11
    v2 = (flat_packed >> 2) & 0b11
    v3 = (flat_packed)      & 0b11

    # Concatenate and restore original shape
    unpacked = torch.cat([v0, v1, v2, v3], dim=-1)
    return unpacked.view(original_shape).to(dtype)
