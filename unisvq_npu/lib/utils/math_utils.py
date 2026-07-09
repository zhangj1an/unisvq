import math
import glog
import torch


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def block_LDL(H, b, check_nan=True, sigma_reg=0.01):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    retry = 0
    while retry <= 3:
        try:
            if retry == 0:
                L = torch.linalg.cholesky(H)
            else:
                glog.warning(f"Hr is not positive-definite; scaling sigma reg: {sigma_reg} -> {(10**(retry+1))*sigma_reg}")
                L = torch.linalg.cholesky(H + (10**retry)*sigma_reg*torch.eye(H.size(0), dtype=H.dtype, device=H.device))
            break
        except:
            retry += 1
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))


def approx_int_sqrt(n):
    p = int(math.floor(math.sqrt(n)))
    while (n % p != 0):
        p -= 1
    return (p, n // p)


def regularize_H(H, n, sigma_reg):
    H.div_(torch.diag(H).mean())
    idx = torch.arange(n)
    H[idx, idx] += sigma_reg
    return H
    # return H / torch.diag(H).mean() + sigma_reg * torch.eye(n, device=H.device)
