import torch
import numpy as np
import os
from tda_module.ph_generation import batch_job

def dgm_to_tensor(dgm, dtype=torch.float32):
    """
    Convert a persistence diagram into Tensor(N,2)

    Parameters
    ----------
    dgm :
        PershomBox persistence diagram

    Returns
    -------
    tensor : (N,2)
    """

    if dgm is None:
        return torch.zeros((0, 2), dtype=dtype)

    if len(dgm) == 0:
        return torch.zeros((0, 2), dtype=dtype)

    return torch.tensor(
        np.asarray(dgm),
        dtype=dtype
    )


def prepare_pd_batch(
        diagrams,
        device=None,
        dtype=torch.float32):
    """
    Pad a list of persistence diagrams.

    Parameters
    ----------
    diagrams :
        List[Tensor(N_i,2)]

    Returns
    -------
    batch_tensor :
        (batch_size,max_points,2)

    mask :
        (batch_size,max_points)

    max_points :
        int

    batch_size :
        int
    """

    batch_size = len(diagrams)

    if batch_size == 0:
        raise ValueError("Empty diagram list.")

    if device is None:
        device = diagrams[0].device

    max_points = max(
        max(d.size(0), 1)
        for d in diagrams
    )

    batch_tensor = torch.zeros(
        batch_size,
        max_points,
        2,
        dtype=dtype,
        device=device
    )

    mask = torch.zeros(
        batch_size,
        max_points,
        dtype=dtype,
        device=device
    )

    for i, dgm in enumerate(diagrams):

        n_points = dgm.size(0)

        if n_points > 0:

            batch_tensor[i, :n_points] = dgm

            mask[i, :n_points] = 1.0

    return (
        batch_tensor,
        mask,
        max_points,
        batch_size
    )


def dgms_to_tensor(
        all_dgms,
        key='top',
        device='cpu',
        dtype=torch.float32):
    """
    Convert nested dgms structure into tensor.

    Input structure
    ---------------
    all_dgms[b][key][v]

    Returns
    -------
    pd_tensor :
        (batch,n_var,max_points,2)

    mask :
        (batch,n_var,max_points)

    max_points :
        int
    """

    batch_size = len(all_dgms)

    if batch_size == 0:
        raise ValueError("Empty all_dgms.")

    n_var = len(all_dgms[0][key])

    diagrams = []

    for b in range(batch_size):

        for v in range(n_var):

            dgm = all_dgms[b][key][v]

            diagrams.append(
                dgm_to_tensor(
                    dgm,
                    dtype=dtype
                )
            )

    (
        batch_tensor,
        mask,
        max_points,
        _
    ) = prepare_pd_batch(
        diagrams,
        device=device,
        dtype=dtype
    )

    batch_tensor = batch_tensor.view(
        batch_size,
        n_var,
        max_points,
        2
    )

    mask = mask.view(
        batch_size,
        n_var,
        max_points
    )

    return (
        batch_tensor,
        mask,
        max_points
    )


def dgms_to_top_bottom_tensor(
        all_dgms,
        device='cpu',
        dtype=torch.float32):
    """
    Convert both top and bottom diagrams.

    Returns
    -------
    top_tensor :
        (batch,n_var,max_top_pts,2)

    top_mask :
        (batch,n_var,max_top_pts)

    bottom_tensor :
        (batch,n_var,max_bottom_pts,2)

    bottom_mask :
        (batch,n_var,max_bottom_pts)
    """

    (
        top_tensor,
        top_mask,
        top_max_points
    ) = dgms_to_tensor(
        all_dgms,
        key='top',
        device=device,
        dtype=dtype
    )

    (
        bottom_tensor,
        bottom_mask,
        bottom_max_points
    ) = dgms_to_tensor(
        all_dgms,
        key='bottom',
        device=device,
        dtype=dtype
    )

    return {
        'top_tensor': top_tensor,
        'top_mask': top_mask,
        'top_max_points': top_max_points,

        'bottom_tensor': bottom_tensor,
        'bottom_mask': bottom_mask,
        'bottom_max_points': bottom_max_points
    }


def combine_top_bottom(result_dict):
    """
    Combine top and bottom persistence diagrams.

    Parameters
    ----------
    result_dict : dict

        Output from dgms_to_top_bottom_tensor()

    Returns
    -------
    combined_tensor :
        (batch, n_var, 2, max_points, 2)

    combined_mask :
        (batch, n_var, 2, max_points)

    max_points :
        int
    """

    top_tensor = result_dict["top_tensor"]
    bottom_tensor = result_dict["bottom_tensor"]

    top_mask = result_dict["top_mask"]
    bottom_mask = result_dict["bottom_mask"]

    batch_size = top_tensor.size(0)
    n_var = top_tensor.size(1)

    top_max = top_tensor.size(2)
    bottom_max = bottom_tensor.size(2)

    max_points = max(
        top_max,
        bottom_max
    )

    device = top_tensor.device
    dtype = top_tensor.dtype

    combined_tensor = torch.zeros(
        batch_size,
        n_var,
        2,
        max_points,
        2,
        device=device,
        dtype=dtype
    )

    combined_tensor[:, :, 0, :top_max, :] = top_tensor
    combined_tensor[:, :, 1, :bottom_max, :] = bottom_tensor

    combined_mask = torch.zeros(
        batch_size,
        n_var,
        2,
        max_points,
        device=device,
        dtype=top_mask.dtype
    )

    combined_mask[:, :, 0, :top_max] = top_mask
    combined_mask[:, :, 1, :bottom_max] = bottom_mask

    return {
        "pd_tensor": combined_tensor,
        "pd_mask": combined_mask,
        "max_points": max_points
    }

def compute_and_cache_tda(
    seq_x,
    seq_len,
    index,
    cache_dir,
    patch_len=24,
    device="cpu",
    recompute=False
):
    """
    Compute and cache TDA results for a single time-series sample.

    Args:
        seq_x: (L, N) or torch.Tensor
        index: sample index (int)
        seq_len: length of lookback window of seq_x
        cache_dir: directory for saving results
        patch_len: patch size used in TDACollateFn
        device: device for TDA tensors
        recompute: force recomputation even if cache exists

    Returns:
        dict:
            {
                "pd_tensor": ...,
                "pd_mask": ...,
                "max_points": ...
            }
    """

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{seq_len}_{index}.pt")

    if os.path.exists(cache_path) and not recompute:
        return torch.load(cache_path)

    if not isinstance(seq_x, torch.Tensor):
        seq_x = torch.tensor(seq_x, dtype=torch.float32)

    # shape: (L, N) -> (1, L, N)
    batch_x = seq_x.unsqueeze(0)

    B, L, N = batch_x.shape

    if L > patch_len:
        assert L % patch_len == 0, (
            f"L={L} must be divisible by patch_len={patch_len}"
        )

        n_patches = L // patch_len

        tda_x = batch_x.reshape(B, n_patches, patch_len, N)
        tda_x = tda_x.reshape(B * n_patches, patch_len, N)
    else:
        tda_x = batch_x
        n_patches = 1

    all_dgms = batch_job(tda_x)

    pd_dict = dgms_to_top_bottom_tensor(
        all_dgms,
        device=device
    )

    pd_result = combine_top_bottom(pd_dict)
    pd_tensor = pd_result["pd_tensor"]
    pd_mask = pd_result["pd_mask"]

    pd_tensor = pd_tensor.reshape(
        B,
        n_patches,
        *pd_tensor.shape[1:]
    )

    pd_mask = pd_mask.reshape(
        B,
        n_patches,
        *pd_mask.shape[1:]
    )

    result = {
        "pd_tensor": pd_tensor.squeeze(0),
        "pd_mask": pd_mask.squeeze(0),
        "max_points": pd_result["max_points"]
    }

    torch.save(result, cache_path)

    return result