import sys
import numpy as np
import torch

sys.path.append('/root/project12/tda-toolkit')

from collections import defaultdict

import pershombox
import pershombox._software_backends.resource_handler as rh
from pershombox.toplex import toplex_persistence_diagrams


# ============================================================
# Initialize PershomBox Backends
# ============================================================

rh.init_backend(rh.Backends.dipha)
rh.init_backend(rh.Backends.perseus)
rh.init_backend(rh.Backends.hera_wasserstein_dist)


# ============================================================
# Filtration Functions
# ============================================================

def height_filtration_from_top(value):
    return -float(value)


def height_filtration_from_bottom(value):
    return float(value)


# ============================================================
# Persistence Diagram of 1D Time Series
# ============================================================

def pershom_of_timeseries(timeseries, filtration):
    """
    Parameters
    ----------
    timeseries : np.ndarray
        shape = (length,)

    filtration : function

    Returns
    -------
    diagrams
    """

    assert isinstance(timeseries, np.ndarray)
    assert timeseries.ndim == 1

    timeseries = timeseries.tolist()

    toplices = []
    filt_values = []

    for i in range(len(timeseries) - 1):

        # vertex
        toplices.append((i,))
        filt_values.append(
            filtration(timeseries[i])
        )

        # edge
        toplices.append((i, i + 1))
        filt_values.append(
            max(
                filtration(timeseries[i]),
                filtration(timeseries[i + 1])
            )
        )

    # last vertex
    toplices.append((len(timeseries) - 1,))
    filt_values.append(
        filtration(timeseries[-1])
    )

    diagrams = toplex_persistence_diagrams(
        toplices=toplices,
        filtration_values=filt_values,
        deessentialize=True
    )

    return diagrams


# ============================================================
# Single Sample
# ============================================================

def compute_sample_dgms(data):
    """
    Parameters
    ----------
    data : np.ndarray
        shape = (seq_len, d_dim)

    Returns
    -------
    dgms
    """

    dgms = defaultdict(list)

    seq_len, d_dim = data.shape

    for i_sensor in range(d_dim):

        signal = data[:, i_sensor]

        signal = (
            signal - signal.mean()
        ) / (
            signal.std() + 1e-8
        )

        try:

            top_dgm = pershom_of_timeseries(
                signal,
                height_filtration_from_top
            )[0]

            bottom_dgm = pershom_of_timeseries(
                signal,
                height_filtration_from_bottom
            )[0]

        except Exception as e:

            print(
                f"[WARNING] Persistence failed "
                f"sensor={i_sensor}"
            )
            print(e)

            top_dgm = []
            bottom_dgm = []

        dgms["top"].append(top_dgm)
        dgms["bottom"].append(bottom_dgm)

    return dgms


# ============================================================
# Batch Processing
# ============================================================

def batch_job(x):
    """
    Parameters
    ----------
    x : torch.Tensor or np.ndarray

    shape:
        (batch, seq_len, d_dim)

    Returns
    -------
    all_dgms : list

    all_dgms[b]['top'][i]
    all_dgms[b]['bottom'][i]
    """

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    assert isinstance(x, np.ndarray)
    assert x.ndim == 3

    batch_size = x.shape[0]

    all_dgms = []

    for b in range(batch_size):

        sample = x[b]

        dgms = compute_sample_dgms(sample)

        all_dgms.append(dgms)

    return all_dgms
