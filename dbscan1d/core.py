"""
A simple implementation of DBSCAN for 1D data.

It should be *much* more efficient for large datasets.
"""

import numba as nb
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional

__all__ = ["DBSCAN1D"]


@nb.njit(
    [
        nb.bool_[:](nb.float64[:], nb.float64, nb.int64),
        nb.bool_[::1](nb.float64[::1], nb.float64, nb.int64),
    ],
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_is_core(
    ar: NDArray[np.float64], eps: float, min_samples: int
) -> NDArray[np.bool_]:
    min_eps = np.searchsorted(ar, ar - eps, side="left")
    max_eps = np.searchsorted(ar, ar + eps, side="right")
    core = (max_eps - min_eps) >= min_samples
    return core


@nb.njit(
    [
        nb.int64[:](nb.int64[:], nb.int64),
        nb.int64[::1](nb.int64[::1], nb.int64),
    ],
    cache=True,
    nogil=True,
    fastmath=True,
)
def _bound_on(arr: NDArray[np.int64], max_len: int) -> NDArray[np.int64]:
    """Ensure all values in array are bounded between 0 and max_len."""
    arr[arr < 0] = 0
    arr[arr >= max_len] = max_len - 1
    return arr


@nb.njit(
    [
        nb.int64[:](nb.float64[:], nb.float64),
    ],
    cache=True,
    nogil=True,
    fastmath=True,
)
def _assign_core_group_numbers(
    cores: NDArray[np.floating], eps: float
) -> NDArray[np.int64]:
    """Given a group of core points, assign group numbers to each."""
    gt_eps = np.abs(cores - np.roll(cores, 1)) > eps
    # The first value doesn't need to be compared to last, set to False so
    # that cluster names are consistent (see issue #3).
    if len(gt_eps):
        gt_eps[0] = False
    return gt_eps.astype(np.int64).cumsum()


@nb.njit(
    [
        nb.int64[:](nb.float64[:], nb.float64[:], nb.int64[:], nb.float64),
    ],
    cache=True,
    nogil=True,
    fastmath=True,
)
def _get_non_core_labels(
    non_cores: NDArray[np.floating],
    cores: NDArray[np.floating],
    core_nums: NDArray[np.integer],
    eps: float,
) -> NDArray[np.integer]:
    """Get labels for non-core points."""
    # start out with noise labels (-1)
    out = (np.ones(len(non_cores)) * -1).astype(np.int64)
    if not len(cores):  # there are no core points, bail out early
        return out
    # get index where non-core point would be inserted into core points
    cc_right = np.searchsorted(cores, non_cores)
    cc_left = cc_right - 1
    # make sure these respect bounds of cores
    cc_left = _bound_on(cc_left, len(cores))
    cc_right = _bound_on(cc_right, len(cores))

    # now get index and values of closest core point (on right and left)
    core_index = np.zeros((len(cc_left), 2), dtype=np.int64)
    core_index[:, 0] = cc_left
    core_index[:, 1] = cc_right

    vals = np.zeros((len(non_cores), 2), dtype=non_cores.dtype)
    vals[:, 0] = cores[cc_left]
    vals[:, 1] = cores[cc_right]

    # calculate the difference between each non-core and its neighbor cores
    diffs = np.zeros_like(vals)
    for j in range(2):
        diffs[:, j] = np.abs(vals[:, j] - non_cores)
    min_vals = np.zeros(len(cc_left), dtype=diffs.dtype)
    inds = np.zeros_like(min_vals, dtype=core_index.dtype)
    for i in range(len(cc_left)):
        _argmin = diffs[i, :].argmin()
        min_vals[i] = diffs[i, _argmin]
        inds[i] = core_index[i, _argmin]

    # determine if closest core point is close enough to assign to group
    is_connected = min_vals <= eps
    # update group and return
    out[is_connected] = core_nums[inds[is_connected]]
    return out


@nb.njit(
    [
        nb.types.UniTuple(nb.int64[::1], 2)(nb.float64[::1], nb.float64, nb.int64),
        nb.types.UniTuple(nb.int64[::1], 2)(nb.float64[:, ::1], nb.float64, nb.int64),
    ],
    cache=True,
    nogil=True,
    fastmath=True,
)
def fit(
    X: NDArray[np.floating], eps: float, min_samples: int
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    # get sorted array and sorted order
    ar = X.flatten()
    ar_sorted = np.sort(ar)
    undo_sorted = np.argsort(np.argsort(ar))

    # get core points, and separate core from non-core
    is_core = _get_is_core(ar_sorted, eps, min_samples)
    group_nums = np.ones_like(is_core) * -1  # empty group numbers
    cores = ar_sorted[is_core]
    non_cores = ar_sorted[~is_core]
    # get core numbers and non-core numbers
    core_nums = _assign_core_group_numbers(cores, eps)
    non_core_nums = _get_non_core_labels(
        non_cores,
        cores,
        core_nums,
        eps,
    )
    group_nums[is_core] = core_nums
    group_nums[~is_core] = non_core_nums
    # unsort group nums and core indices
    out = group_nums[undo_sorted]
    is_core_original_sorting = is_core[undo_sorted]
    # set class attrs and return predicted labels
    core_sample_indices_ = np.where(is_core_original_sorting)[0]
    # self.components_ = cores.values
    labels_ = out
    return core_sample_indices_, labels_


class DBSCAN1D:
    """
    A one dimensional implementation of DBSCAN.

    This class has a very similar interface as sklearn's implementation. In
    most cases they should be interchangeable.
    """

    # params that change upon fit/training
    core_sample_indices_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None

    def __init__(
        self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
    ):
        self.eps = eps
        self.min_samples = min_samples

        if metric.lower() != "euclidean":
            msg = "only euclidean distance is supported by DBSCAN1D"
            raise ValueError(msg)

    def fit(self, X, y=None, sample_weight=None):
        """
        Performing DBSCAN clustering on 1D array.

        Parameters
        ----------
        X
            The input array
        y
            Not used
        sample_weight
            Not yet supported
        """
        assert len(X.shape) == 1 or X.shape[-1] == 1, "X must be 1d array"
        assert y is None, "y parameter is ignored"
        assert sample_weight is None, "sample weights are not yet supported"
        X = np.asarray(X, dtype=np.float64)
        self.core_sample_indices_, self.labels_ = fit(X, self.eps, self.min_samples)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Performing DBSCAN clustering on 1D array and return the label array.

        Parameters
        ----------
        X
            The input array
        y
            Not used
        sample_weight
            Not yet supported
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.labels_
