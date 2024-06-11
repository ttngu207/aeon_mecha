from collections import deque

import numpy as np
import pandas as pd


def resolve_duplicate_identities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reassign ID of the row with duplicated ID and lower likelihood.

    This function checks for duplicated ``identity_name`` for each
    unique DatetimeIndex, and randomly assigns another
    ``identity_name`` from the available identity names in the
    DataFrame to the row having duplicated identity and lower
    likelihood. This function is useful in a 2-subject case, but
    is not guaranteed to work in a >2-subject case.

    Args:
        df (pandas.DataFrame): DataFrame with columns
            ``identity_name`` and ``likelihood``.

    Returns:
        pandas.DataFrame: DataFrame without duplicate identities
            per unique DatetimeIndex.
    """
    df_cp = df.reset_index().copy()
    names = df_cp["identity_name"].unique()
    # Mask for rows with multiple assignments of the same ID at the same time
    many_to_one_mask = df_cp.groupby(["time", "identity_name"]).transform("size") > 1
    duplicated_data = df_cp.loc[many_to_one_mask]
    # Indices for rows with lower likelihood
    low_likelihood_idx = duplicated_data.loc[
        ~duplicated_data.index.isin(
            duplicated_data.groupby(["time", "identity_name"])["likelihood"].idxmax()
        )
    ].index
    # This assigns another class randomly (in 2-animal case, it's the other animal,
    # but in >2-animal case, it may assign duplicate IDs again)
    df_cp.loc[low_likelihood_idx, "identity_name"] = df_cp.loc[
        low_likelihood_idx
    ].apply(lambda x: np.random.choice(names[names != x["identity_name"]]), axis=1)
    return df_cp.set_index("time")


def compute_class_speed(df: pd.DataFrame) -> pd.Series:
    """Compute the instantaneous speed of each class.

    Args:
        df (pandas.DataFrame): DataFrame with columns ``x``,
            ``y``, ``identity_name``, and DatetimeIndex ``time``.

    Returns:
        pandas.Series: Series with the instantaneous speed of each class.
    """
    return (
        df.groupby("identity_name")[["x", "y"]].diff().apply(np.linalg.norm, axis=1)
        / df.reset_index()
        .groupby("identity_name")["time"]
        .diff()
        .dt.total_seconds()
        .values
    )


def compute_speed_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Compute the boolean mask of rows with ``speed`` exceeding threshold.

    Args:
        df (pandas.DataFrame): DataFrame with columns ``speed``.
        threshold (float): Speed threhold.

    Returns:
        pandas.Series: Boolean mask of rows with ``speed`` greater
            than threshold.
    """
    speed_mask = (np.isfinite(df["speed"].values)) & (df["speed"] > threshold)
    # select only rows when more than 1 subject has speed > threshold
    speed_mask &= speed_mask.groupby(level=0).transform("sum") > 1
    return speed_mask


def resolve_swapped_identities(
    df: pd.DataFrame, threshold: float = 700.0, max_window_length: int = 6
) -> pd.DataFrame:
    """
    Reassign ID of the row with identity swaps.

    This function attempts to identify windows of identity swaps
    based on pairs of speed "violations". The windows are incremented
    by a factor of 2 each iteration, starting at a minimum
    of 3s, up to the maximum duration specified by ``max_window_length``
    seconds. Within each window, the identity of the subjects are
    randomly assigned to the other subject's identity. This method
    will not resolve all identity swaps, especially if the swaps occur
    for extended durations. It also does not account for more than
    2 subjects.

    Args:
        df (pandas.DataFrame): DataFrame with columns ``x``,
            ``y``, ``identity_name``, and DatetimeIndex ``time``.
        threshold (float): Speed threshold. Default is 700.0.
        max_window_length (int): Maximum duration in seconds for swapping
            identities. Potential swaps spanning longer durations will be
            ignored. Default is 6 seconds.

    Returns:
        pandas.DataFrame: DataFrame with resolved identity swaps within
            the specified ``max_window_length``.
    """
    df["speed"] = compute_class_speed(df)
    speed_mask = compute_speed_mask(df, threshold=threshold)
    names = df["identity_name"].unique()
    timedelta = 3
    iter = 0
    # limit swap window duration to 3 * 2**(max_iter) = max_window_length seconds
    max_iter = np.sqrt((max_window_length // 3) / 2)
    while speed_mask.sum() > 2 and iter <= max_iter:
        print(f"Iteration {iter}: {speed_mask.sum()} rows with speed > {threshold}")
        q = deque(df[speed_mask].index.unique())
        while q:
            start = q.popleft()
            try:
                end = q[0]
            except IndexError:
                break
            # compute timedelta between start and end
            # ignore if timedelta is more than t seconds
            if (end - start) > pd.Timedelta(timedelta, unit="s"):
                continue
            end = q.popleft()
            # ``end`` needs to be exclusive
            end = df.index[df.index < end].max()
            df.loc[start:end, "identity_name"] = df.loc[start:end].apply(
                lambda x: np.random.choice(names[names != x["identity_name"]]), axis=1
            )
        # recompute speed and speed_mask
        df["speed"] = compute_class_speed(df)
        speed_mask = compute_speed_mask(df, threshold=threshold)
        # update timedelta
        timedelta *= 2
        # update iter count
        iter += 1
    return df.drop(columns=["speed"])