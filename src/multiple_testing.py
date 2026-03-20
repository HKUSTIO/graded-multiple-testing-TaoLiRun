from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    p_values = np.asarray(p_values)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    threshold = alpha / float(m)
    return p_values <= threshold


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    p_values = np.asarray(p_values)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    k_max = 0
    for k in range(1, m + 1):
        if sorted_p[k - 1] > alpha / (m - k + 1):
            k_max = k - 1
            break
    else:
        k_max = m
    out = np.zeros(m, dtype=bool)
    out[order[:k_max]] = True
    return out


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    p_values = np.asarray(p_values)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    k_max = 0
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= (k / m) * alpha:
            k_max = k
            break
    out = np.zeros(m, dtype=bool)
    out[order[:k_max]] = True
    return out


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    p_values = np.asarray(p_values)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    h_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    k_max = 0
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= (k / m) * (alpha / h_m):
            k_max = k
            break
    out = np.zeros(m, dtype=bool)
    out[order[:k_max]] = True
    return out


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    rejections_null = np.asarray(rejections_null, dtype=bool)
    if rejections_null.size == 0:
        return 0.0
    any_reject = np.any(rejections_null, axis=1)
    return float(np.mean(any_reject))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    rejections = np.asarray(rejections, dtype=bool)
    is_true_null = np.asarray(is_true_null, dtype=bool)
    n_rej = int(np.sum(rejections))
    if n_rej == 0:
        return 0.0
    false_discoveries = int(np.sum(rejections & is_true_null))
    return float(false_discoveries) / float(n_rej)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    rejections = np.asarray(rejections, dtype=bool)
    is_true_null = np.asarray(is_true_null, dtype=bool)
    false_null = ~is_true_null
    n_false = int(np.sum(false_null))
    if n_false == 0:
        return 0.0
    return float(np.sum(rejections & false_null)) / float(n_false)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    unc_rows: list[np.ndarray] = []
    bonf_rows: list[np.ndarray] = []
    holm_rows: list[np.ndarray] = []

    for _, g in null_pvalues.groupby("sim_id", sort=True):
        g = g.sort_values("hypothesis_id")
        p = g["p_value"].to_numpy(dtype=float)
        unc_rows.append(p <= alpha)
        bonf_rows.append(bonferroni_rejections(p_values=p, alpha=alpha))
        holm_rows.append(holm_rejections(p_values=p, alpha=alpha))

    if unc_rows:
        unc_m = np.stack(unc_rows, axis=0)
        bonf_m = np.stack(bonf_rows, axis=0)
        holm_m = np.stack(holm_rows, axis=0)
    else:
        unc_m = np.zeros((0, 0), dtype=bool)
        bonf_m = np.zeros((0, 0), dtype=bool)
        holm_m = np.zeros((0, 0), dtype=bool)

    fdr_unc: list[float] = []
    fdr_bh: list[float] = []
    fdr_by: list[float] = []
    pow_unc: list[float] = []
    pow_bh: list[float] = []
    pow_by: list[float] = []

    for _, g in mixed_pvalues.groupby("sim_id", sort=True):
        g = g.sort_values("hypothesis_id")
        p = g["p_value"].to_numpy(dtype=float)
        is_true = g["is_true_null"].to_numpy(dtype=bool)

        rej_unc = p <= alpha
        rej_bh = benjamini_hochberg_rejections(p_values=p, alpha=alpha)
        rej_by = benjamini_yekutieli_rejections(p_values=p, alpha=alpha)

        fdr_unc.append(compute_fdr(rejections=rej_unc, is_true_null=is_true))
        fdr_bh.append(compute_fdr(rejections=rej_bh, is_true_null=is_true))
        fdr_by.append(compute_fdr(rejections=rej_by, is_true_null=is_true))
        pow_unc.append(compute_power(rejections=rej_unc, is_true_null=is_true))
        pow_bh.append(compute_power(rejections=rej_bh, is_true_null=is_true))
        pow_by.append(compute_power(rejections=rej_by, is_true_null=is_true))

    n_mixed = len(fdr_unc)
    mean_fdr_unc = float(np.mean(fdr_unc)) if n_mixed else 0.0
    mean_fdr_bh = float(np.mean(fdr_bh)) if n_mixed else 0.0
    mean_fdr_by = float(np.mean(fdr_by)) if n_mixed else 0.0
    mean_pow_unc = float(np.mean(pow_unc)) if n_mixed else 0.0
    mean_pow_bh = float(np.mean(pow_bh)) if n_mixed else 0.0
    mean_pow_by = float(np.mean(pow_by)) if n_mixed else 0.0

    return {
        "fwer_uncorrected": compute_fwer(rejections_null=unc_m),
        "fwer_bonferroni": compute_fwer(rejections_null=bonf_m),
        "fwer_holm": compute_fwer(rejections_null=holm_m),
        "fdr_uncorrected": mean_fdr_unc,
        "fdr_bh": mean_fdr_bh,
        "fdr_by": mean_fdr_by,
        "power_uncorrected": mean_pow_unc,
        "power_bh": mean_pow_bh,
        "power_by": mean_pow_by,
    }
