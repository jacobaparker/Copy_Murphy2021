"""Nested cross-fitted predictive information for a binary response."""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

EPS = np.finfo(float).eps


def _arrays(X, R):
    X = np.asarray(X, float)
    R = np.asarray(R)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or R.ndim != 1 or len(X) != len(R):
        raise ValueError("X must be (n_trials, n_features) and R must be (n_trials,).")
    if len(R) < 20 or np.isinf(X).any() or pd.isna(R).any():
        raise ValueError("Need >=20 rows, finite-or-NaN X, and nonmissing R.")
    labels = np.unique(R)
    if len(labels) != 2:
        raise ValueError(f"R must contain exactly two classes; got {labels!r}.")
    return X, (R == labels[1]).astype(int)


def _entropy(p):
    return 0.0 if p in (0.0, 1.0) else float(-p*np.log2(p)-(1-p)*np.log2(1-p))


def _ce(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return float(-np.mean(y*np.log2(p)+(1-y)*np.log2(1-p)))


def _normalize(labels, n):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n or pd.isna(labels).any():
        raise ValueError("Fold labels must be nonmissing and have length len(R).")
    _, labels = np.unique(labels, return_inverse=True)
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two folds are required.")
    return labels.astype(int)


def _index_labels(spec, n):
    arr = np.asarray(spec)
    if arr.ndim == 1 and len(arr) == n and arr.dtype != object:
        return _normalize(arr, n)
    labels = np.full(n, -1, int)
    try:
        specs = list(spec)
    except TypeError as exc:
        raise ValueError("fold_indices must be labels or test-index arrays.") from exc
    for fold, idx in enumerate(specs):
        idx = np.asarray(idx, int)
        if idx.ndim != 1 or np.any((idx < 0) | (idx >= n)) or np.any(labels[idx] >= 0):
            raise ValueError(f"Invalid or overlapping indices in fold {fold}.")
        labels[idx] = fold
    if np.any(labels < 0):
        raise ValueError("Arbitrary folds must cover each row exactly once.")
    return _normalize(labels, n)


def _interval_labels(intervals, n, n_splits):
    if intervals is None:
        labels = np.empty(n, int)
        for fold, idx in enumerate(np.array_split(np.arange(n), n_splits)):
            labels[idx] = fold
        return labels
    arr = np.asarray(intervals)
    if arr.ndim == 1 and len(arr) == n and arr.dtype != object:
        return _normalize(arr, n)
    labels = np.full(n, -1, int)
    for fold, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError("Intervals must be half-open (start, stop) pairs.")
        start, stop = map(int, interval)
        if not 0 <= start < stop <= n or np.any(labels[start:stop] >= 0):
            raise ValueError(f"Invalid or overlapping interval {(start, stop)}.")
        labels[start:stop] = fold
    if np.any(labels < 0):
        raise ValueError("Intervals must cover every row exactly once.")
    return _normalize(labels, n)


def _group_labels(groups, n_splits, seed):
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("fold_groups must be one-dimensional.")
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError(f"Need {n_splits} groups; got {len(unique)}.")
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique)); rng.shuffle(order)
    order = order[np.argsort(-counts[order], kind="stable")]
    sizes, assignment = np.zeros(n_splits, int), np.empty(len(unique), int)
    for group in order:
        target = int(np.argmin(sizes)); assignment[group] = target; sizes[target] += counts[group]
    return assignment[inverse]


def _outer_labels(y, mode, n_splits, seed, intervals, fold_indices, fold_groups):
    mode, n = mode.lower(), len(y)
    if not 2 <= n_splits <= n:
        raise ValueError("n_splits must be between 2 and len(R).")
    if mode == "random":
        if np.min(np.bincount(y)) < n_splits:
            raise ValueError("Each class needs at least n_splits observations.")
        labels = np.empty(n, int)
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for fold, (_, test) in enumerate(cv.split(np.zeros(n), y)): labels[test] = fold
        return labels
    if mode in {"interval", "intervals", "designated_intervals"}:
        return _interval_labels(intervals, n, n_splits)
    if mode in {"index", "indices", "arbitrary_indices"}:
        if fold_indices is None: raise ValueError("fold_indices is required.")
        return _index_labels(fold_indices, n)
    if mode in {"group", "groups"}:
        if fold_groups is None or len(fold_groups) != n: raise ValueError("fold_groups is required per row.")
        return _group_labels(fold_groups, n_splits, seed)
    raise ValueError("fold_mode must be random, intervals, indices, or groups.")


def default_model_library(n_features, *, random_state=0, include_splines=True,
                          max_spline_features=30):
    """A compact library suited to roughly 1,500 trials."""
    models = []
    for C in (0.1, 1.0, 10.0):
        models.append((f"logistic_l2_C={C:g}", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=3000))])))
    if include_splines and n_features <= max_spline_features:
        models.append(("additive_spline_logistic", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=3000))])))
    for leaves in (7, 15):
        models.append((f"hist_gradient_boosting_leaves={leaves}", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=.05, max_iter=150, max_leaf_nodes=leaves,
                min_samples_leaf=20, l2_regularization=1., early_stopping=False,
                random_state=random_state))])))
    return models


def _validate_models(models):
    models = list(models)
    if not models or len({n for n, _ in models}) != len(models):
        raise ValueError("model_library must have uniquely named candidates.")
    for name, est in models:
        if not isinstance(name, str) or not hasattr(est, "fit") or not hasattr(est, "predict_proba"):
            raise TypeError("Each candidate must be (name, sklearn probabilistic estimator).")
    return models


def _inner_splits(y, desired, seed, groups):
    if groups is None:
        k = min(desired, int(np.min(np.bincount(y))))
        if k < 2: raise ValueError("Too few examples of a class for inner CV.")
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    k = min(desired, len(np.unique(groups)))
    if k < 2: raise ValueError("Too few intact blocks for inner CV.")
    return list(GroupKFold(k).split(np.zeros(len(y)), y, groups))


def _select(X, y, models, splits):
    rows = []
    for name, estimator in models:
        losses, error = [], ""
        try:
            for train, valid in splits:
                if len(np.unique(y[train])) != 2: raise ValueError("inner training fold has one class")
                fitted = clone(estimator).fit(X[train], y[train])
                losses.append(_ce(y[valid], fitted.predict_proba(X[valid])[:, 1]))
        except Exception as exc:
            losses, error = [], f"{type(exc).__name__}: {exc}"
        rows.append({"candidate": name,
                     "mean_inner_log_loss_bits": np.mean(losses) if losses else np.nan,
                     "sd_inner_log_loss_bits": np.std(losses, ddof=1) if len(losses)>1 else np.nan,
                     "inner_folds_scored": len(losses), "error": error})
    scores = pd.DataFrame(rows)
    valid = scores.mean_inner_log_loss_bits.notna()
    if not valid.any(): raise RuntimeError(f"All candidate models failed: {scores[['candidate','error']].to_dict('records')}")
    best = scores.loc[scores.loc[valid, "mean_inner_log_loss_bits"].idxmin(), "candidate"]
    return best, dict(models)[best], scores


def _reliability(y, p, n_bins):
    try: bins = pd.qcut(p, q=max(2, min(n_bins, len(y))), duplicates="drop")
    except ValueError: bins = pd.cut(p, np.linspace(0, 1, n_bins+1), include_lowest=True)
    out = pd.DataFrame({"observed": y, "predicted": p, "bin": bins}).groupby("bin", observed=True).agg(
        n=("observed", "size"), mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"), min_predicted=("predicted", "min"),
        max_predicted=("predicted", "max")).reset_index(drop=True)
    out["abs_calibration_error"] = abs(out.observed_rate-out.mean_predicted)
    return out


def estimate_choice_information(
    X: Any, R: Any, *, fold_mode="random", n_splits=5, inner_n_splits=3,
    intervals=None, fold_indices=None, fold_groups=None,
    model_library: Sequence[tuple[str, BaseEstimator]] | None=None,
    random_state=2026, reliability_bins=10, return_estimators=False, plot=False):
    """Estimate H_hat(R) - held-out CE(q) with nested cross-fitting.

    ``fold_mode`` may be ``random``, ``intervals``, ``indices``, or ``groups``.
    For intervals, pass half-open ``[(start, stop), ...]`` or omit them for
    automatic contiguous folds. For indices, pass one fold label per row or a
    list of test-index arrays. For groups, pass one group ID per row.

    Model choice and all preprocessing occur inside each outer training set.
    The population analogue is a variational lower bound on I(X;R); this sample
    estimate may be negative and requires uncertainty analysis.
    """
    X, y = _arrays(X, R)
    folds = _outer_labels(y, fold_mode, n_splits, random_state, intervals, fold_indices, fold_groups)
    unique_folds = np.unique(folds)
    if len(unique_folds) < 3: warnings.warn("Fewer than three outer folds is fragile.", stacklevel=2)
    models = _validate_models(model_library) if model_library is not None else default_model_library(X.shape[1], random_state=random_state)
    q, q0 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_rows, selection_rows, fitted_models = [], [], []
    structured = fold_mode.lower() != "random"
    supplied_groups = np.asarray(fold_groups) if fold_mode.lower() in {"group", "groups"} else None
    for fold in unique_folds:
        test, train = np.flatnonzero(folds==fold), np.flatnonzero(folds!=fold)
        if len(np.unique(y[train])) != 2: raise ValueError(f"Outer training fold {fold} has one class.")
        inner_groups = supplied_groups[train] if supplied_groups is not None else (folds[train] if structured else None)
        seed = None if random_state is None else int(random_state)+int(fold)+1
        splits = _inner_splits(y[train], inner_n_splits, seed, inner_groups)
        best, estimator, scores = _select(X[train], y[train], models, splits)
        scores.insert(0, "outer_fold", int(fold)); selection_rows.append(scores)
        fitted = clone(estimator).fit(X[train], y[train]); q[test] = fitted.predict_proba(X[test])[:,1]
        pi = (y[train].sum()+.5)/(len(train)+1.); q0[test] = pi
        model_ce, null_ce = _ce(y[test], q[test]), _ce(y[test], q0[test])
        fold_rows.append({"outer_fold": int(fold), "n_train": len(train), "n_test": len(test),
            "test_response_rate": y[test].mean(), "selected_model": best,
            "best_inner_log_loss_bits": scores.loc[scores.candidate==best, "mean_inner_log_loss_bits"].iloc[0],
            "model_test_log_loss_bits": model_ce, "intercept_test_log_loss_bits": null_ce,
            "test_log_score_gain_bits": null_ce-model_ce})
        if return_estimators: fitted_models.append(fitted)
    if np.isnan(q).any(): raise RuntimeError("Some rows did not receive OOF predictions.")
    p, h, ce, ce0 = y.mean(), _entropy(float(y.mean())), _ce(y,q), _ce(y,q0)
    rel = _reliability(y,q,reliability_bins)
    logits = np.log(np.clip(q,EPS,1-EPS)/np.clip(1-q,EPS,1-EPS))
    cal = LogisticRegression(C=1e6, max_iter=3000).fit(logits[:,None], y)
    per_fold = pd.DataFrame(fold_rows)
    result = {"mi_lower_bound_bits": h-ce, "response_entropy_bits": h,
        "conditional_cross_entropy_bits": ce, "intercept_cv_cross_entropy_bits": ce0,
        "cv_log_score_gain_vs_intercept_bits": ce0-ce, "response_rate": float(p),
        "brier_score": brier_score_loss(y,q), "intercept_brier_score": brier_score_loss(y,q0),
        "roc_auc": roc_auc_score(y,q),
        "expected_calibration_error": float(np.average(rel.abs_calibration_error, weights=rel.n)),
        "calibration_intercept_descriptive": float(cal.intercept_[0]),
        "calibration_slope_descriptive": float(cal.coef_[0,0]),
        "oof_probability": q, "oof_intercept_probability": q0, "fold_labels": folds,
        "binary_response": y, "per_fold": per_fold,
        "model_selection": pd.concat(selection_rows, ignore_index=True),
        "model_selection_counts": pd.Series(Counter(per_fold.selected_model), name="outer_folds_selected").sort_values(ascending=False),
        "reliability": rel, "settings": {"fold_mode": fold_mode, "n_outer_folds": len(unique_folds),
        "inner_n_splits_requested": inner_n_splits, "random_state": random_state,
        "n_trials": len(y), "n_features": X.shape[1]}}
    if return_estimators: result["estimators"] = fitted_models
    if plot: result["figure"] = plot_choice_information_diagnostics(result)
    return result


def plot_choice_information_diagnostics(result: Mapping[str, Any]):
    y, q, rel, pf = np.asarray(result["binary_response"]), np.asarray(result["oof_probability"]), result["reliability"], result["per_fold"]
    fig, axes = plt.subplots(1,3,figsize=(15,4.2))
    axes[0].plot([0,1],[0,1],"--",color=".55"); axes[0].plot(rel.mean_predicted,rel.observed_rate,"o-")
    axes[0].set(xlabel="Mean predicted P(R=1)",ylabel="Observed response rate",xlim=(0,1),ylim=(0,1))
    bins=np.linspace(0,1,21); axes[1].hist(q[y==0],bins=bins,alpha=.65,label="R=0",density=True); axes[1].hist(q[y==1],bins=bins,alpha=.65,label="R=1",density=True)
    axes[1].set(xlabel="OOF predicted P(R=1)",ylabel="Density"); axes[1].legend(frameon=False)
    x=np.arange(len(pf)); w=.38; axes[2].bar(x-w/2,pf.model_test_log_loss_bits,w,label="selected model"); axes[2].bar(x+w/2,pf.intercept_test_log_loss_bits,w,label="intercept")
    axes[2].set(xticks=x,xticklabels=pf.outer_fold.astype(str),xlabel="Outer fold",ylabel="Held-out log loss (bits)"); axes[2].legend(frameon=False)
    fig.suptitle(f"Cross-fitted choice information: {result['mi_lower_bound_bits']:.4f} bits/trial"); fig.tight_layout(); return fig


def _resample(rng, n, groups):
    if groups is None: return rng.integers(0,n,size=n)
    unique=np.unique(groups); sampled=rng.choice(unique,size=len(unique),replace=True)
    return np.concatenate([np.flatnonzero(groups==g) for g in sampled])


def bootstrap_choice_information(
    X: Any, R: Any, *, n_bootstrap=200, confidence=.95, resample_groups=None,
    estimator_kwargs: Mapping[str,Any] | None=None, random_state=2027,
    n_jobs=1, min_success_fraction=.8, plot=False):
    """Rerun the complete nested-CV estimator in a trial or cluster bootstrap.

    Duplicate copies of an original trial/cluster remain in one outer fold, so
    a duplicate never appears directly in both training and test data.
    """
    X,y=_arrays(X,R)
    if n_bootstrap<2 or not 0<confidence<1: raise ValueError("Need n_bootstrap>=2 and 0<confidence<1.")
    kwargs=dict(estimator_kwargs or {}); kwargs.pop("plot",None); kwargs.pop("return_estimators",None)
    point=estimate_choice_information(X,y,**kwargs); k=point["settings"]["n_outer_folds"]
    groups=None if resample_groups is None else np.asarray(resample_groups)
    if groups is not None and (groups.ndim!=1 or len(groups)!=len(y)): raise ValueError("resample_groups must have one label per row.")
    source_folds=np.asarray(point["fold_labels"]) if groups is None else _group_labels(groups,k,random_state)
    seeds=np.random.SeedSequence(random_state).spawn(n_bootstrap)
    def one(rep):
        seed=int(seeds[rep].generate_state(1)[0]); rng=np.random.default_rng(seed); idx=_resample(rng,len(y),groups)
        boot_kwargs=dict(kwargs)
        for key in ("intervals","fold_groups"): boot_kwargs.pop(key,None)
        boot_kwargs.update(fold_mode="indices",fold_indices=source_folds[idx],random_state=seed,plot=False,return_estimators=False)
        try:
            est=estimate_choice_information(X[idx],y[idx],**boot_kwargs)
            return {"replicate":rep,"mi_lower_bound_bits":est["mi_lower_bound_bits"],
                "conditional_cross_entropy_bits":est["conditional_cross_entropy_bits"],
                "cv_log_score_gain_vs_intercept_bits":est["cv_log_score_gain_vs_intercept_bits"],
                "brier_score":est["brier_score"],"selected_models":dict(est["model_selection_counts"].astype(int)),"error":""}
        except Exception as exc:
            return {"replicate":rep,"mi_lower_bound_bits":np.nan,"conditional_cross_entropy_bits":np.nan,
                "cv_log_score_gain_vs_intercept_bits":np.nan,"brier_score":np.nan,"selected_models":{},"error":f"{type(exc).__name__}: {exc}"}
    records=[one(i) for i in range(n_bootstrap)] if n_jobs==1 else Parallel(n_jobs=n_jobs,verbose=5)(delayed(one)(i) for i in range(n_bootstrap))
    selected=Counter()
    for row in records: selected.update(row.pop("selected_models"))
    samples=pd.DataFrame(records); valid=samples.mi_lower_bound_bits.dropna()
    if len(valid)/n_bootstrap<min_success_fraction:
        raise RuntimeError(f"Only {len(valid)}/{n_bootstrap} bootstrap replicates succeeded: {samples.error.value_counts().head(3).to_dict()}")
    if len(valid)<n_bootstrap: warnings.warn(f"{n_bootstrap-len(valid)} bootstrap fits failed; inspect bootstrap_samples.error.",stacklevel=2)
    alpha=(1-confidence)/2; low,high=np.quantile(valid,[alpha,1-alpha])
    result={"point_estimate":point,"point_mi_lower_bound_bits":point["mi_lower_bound_bits"],
        "bootstrap_standard_error_bits":float(valid.std(ddof=1)),"confidence":confidence,
        "percentile_ci_bits":(float(low),float(high)),"bootstrap_mean_bits":float(valid.mean()),
        "bootstrap_bias_bits":float(valid.mean()-point["mi_lower_bound_bits"]),
        "n_bootstrap_requested":n_bootstrap,"n_bootstrap_successful":len(valid),
        "bootstrap_samples":samples,"bootstrap_model_selection_counts":pd.Series(selected,name="outer_folds_selected").sort_values(ascending=False),
        "resampling_unit":"trials" if groups is None else "groups"}
    if plot: result["figure"]=plot_bootstrap_choice_information(result)
    return result


def plot_bootstrap_choice_information(result: Mapping[str,Any]):
    values=result["bootstrap_samples"].mi_lower_bound_bits.dropna(); low,high=result["percentile_ci_bits"]; point=result["point_mi_lower_bound_bits"]
    fig,ax=plt.subplots(figsize=(7.5,4.5)); ax.hist(values,bins="auto",alpha=.78,edgecolor="white"); ax.axvline(point,color="black",lw=2,label=f"Point: {point:.4f}"); ax.axvspan(low,high,color="tab:orange",alpha=.22,label="Percentile CI")
    ax.set(xlabel="Information lower bound (bits/trial)",ylabel="Bootstrap replicates",title=f"{100*result['confidence']:.1f}% bootstrap CI: [{low:.4f}, {high:.4f}]"); ax.legend(frameon=False); fig.tight_layout(); return fig


__all__=["default_model_library","estimate_choice_information","bootstrap_choice_information","plot_choice_information_diagnostics","plot_bootstrap_choice_information"]
