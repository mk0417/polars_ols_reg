"""
Polars OLS Regression Plugin
============================
High-performance OLS regression plugin for Polars with White (HC0)
and Newey-West (HAC) robust standard errors.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import types
from typing import Any
import polars as pl
from polars.plugins import register_plugin_function

PLUGIN_PATH = Path(__file__).parent

_RESULT_MAP: dict[str, str] = {
    # All (default): slopes, t-values, p-values, and adjr2
    "all": "ols_all",
    "full": "ols_all",
    # Coefficients / slopes
    "coeff": "ols_coeff",
    "coeffs": "ols_coeff",
    "coef": "ols_coeff",
    "beta": "ols_coeff",
    "slope": "ols_coeff",
    "slopes": "ols_coeff",
    # t-values
    "tvalue": "ols_tvalue",
    "tvalues": "ols_tvalue",
    "t": "ols_tvalue",
    "tstat": "ols_tvalue",
    "t_stat": "ols_tvalue",
    # p-values
    "pvalue": "ols_pvalue",
    "pvalues": "ols_pvalue",
    "p": "ols_pvalue",
    "pval": "ols_pvalue",
    "p_val": "ols_pvalue",
    # Adjusted R2
    "adjr2": "ols_adjr2",
    "adj_r2": "ols_adjr2",
    "r2": "ols_adjr2",
}


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Parse regression formula such as 'y ~ x1 + x2'."""
    if "~" not in formula:
        raise ValueError(
            f"Invalid regression formula '{formula}'. Must have the form 'y ~ x1 + x2'."
        )

    lhs, rhs = formula.split("~", 1)
    y = lhs.strip()
    if not y:
        raise ValueError("Dependent variable (left-hand side of '~') cannot be empty.")

    raw_terms = [t.strip() for t in rhs.split("+")]
    terms: list[str] = []
    for t in raw_terms:
        if not t:
            continue
        if t in ("1", "const", "intercept"):
            # Intercept is automatically included by default
            continue
        if t in ("0", "-1"):
            raise ValueError(
                "Regression without intercept is currently not supported."
            )
        terms.append(t)

    if not terms:
        raise ValueError(
            "Formula must include at least one independent regressor (e.g. 'y ~ x1')."
        )

    # Validate regressors and duplicates
    seen: set[str] = set()
    for t in terms:
        if t == y:
            raise ValueError(
                f"Dependent variable '{y}' cannot also be an explanatory variable."
            )
        if t in seen:
            raise ValueError(f"Duplicate regressor '{t}' in formula.")
        seen.add(t)

    return y, terms


def _parse_cov_type(cov_type: str | None) -> int:
    """
    Parse covariance estimator type into Newey-West lag length.
    Returns 0 for White (HC0) standard errors.
    """
    if cov_type is None:
        return 0

    cov = cov_type.strip().lower()
    if cov in ("white", "hc0", "hc", "robust"):
        return 0

    # Match patterns such as nw+3, nw3, nw_3, nw-3, nw(3)
    m = re.fullmatch(r"nw(?:[\+_\-\(])?(\d+)\)?", cov)
    if m:
        return int(m.group(1))

    if cov in ("nw", "newey-west", "newey_west", "hac"):
        # Default lag if nw is specified without number
        return 1

    raise ValueError(
        f"Unsupported cov_type '{cov_type}'. Use 'white' or 'nw+<lags>' (e.g. 'nw+3')."
    )


def ols_reg(
    formula: str,
    cov_type: str = "white",
    result: str = "all",
) -> pl.Expr:
    """
    Polars expression plugin for OLS regression.

    Parameters
    ----------
    formula : str
        Formula of the form 'y ~ x1 + x2'. An intercept (const) is automatically included.
    cov_type : str, default 'white'
        Standard error type:
        - 'white': White (HC0) heteroskedasticity-robust standard errors (default).
        - 'nw+N' (e.g. 'nw+3'): Newey-West (HAC) standard errors with N lags.
    result : str, default 'all'
        Estimation target to return:
        - 'all' (default): Returns complete regression summary including:
          b_const, b_<x1>..., t_const, t_<x1>..., p_const, p_<x1>..., adjr2.
        - 'coeff': Returns struct with b_const, b_<x1>...
        - 'tvalue': Returns struct with t_const, t_<x1>...
        - 'pvalue': Returns struct with p_const, p_<x1>...
        - 'adjr2': Returns scalar Float64 adjusted R-squared.

    Returns
    -------
    pl.Expr
        Expression that can be aggregated in `df.group_by(...).agg(...)`
        or evaluated in `df.select(...)`.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_ols_reg import ols_reg
    >>>
    >>> # 1. Default: returns everything (coefficients, t-values, p-values, adjr2)
    >>> df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw+3')).unnest('est')
    >>> # Output columns: var, b_const, b_x1, b_x2, t_const, t_x1, t_x2, p_const, p_x1, p_x2, adjr2
    >>>
    >>> # 2. Only coefficients
    >>> df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw+3', 'coeff')).unnest('est')
    >>>
    >>> # 3. Only t-values
    >>> df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw+3', 'tvalue')).unnest('est')
    >>>
    >>> # 4. Only p-values
    >>> df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'white', 'pvalue')).unnest('est')
    >>>
    >>> # 5. Adjusted R-squared
    >>> df.group_by('var').agg(adjr2=ols_reg('y ~ x1+x2', 'nw+3', 'adjr2'))
    """
    # Gracefully handle when user omits cov_type and passes result in 2nd position
    cov_lower = cov_type.strip().lower() if isinstance(cov_type, str) else ""
    if cov_lower in _RESULT_MAP and result == "all":
        result = cov_lower
        cov_type = "white"

    res_key = result.strip().lower()
    if res_key not in _RESULT_MAP:
        valid_options = ["all", "coeff", "tvalue", "pvalue", "adjr2"]
        raise ValueError(
            f"Invalid result '{result}'. Supported options: {valid_options}"
        )

    fn_name = _RESULT_MAP[res_key]
    nw_lags = _parse_cov_type(cov_type)
    y_col, x_cols = _parse_formula(formula)

    args = [pl.col(y_col)] + [pl.col(x) for x in x_cols]

    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name=fn_name,
        args=args,
        kwargs={"nw_lags": nw_lags, "names": x_cols},
        is_elementwise=False,
        returns_scalar=True,
    )


# ---------------------------------------------------------------------------
# Allow calling polars_ols_reg directly if imported as:
# `import polars_ols_reg as ols_reg`
# ---------------------------------------------------------------------------
class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> pl.Expr:
        return ols_reg(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["ols_reg"]
