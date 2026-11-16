"""
Example Usage of polars_ols_reg plugin
"""

import polars as pl
from polars_ols_reg import ols_reg

# 1. Create a synthetic panel dataset with groups
data = pl.DataFrame({
    "var": ["A"] * 50 + ["B"] * 50,
    "x1": [float(i % 10) for i in range(100)],
    "x2": [float((i * 3) % 7) for i in range(100)],
    "y": [2.0 + 1.5 * (i % 10) - 0.5 * ((i * 3) % 7) + 0.1 * (i % 5) for i in range(100)],
})

print("Sample Data:")
print(data.head(6))

# 2. DEFAULT BEHAVIOR: returns everything (coefficients, t-values, p-values, and adjr2)
print("\n--- 1. Default Behavior (Returns All: b_*, t_*, p_*, adjr2) ---")
all_df = (
    data.group_by("var")
    .agg(est=ols_reg("y ~ x1+x2", "nw+3"))
    .unnest("est")
)
print(all_df)
print("Returned column names:", all_df.columns)

# 3. Only Coefficients
print("\n--- 2. Only Coefficients (Newey-West, 3 lags) ---")
coeff_df = (
    data.group_by("var")
    .agg(est=ols_reg("y ~ x1+x2", "nw3", "coeff"))
    .unnest("est")
)
print(coeff_df)

# 4. Only t-values
print("\n--- 3. Only t-values (Newey-West, 3 lags) ---")
tval_df = (
    data.group_by("var")
    .agg(est=ols_reg("y ~ x1+x2", "nw3", "tvalue"))
    .unnest("est")
)
print(tval_df)

# 5. Only p-values
print("\n--- 4. Only p-values (White standard errors by default) ---")
pval_df = (
    data.group_by("var")
    .agg(est=ols_reg("y ~ x1+x2", "white", "pvalue"))
    .unnest("est")
)
print(pval_df)

# 6. Only Adjusted R-squared
print("\n--- 5. Only Adjusted R-squared ---")
adjr2_df = (
    data.group_by("var")
    .agg(adjr2=ols_reg("y ~ x1+x2", "nw3", "adjr2"))
)
print(adjr2_df)

