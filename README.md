# polars_ols_reg: high-performance OLS plugin for Polars

This is a simple tool for running regression using **Polars**. It provides:

- R-style expression (e.g. `y ~ x`)
- Point estimates (coefficients)
- $t$-statistics and $p$-values
- Adjusted $R^2$
- Robust covariance estimators:
  - White (HC0) heteroskedasticity-consistent standard errors (default)
  - Newey-West (HAC) heteroskedasticity and autocorrelation consistent standard errors with user-configurable lag length (e.g. `'nw3'`)
- High-performance multi-threaded parallel regressions across large numbers of groups (`ols_reg_group_parallel`)

I used **Antigravity CLI** + **Gemini 3.8 Flash (High)** to develop this package.

This package might not fit your workflow as it does not ship
comprehensive features.

I probably not publish it on PyPI. Please install it from the
available links in release page if you want to use it. For example,
for MacOS,

```
pip install https://github.com/mk0417/polars_ols_reg/releases/download/v0.3.0/polars_ols_reg-0.3.0-cp311-abi3-macosx_11_0_arm64.whl
```

---

## 1. Quickstart & Usage Syntax

```python
import polars as pl
from polars_ols_reg import ols_reg

# Sample panel DataFrame
df = pl.DataFrame({
    'var': ['A']*50+['B']*50,
    'x1': [float(i%10) for i in range(100)],
    'x2': [float((i*3)%7) for i in range(100)],
    'y': [1.5+2*(i%10)-0.8*((i*3)%7)+0.1*(i%5) for i in range(100)]})
```

### 1) Default Behavior: Returns Everything
If the 3rd parameter is ignored, the plugin returns **all** statistics at once:
```python
df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw3')).unnest('est')
```
**Output columns:**
`var`, `b_const`, `b_x1`, `b_x2`, `t_const`, `t_x1`, `t_x2`, `p_const`, `p_x1`, `p_x2`, `adjr2`

### 2) Only Estimated Coefficients
```python
df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw3', 'coeff')).unnest('est')
```
**Output columns:** `var`, `b_const`, `b_x1`, `b_x2`

### 3) Only $t$-values
```python
df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'nw3', 'tvalue')).unnest('est')
```
**Output columns:** `var`, `t_const`, `t_x1`, `t_x2`

### 4) Only $p$-values
```python
df.group_by('var').agg(est=ols_reg('y ~ x1+x2', 'white', 'pvalue')).unnest('est')
```
**Output columns:** `var`, `p_const`, `p_x1`, `p_x2`

### 5) Only Adjusted $R^2$
```python
df.group_by('var').agg(adjr2=ols_reg('y ~ x1+x2', 'nw3', 'adjr2'))
```
**Output columns:** `var`, `adjr2`

---

## 2. API Reference

```python
ols_reg(
    formula: str,
    cov_type: str = 'white',
    result: str = 'all',
) -> pl.Expr
```

### Parameters:
- **`formula`** (`str`):
  Regression specification in standard R / statsmodels style: `'y ~ x1 + x2'`. An intercept (`const`) is automatically added.
- **`cov_type`** (`str`, default: `'white'`):
  - `'white'` or `'hc0'`: White (1980) heteroskedasticity-consistent covariance estimator.
  - `'nwN'` (e.g. `'nw3'`, `'nw5'`): Newey-West (1987) HAC covariance estimator with $N$ lags.
  - If omitted, defaults to `'white'`.
- **`result`** (`str`, default: `'all'`):
  - `'all'` (default): Returns complete regression summary (`b_*`, `t_*`, `p_*`, `adjr2`).
  - `'coeff'` (aliases: `'beta'`, `'slopes'`): Returns struct with fields `b_const, b_x1, b_x2, ...`
  - `'tvalue'` (aliases: `'t'`, `'tstat'`): Returns struct with fields `t_const, t_x1, t_x2, ...`
  - `'pvalue'` (aliases: `'p'`, `'pval'`): Returns struct with fields `p_const, p_x1, p_x2, ...`
  - `'adjr2'` (aliases: `'r2'`): Returns scalar with adjusted $R^2$.

---

## 3. High-Performance Multi-Core Parallel Regressions: `ols_reg_group_parallel`

For panel datasets with a large number of groups, Polars evaluates custom plugin expressions sequentially when using standard `df.group_by().agg()`. 

To achieve **linear multi-core parallel scaling (3x – 8x faster)** across all CPU cores, use `ols_reg_group_parallel`:

```python
from polars_ols_reg import ols_reg_group_parallel

# 1. Single group variable (uses all CPU cores by default)
res = ols_reg_group_parallel(df, by='var', formula='y ~ x1 + x2', cov_type='nw3')

# 2. Multiple group variables (e.g. firm and year)
res = ols_reg_group_parallel(df, by=['firm', 'year'], formula='y ~ x1 + x2', cov_type='white', result='coeff')

# 3. Explicitly set number of threads
res = ols_reg_group_parallel(df, by='var', formula='y ~ x1 + x2', n_threads=16)
```

### Parameters:
- **`df`** (`pl.DataFrame`): Input Polars DataFrame.
- **`by`** (`str | Sequence[str]`): Column name or list of column names to group by.
- **`formula`** (`str`): Regression formula (e.g. `'y ~ x1 + x2'`).
- **`cov_type`** (`str`, default: `'white'`): `'white'` or `'nw+N'` (e.g. `'nw3'`).
- **`result`** (`str`, default: `'all'`): `'all'`, `'coeff'`, `'tvalue'`, `'pvalue'`, or `'adjr2'`.
- **`n_threads`** (`int`, optional): Number of worker threads. Defaults to `pl.thread_pool_size()`.
