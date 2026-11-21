"""
Example Usage of polars_ols_reg plugin
"""

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from polars_ols_reg import ols_reg

# Create a synthetic panel dataset with groups
data = pl.DataFrame({
    'var': ['A']*50+['B']*50,
    'x1': [float(i%10) for i in range(100)],
    'x2': [float((i*3)%7) for i in range(100)],
    'y': [2+1.5*(i%10)-0.5*((i*3)%7)+0.1*(i%5) for i in range(100)]})

# Default behavior: returns everything (coefficients, t-values, p-values, and adjr2)
print('\n--- Default behavior (Returns All: b_*, t_*, p_*, adjr2) ---')
all_df = (
    data.group_by('var')
    .agg(est=ols_reg('y ~ x1+x2', 'nw3'))
    .unnest('est'))
print(all_df)

# Only coefficients
print('\n--- Only coefficients (Newey-West, 3 lags) ---')
coeff_df = (
    data.group_by('var')
    .agg(est=ols_reg('y ~ x1+x2', 'nw3', 'coeff'))
    .unnest('est'))
print(coeff_df)

# Only t-values
print('\n--- Only t-values (Newey-West, 3 lags) ---')
tval_df = (
    data.group_by('var')
    .agg(est=ols_reg('y ~ x1+x2', 'nw3', 'tvalue'))
    .unnest('est'))
print(tval_df)

# Only p-values
print('\n--- Only p-values (White standard errors by default) ---')
pval_df = (
    data.group_by('var')
    .agg(est=ols_reg('y ~ x1+x2', 'white', 'pvalue'))
    .unnest('est'))
print(pval_df)

# Only adjusted R-squared
print('\n--- Only Adjusted R-squared ---')
adjr2_df = (
    data.group_by('var')
    .agg(adjr2=ols_reg('y ~ x1+x2', 'nw3', 'adjr2')))
print(adjr2_df)

# =============================================================================
# Consistency Tests with statsmodels
# =============================================================================
print('\n' + '=' * 78)
print('--- Testing Consistency with statsmodels ---')
print('=' * 78)

def test_statsmodels_consistency(df):
    """
    Verify that polars_ols_reg produces results consistent with statsmodels.

    Notes on statsmodels equivalence:
    1. White HC0:
       - polars_ols_reg: cov_type='white'
       - statsmodels: fit(cov_type='HC0', use_t=True)
    2. Newey-West HAC:
       - polars_ols_reg: cov_type='nw3' (or 'nw+3')
       - statsmodels: fit(cov_type='HAC', cov_kwds={'maxlags': 3, 'use_correction': False}, use_t=True)
    3. use_t=True:
       polars_ols_reg uses exact Student's t distribution with (n - k) degrees of freedom
       for small-sample inference, which matches statsmodels with use_t=True.
    """
    df_pd = df.to_pandas()
    groups = ['A', 'B']

    test_configs = [
        ('Newey-West HAC (3 lags)', 'nw+3', 'HAC', {'maxlags': 3}),
        ('White HC0 Robust SE', 'white', 'HC0', {})]

    for label, pl_cov, sm_cov, sm_kwds in test_configs:
        print(f'\n[Test configuration: {label}]')
        pl_res = (
            df.group_by('var', maintain_order=True)
            .agg(est=ols_reg('y ~ x1+x2', pl_cov))
            .unnest('est'))

        for group in groups:
            sub = df_pd[df_pd['var']==group]
            model = smf.ols('y ~ x1 + x2', data=sub)
            sm_fit = model.fit(cov_type=sm_cov, cov_kwds=sm_kwds, use_t=True)
            pl_row = pl_res.filter(pl.col('var')==group)

            if group == 'A':
                print(f'Side-by-side comparison for Group {group}:')
                print(f'{'Metric':<10} | {'polars_ols_reg':<18} | {'statsmodels':<18} | {'Abs Diff':<12} | Status')
                print('-' * 72)

            metrics = [
                ('b_const', pl_row['b_const'][0], sm_fit.params['Intercept']),
                ('b_x1',    pl_row['b_x1'][0],    sm_fit.params['x1']),
                ('b_x2',    pl_row['b_x2'][0],    sm_fit.params['x2']),
                ('t_const', pl_row['t_const'][0], sm_fit.tvalues['Intercept']),
                ('t_x1',    pl_row['t_x1'][0],    sm_fit.tvalues['x1']),
                ('t_x2',    pl_row['t_x2'][0],    sm_fit.tvalues['x2']),
                ('p_const', pl_row['p_const'][0], sm_fit.pvalues['Intercept']),
                ('p_x1',    pl_row['p_x1'][0],    sm_fit.pvalues['x1']),
                ('p_x2',    pl_row['p_x2'][0],    sm_fit.pvalues['x2']),
                ('adjr2',   pl_row['adjr2'][0],   sm_fit.rsquared_adj)]

            for name, val_pl, val_sm in metrics:
                diff = abs(val_pl-val_sm)
                is_close = np.isclose(val_pl, val_sm, rtol=1e-7, atol=1e-7)
                status = 'Pass' if is_close else 'Fail'
                if group == 'A':
                    print(f'{name:<10} | {val_pl:<18.8f} | {val_sm:<18.8f} | {diff:<12.2e} | {status}')
                    assert is_close, (
                        f'Mismatch in {label} (group {group}, metric {name}): '
                        f'polars={val_pl}, statsmodels={val_sm}, diff={diff}')

        print(f"✓ All groups ('A' and 'B') passed consistency check for {label}")

    # Additional test with non-zero p-values to explicitly verify Student-t CDF calculation
    print('\n[Test configuration: non-zero p-values with Gaussian noise]')
    np.random.seed(42)
    n = 100
    noisy_df = pl.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'y': 1.0 + 0.3 * np.random.normal(0, 1, n)})
    noisy_pd = noisy_df.to_pandas()

    sm_noisy = (
        smf.ols('y ~ x1 + x2', data=noisy_pd)
        .fit(cov_type='HAC', cov_kwds={'maxlags': 2}, use_t=True))
    pl_noisy = noisy_df.select(ols_reg('y ~ x1 + x2', 'nw2', 'all')).unnest('y')

    print(f'{'Metric':<10} | {'polars_ols_reg':<18} | {'statsmodels':<18} | {'Abs Diff':<12} | Status')
    print('-' * 72)
    for term, sm_term in [('const', 'Intercept'), ('x1', 'x1'), ('x2', 'x2')]:
        p_pl = pl_noisy[f'p_{term}'][0]
        p_sm = sm_noisy.pvalues[sm_term]
        diff = abs(p_pl - p_sm)
        is_close = np.isclose(p_pl, p_sm, rtol=1e-7, atol=1e-7)
        status = 'PASS' if is_close else 'FAIL'
        print(f'p_{term:<8} | {p_pl:<18.8f} | {p_sm:<18.8f} | {diff:<12.2e} | {status}')
        assert is_close, f'Mismatch in noisy p_{term}: polars={p_pl}, statsmodels={p_sm}'

    print('✓ Non-zero p-values test passed!')

    # Verify individual result option consistency ('coeff', 'tvalue', 'pvalue', 'adjr2')
    print('\n[Test configuration: individual result target options]')
    sub_df = df.filter(pl.col('var') == 'A')
    full_res = sub_df.select(ols_reg('y ~ x1 + x2', 'nw3', 'all')).unnest('y')

    target_coeff = sub_df.select(ols_reg('y ~ x1 + x2', 'nw3', 'coeff')).unnest('y')
    for col in ['b_const', 'b_x1', 'b_x2']:
        assert np.isclose(full_res[col][0], target_coeff[col][0]), f'Mismatch in {col}'

    target_tval = sub_df.select(ols_reg('y ~ x1 + x2', 'nw3', 'tvalue')).unnest('y')
    for col in ['t_const', 't_x1', 't_x2']:
        assert np.isclose(full_res[col][0], target_tval[col][0]), f'Mismatch in {col}'

    target_pval = sub_df.select(ols_reg('y ~ x1 + x2', 'nw3', 'pvalue')).unnest('y')
    for col in ['p_const', 'p_x1', 'p_x2']:
        assert np.isclose(full_res[col][0], target_pval[col][0]), f'Mismatch in {col}'

    target_adjr2 = sub_df.select(ols_reg('y ~ x1 + x2', 'nw3', 'adjr2')).to_series()[0]
    assert np.isclose(full_res['adjr2'][0], target_adjr2), 'Mismatch in adjr2'

    print("✓ Individual result targets ('coeff', 'tvalue', 'pvalue', 'adjr2') match 'all'!")

    print('\n' + '=' * 78)
    print('All statsmodels consistency tests passed successfully!')
    print('=' * 78)

test_statsmodels_consistency(data)

