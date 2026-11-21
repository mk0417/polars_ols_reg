use nalgebra::{DMatrix, DVector};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{ContinuousCDF, StudentsT};

// ------------------------------------------------------------
// Keyword arguments deserialization
// ------------------------------------------------------------

#[derive(Deserialize, Clone, Debug, Default)]
pub struct OlsKwargs {
    #[serde(default)]
    pub nw_lags: usize,
    #[serde(default)]
    pub names: Vec<String>,
}

// ------------------------------------------------------------
// Output schema functions for Struct output
// ------------------------------------------------------------

fn output_fields(input_fields: &[Field], kwargs: &OlsKwargs, prefix: &str) -> PolarsResult<Field> {
    let num_regressors = if !kwargs.names.is_empty() {
        kwargs.names.len()
    } else {
        input_fields.len().saturating_sub(1)
    };

    let mut fields = Vec::with_capacity(num_regressors + 1);

    // Intercept (constant) is always the first output field
    fields.push(Field::new(
        format!("{prefix}const").into(),
        DataType::Float64,
    ));

    // Regressor columns follow in order
    if !kwargs.names.is_empty() {
        for name in &kwargs.names {
            fields.push(Field::new(
                format!("{prefix}{name}").into(),
                DataType::Float64,
            ));
        }
    } else {
        for (i, field) in input_fields.iter().skip(1).enumerate() {
            let field_name = field.name();
            let col_name = if field_name.is_empty() {
                format!("x{}", i + 1)
            } else {
                field_name.to_string()
            };
            fields.push(Field::new(
                format!("{prefix}{col_name}").into(),
                DataType::Float64,
            ));
        }
    }

    Ok(Field::new(
        "ols".into(),
        DataType::Struct(fields),
    ))
}

fn coeff_schema(fields: &[Field], kwargs: OlsKwargs) -> PolarsResult<Field> {
    output_fields(fields, &kwargs, "b_")
}

fn tvalue_schema(fields: &[Field], kwargs: OlsKwargs) -> PolarsResult<Field> {
    output_fields(fields, &kwargs, "t_")
}

fn pvalue_schema(fields: &[Field], kwargs: OlsKwargs) -> PolarsResult<Field> {
    output_fields(fields, &kwargs, "p_")
}

fn all_schema(input_fields: &[Field], kwargs: OlsKwargs) -> PolarsResult<Field> {
    let names: Vec<String> = if !kwargs.names.is_empty() {
        kwargs.names
    } else {
        input_fields.iter().skip(1).enumerate().map(|(i, f)| {
            let n = f.name();
            if n.is_empty() {
                format!("x{}", i + 1)
            } else {
                n.to_string()
            }
        }).collect()
    };

    let total_fields = 3 * (names.len() + 1) + 1;
    let mut fields = Vec::with_capacity(total_fields);

    // 1. b_const, b_<name>
    fields.push(Field::new("b_const".into(), DataType::Float64));
    for name in &names {
        fields.push(Field::new(format!("b_{name}").into(), DataType::Float64));
    }

    // 2. t_const, t_<name>
    fields.push(Field::new("t_const".into(), DataType::Float64));
    for name in &names {
        fields.push(Field::new(format!("t_{name}").into(), DataType::Float64));
    }

    // 3. p_const, p_<name>
    fields.push(Field::new("p_const".into(), DataType::Float64));
    for name in &names {
        fields.push(Field::new(format!("p_{name}").into(), DataType::Float64));
    }

    // 4. adjr2
    fields.push(Field::new("adjr2".into(), DataType::Float64));

    Ok(Field::new("ols".into(), DataType::Struct(fields)))
}

// ------------------------------------------------------------
// Core Regression Engine
// ------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq)]
enum OlsTarget {
    Tvalue,
    Pvalue,
    All,
}

struct OlsResult {
    beta: DVector<f64>,
    tvalue: DVector<f64>,
    pvalue: DVector<f64>,
    adjr2: f64,
}

impl OlsResult {
    fn nan(k: usize) -> Self {
        Self {
            beta: DVector::from_element(k, f64::NAN),
            tvalue: DVector::from_element(k, f64::NAN),
            pvalue: DVector::from_element(k, f64::NAN),
            adjr2: f64::NAN,
        }
    }
}

/// Helper to cast inputs to Float64 ChunkedArray without re-casting if already Float64
#[inline]
fn extract_columns(inputs: &[Series]) -> PolarsResult<Vec<Float64Chunked>> {
    let mut columns = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.dtype() == &DataType::Float64 {
            columns.push(input.f64()?.clone());
        } else {
            let casted = input.cast(&DataType::Float64)?;
            columns.push(casted.f64()?.clone());
        }
    }
    Ok(columns)
}

struct ValidData {
    n: usize,
    k: usize,
    y: Vec<f64>,
    // x is stored row-major with k elements per row: [1.0, x_1, x_2, ..., x_{k-1}]
    x: Vec<f64>,
    xtx: DMatrix<f64>,
    xty: DVector<f64>,
    y_sum: f64,
}

/// Extract complete-case finite observations into flat contiguous memory.
/// X'X, X'y, and y_sum are accumulated simultaneously in a single streaming pass.
fn extract_valid_data(inputs: &[Series]) -> PolarsResult<Option<ValidData>> {
    if inputs.len() < 2 {
        polars_bail!(InvalidOperation:
            "OLS requires a dependent variable and at least one regressor"
        );
    }

    let columns = extract_columns(inputs)?;
    let n_rows = inputs[0].len();
    let k = inputs.len();

    if n_rows <= k {
        return Ok(None);
    }

    // Fast path: contiguous single-chunk columns without nulls
    let is_clean = columns.iter().all(|c| c.null_count() == 0 && c.chunks().len() == 1);

    if is_clean {
        let col_slices: Vec<&[f64]> = columns.iter().map(|c| c.cont_slice().unwrap()).collect();

        let mut all_finite = true;
        for s in &col_slices {
            for &v in *s {
                if !v.is_finite() {
                    all_finite = false;
                    break;
                }
            }
            if !all_finite {
                break;
            }
        }

        if all_finite {
            let n = n_rows;
            let mut y = Vec::with_capacity(n);
            let mut x = Vec::with_capacity(n * k);
            let mut xtx = DMatrix::<f64>::zeros(k, k);
            let mut xty = DVector::<f64>::zeros(k);

            xtx[(0, 0)] = n as f64;

            // Intercept interactions
            for j in 1..k {
                let sum_xj: f64 = col_slices[j].iter().sum();
                xtx[(0, j)] = sum_xj;
                xtx[(j, 0)] = sum_xj;
            }

            // Regressor interactions (exploiting symmetry)
            for i in 1..k {
                for j in 1..=i {
                    let dot: f64 = if i == j {
                        col_slices[i].iter().map(|&v| v * v).sum()
                    } else {
                        col_slices[i].iter().zip(col_slices[j].iter()).map(|(&a, &b)| a * b).sum()
                    };
                    xtx[(i, j)] = dot;
                    xtx[(j, i)] = dot;
                }
            }

            // y interactions
            let y_slice = col_slices[0];
            let sum_y: f64 = y_slice.iter().sum();
            xty[0] = sum_y;
            let y_sum = sum_y;

            for j in 1..k {
                let dot: f64 = col_slices[j].iter().zip(y_slice.iter()).map(|(&a, &b)| a * b).sum();
                xty[j] = dot;
            }

            // Populate contiguous flat buffers
            for t in 0..n {
                y.push(y_slice[t]);
                x.push(1.0);
                for j in 1..k {
                    x.push(col_slices[j][t]);
                }
            }

            return Ok(Some(ValidData {
                n,
                k,
                y,
                x,
                xtx,
                xty,
                y_sum,
            }));
        }
    }

    // General complete-case filtering path: drops nulls and non-finite values
    let mut y = Vec::with_capacity(n_rows);
    let mut x = Vec::with_capacity(n_rows * k);
    let mut xtx = DMatrix::<f64>::zeros(k, k);
    let mut xty = DVector::<f64>::zeros(k);
    let mut y_sum = 0.0;
    let mut n = 0;

    let mut row_buf = vec![0.0; k];

    for i in 0..n_rows {
        let mut valid = true;
        row_buf[0] = 1.0;

        let y_val = match columns[0].get(i) {
            Some(v) if v.is_finite() => v,
            _ => {
                valid = false;
                0.0
            }
        };

        if valid {
            for j in 1..k {
                match columns[j].get(i) {
                    Some(v) if v.is_finite() => row_buf[j] = v,
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }
        }

        if valid {
            n += 1;
            y.push(y_val);
            y_sum += y_val;

            for j in 0..k {
                x.push(row_buf[j]);
            }

            xty[0] += y_val;
            xtx[(0, 0)] += 1.0;

            for j in 1..k {
                let val_j = row_buf[j];
                xtx[(0, j)] += val_j;
                xtx[(j, 0)] += val_j;
                xty[j] += val_j * y_val;
            }

            for j in 1..k {
                let val_j = row_buf[j];
                for m in 1..=j {
                    let prod = val_j * row_buf[m];
                    xtx[(j, m)] += prod;
                    if j != m {
                        xtx[(m, j)] += prod;
                    }
                }
            }
        }
    }

    if n <= k {
        return Ok(None);
    }

    Ok(Some(ValidData {
        n,
        k,
        y,
        x,
        xtx,
        xty,
        y_sum,
    }))
}

// ------------------------------------------------------------
// Specialized High-Performance Solvers
// ------------------------------------------------------------

/// Fast solver for regression coefficients only: computes (X'X)^(-1) X'y with zero
/// residual/meat/distribution overhead and zero intermediate heap allocations.
fn calculate_coeff(inputs: &[Series]) -> PolarsResult<Vec<f64>> {
    if inputs.len() < 2 {
        polars_bail!(InvalidOperation:
            "OLS requires a dependent variable and at least one regressor"
        );
    }

    let columns = extract_columns(inputs)?;
    let n_rows = inputs[0].len();
    let k = inputs.len();

    if n_rows <= k {
        return Ok(vec![f64::NAN; k]);
    }

    let is_clean = columns.iter().all(|c| c.null_count() == 0 && c.chunks().len() == 1);

    if is_clean {
        let col_slices: Vec<&[f64]> = columns.iter().map(|c| c.cont_slice().unwrap()).collect();

        let mut all_finite = true;
        for s in &col_slices {
            for &v in *s {
                if !v.is_finite() {
                    all_finite = false;
                    break;
                }
            }
            if !all_finite {
                break;
            }
        }

        if all_finite {
            let n = n_rows;
            let mut xtx = DMatrix::<f64>::zeros(k, k);
            let mut xty = DVector::<f64>::zeros(k);

            xtx[(0, 0)] = n as f64;

            for j in 1..k {
                let sum_xj: f64 = col_slices[j].iter().sum();
                xtx[(0, j)] = sum_xj;
                xtx[(j, 0)] = sum_xj;
            }

            for i in 1..k {
                for j in 1..=i {
                    let dot: f64 = if i == j {
                        col_slices[i].iter().map(|&v| v * v).sum()
                    } else {
                        col_slices[i].iter().zip(col_slices[j].iter()).map(|(&a, &b)| a * b).sum()
                    };
                    xtx[(i, j)] = dot;
                    xtx[(j, i)] = dot;
                }
            }

            let y_slice = col_slices[0];
            xty[0] = y_slice.iter().sum();

            for j in 1..k {
                xty[j] = col_slices[j].iter().zip(y_slice.iter()).map(|(&a, &b)| a * b).sum();
            }

            let bread = match xtx.try_inverse() {
                Some(inv) => inv,
                None => return Ok(vec![f64::NAN; k]),
            };

            let beta = bread * xty;
            return Ok(beta.iter().copied().collect());
        }
    }

    // Streaming accumulator: zero heap allocations for data points
    let mut xtx = DMatrix::<f64>::zeros(k, k);
    let mut xty = DVector::<f64>::zeros(k);
    let mut n = 0;
    let mut row_buf = vec![0.0; k];

    for i in 0..n_rows {
        let mut valid = true;
        row_buf[0] = 1.0;

        let y_val = match columns[0].get(i) {
            Some(v) if v.is_finite() => v,
            _ => {
                valid = false;
                0.0
            }
        };

        if valid {
            for j in 1..k {
                match columns[j].get(i) {
                    Some(v) if v.is_finite() => row_buf[j] = v,
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }
        }

        if valid {
            n += 1;
            xty[0] += y_val;
            xtx[(0, 0)] += 1.0;

            for j in 1..k {
                let val_j = row_buf[j];
                xtx[(0, j)] += val_j;
                xtx[(j, 0)] += val_j;
                xty[j] += val_j * y_val;
            }

            for j in 1..k {
                let val_j = row_buf[j];
                for m in 1..=j {
                    let prod = val_j * row_buf[m];
                    xtx[(j, m)] += prod;
                    if j != m {
                        xtx[(m, j)] += prod;
                    }
                }
            }
        }
    }

    if n <= k {
        return Ok(vec![f64::NAN; k]);
    }

    let bread = match xtx.try_inverse() {
        Some(inv) => inv,
        None => return Ok(vec![f64::NAN; k]),
    };

    let beta = bread * xty;
    Ok(beta.iter().copied().collect())
}

/// Fast solver for Adjusted R-squared: skips sandwich covariance and p-value distribution
fn calculate_adjr2(inputs: &[Series]) -> PolarsResult<f64> {
    let data = match extract_valid_data(inputs)? {
        Some(d) => d,
        None => return Ok(f64::NAN),
    };
    let n = data.n;
    let k = data.k;

    let bread = match data.xtx.try_inverse() {
        Some(inv) => inv,
        None => return Ok(f64::NAN),
    };
    let beta = &bread * &data.xty;

    let y_mean = data.y_sum / n as f64;
    let mut rss = 0.0;
    let mut tss = 0.0;

    for t in 0..n {
        let y_t = data.y[t];
        let x_row = &data.x[t * k..(t + 1) * k];
        let mut y_hat = 0.0;
        for j in 0..k {
            y_hat += x_row[j] * beta[j];
        }
        let e = y_t - y_hat;
        rss += e * e;
        let dy = y_t - y_mean;
        tss += dy * dy;
    }

    if tss <= 0.0 || !tss.is_finite() {
        Ok(f64::NAN)
    } else {
        let r2 = 1.0 - rss / tss;
        let adjr2 = 1.0 - (1.0 - r2) * (n - 1) as f64 / (n - k) as f64;
        Ok(adjr2)
    }
}

/// Full robust regression solver for White (HC0) and Newey-West (HAC) standard errors.
/// Uses in-place outer products and symmetric updates with zero per-row heap allocations.
fn calculate_ols_robust(
    inputs: &[Series],
    nw_lags: usize,
    target: OlsTarget,
) -> PolarsResult<OlsResult> {
    let data = match extract_valid_data(inputs)? {
        Some(d) => d,
        None => return Ok(OlsResult::nan(inputs.len())),
    };
    let n = data.n;
    let k = data.k;

    let bread = match data.xtx.try_inverse() {
        Some(inv) => inv,
        None => return Ok(OlsResult::nan(k)),
    };
    let beta = &bread * &data.xty;

    // Compute residuals, RSS, TSS
    let y_mean = data.y_sum / n as f64;
    let mut residuals = Vec::with_capacity(n);
    let mut rss = 0.0;
    let mut tss = 0.0;

    for t in 0..n {
        let y_t = data.y[t];
        let x_row = &data.x[t * k..(t + 1) * k];
        let mut y_hat = 0.0;
        for j in 0..k {
            y_hat += x_row[j] * beta[j];
        }
        let e = y_t - y_hat;
        residuals.push(e);
        rss += e * e;
        let dy = y_t - y_mean;
        tss += dy * dy;
    }

    let adjr2 = if tss <= 0.0 || !tss.is_finite() {
        f64::NAN
    } else {
        let r2 = 1.0 - rss / tss;
        1.0 - (1.0 - r2) * (n - 1) as f64 / (n - k) as f64
    };

    let mut meat = DMatrix::<f64>::zeros(k, k);

    if nw_lags == 0 {
        // White HC0 contemporaneous meat matrix: accumulated in-place without storing scores
        for t in 0..n {
            let e = residuals[t];
            let x_row = &data.x[t * k..(t + 1) * k];
            for i in 0..k {
                let s_i = e * x_row[i];
                for j in 0..=i {
                    let prod = s_i * (e * x_row[j]);
                    meat[(i, j)] += prod;
                    if i != j {
                        meat[(j, i)] += prod;
                    }
                }
            }
        }
    } else {
        // Newey-West HAC: store scores in single flat contiguous buffer of length n * k
        let mut scores = Vec::with_capacity(n * k);
        for t in 0..n {
            let e = residuals[t];
            let x_row = &data.x[t * k..(t + 1) * k];
            for j in 0..k {
                scores.push(e * x_row[j]);
            }
        }

        // Contemporaneous S_0
        for t in 0..n {
            let cur = &scores[t * k..(t + 1) * k];
            for i in 0..k {
                let cur_i = cur[i];
                for j in 0..=i {
                    let prod = cur_i * cur[j];
                    meat[(i, j)] += prod;
                    if i != j {
                        meat[(j, i)] += prod;
                    }
                }
            }
        }

        // Bartlett kernel lagged covariance: accumulated in-place with zero temporary heap matrices
        let max_lag = nw_lags.min(n.saturating_sub(1));
        for lag in 1..=max_lag {
            let weight = 1.0 - (lag as f64) / ((nw_lags + 1) as f64);
            let mut gamma = DMatrix::<f64>::zeros(k, k);

            for t in lag..n {
                let cur = &scores[t * k..(t + 1) * k];
                let prev = &scores[(t - lag) * k..((t - lag) + 1) * k];

                for i in 0..k {
                    let cur_i = cur[i];
                    for j in 0..k {
                        gamma[(i, j)] += cur_i * prev[j];
                    }
                }
            }

            for i in 0..k {
                for j in 0..k {
                    meat[(i, j)] += weight * (gamma[(i, j)] + gamma[(j, i)]);
                }
            }
        }
    }

    // Sandwich covariance: (X'X)^(-1) * Meat * (X'X)^(-1)
    let covariance = &bread * meat * &bread;

    let df = (n - k) as f64;
    let student_t = if target == OlsTarget::Pvalue || target == OlsTarget::All {
        match StudentsT::new(0.0, 1.0, df) {
            Ok(dist) => Some(dist),
            Err(_) => return Ok(OlsResult::nan(k)),
        }
    } else {
        None
    };

    let mut tvalue = DVector::<f64>::zeros(k);
    let mut pvalue = DVector::<f64>::zeros(k);

    for j in 0..k {
        let variance = covariance[(j, j)];

        if variance <= 0.0 || !variance.is_finite() {
            tvalue[j] = f64::NAN;
            pvalue[j] = f64::NAN;
        } else {
            let se = variance.sqrt();
            let t = beta[j] / se;
            tvalue[j] = t;

            if let Some(ref dist) = student_t {
                if t.is_nan() {
                    pvalue[j] = f64::NAN;
                } else {
                    let p = 2.0 * (1.0 - dist.cdf(t.abs()));
                    pvalue[j] = p.clamp(0.0, 1.0);
                }
            } else {
                pvalue[j] = f64::NAN;
            }
        }
    }

    Ok(OlsResult {
        beta,
        tvalue,
        pvalue,
        adjr2,
    })
}

// ------------------------------------------------------------
// Struct output constructor
// ------------------------------------------------------------

fn result_to_struct(
    inputs: &[Series],
    names: &[String],
    values: &[f64],
    prefix: &str,
) -> PolarsResult<Series> {
    let mut fields = Vec::with_capacity(values.len());

    // Intercept (const)
    fields.push(Series::new(
        format!("{prefix}const").into(),
        &[values[0]],
    ));

    // Regressor coefficients / statistics
    for j in 1..values.len() {
        let col_name = if j - 1 < names.len() {
            names[j - 1].clone()
        } else {
            let in_name = inputs[j].name();
            if in_name.is_empty() {
                format!("x{j}")
            } else {
                in_name.to_string()
            }
        };

        fields.push(Series::new(
            format!("{prefix}{col_name}").into(),
            &[values[j]],
        ));
    }

    let struct_ca = StructChunked::from_series(
        "ols".into(),
        1,
        fields.iter(),
    )?;

    Ok(struct_ca.into_series())
}

// ------------------------------------------------------------
// Polars Plugin Entry Points
// ------------------------------------------------------------

#[polars_expr(output_type_func_with_kwargs=coeff_schema)]
fn ols_coeff(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let values = calculate_coeff(inputs)?;
    result_to_struct(inputs, &kwargs.names, &values, "b_")
}

#[polars_expr(output_type_func_with_kwargs=tvalue_schema)]
fn ols_tvalue(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols_robust(inputs, kwargs.nw_lags, OlsTarget::Tvalue)?;
    let values: Vec<f64> = result.tvalue.iter().copied().collect();
    result_to_struct(inputs, &kwargs.names, &values, "t_")
}

#[polars_expr(output_type_func_with_kwargs=pvalue_schema)]
fn ols_pvalue(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols_robust(inputs, kwargs.nw_lags, OlsTarget::Pvalue)?;
    let values: Vec<f64> = result.pvalue.iter().copied().collect();
    result_to_struct(inputs, &kwargs.names, &values, "p_")
}

#[polars_expr(output_type=Float64)]
fn ols_adjr2(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let _ = kwargs;
    let adjr2 = calculate_adjr2(inputs)?;
    Ok(Series::new(
        "adjr2".into(),
        &[adjr2],
    ))
}

#[polars_expr(output_type_func_with_kwargs=all_schema)]
fn ols_all(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols_robust(inputs, kwargs.nw_lags, OlsTarget::All)?;

    let k = result.beta.len();
    let names: Vec<String> = if !kwargs.names.is_empty() {
        kwargs.names
    } else {
        (1..k).map(|j| {
            let in_name = inputs[j].name();
            if in_name.is_empty() {
                format!("x{j}")
            } else {
                in_name.to_string()
            }
        }).collect()
    };

    let total_fields = 3 * k + 1;
    let mut fields = Vec::with_capacity(total_fields);

    // 1. Coefficients: b_const, b_<name>
    fields.push(Series::new("b_const".into(), &[result.beta[0]]));
    for j in 1..k {
        let col_name = if j - 1 < names.len() { &names[j - 1] } else { "x" };
        fields.push(Series::new(format!("b_{col_name}").into(), &[result.beta[j]]));
    }

    // 2. t-values: t_const, t_<name>
    fields.push(Series::new("t_const".into(), &[result.tvalue[0]]));
    for j in 1..k {
        let col_name = if j - 1 < names.len() { &names[j - 1] } else { "x" };
        fields.push(Series::new(format!("t_{col_name}").into(), &[result.tvalue[j]]));
    }

    // 3. p-values: p_const, p_<name>
    fields.push(Series::new("p_const".into(), &[result.pvalue[0]]));
    for j in 1..k {
        let col_name = if j - 1 < names.len() { &names[j - 1] } else { "x" };
        fields.push(Series::new(format!("p_{col_name}").into(), &[result.pvalue[j]]));
    }

    // 4. Adjusted R-squared: adjr2
    fields.push(Series::new("adjr2".into(), &[result.adjr2]));

    let struct_ca = StructChunked::from_series(
        "ols".into(),
        1,
        fields.iter(),
    )?;

    Ok(struct_ca.into_series())
}

// ------------------------------------------------------------
// Python Module Entry Point (Required by PyO3 and Maturin)
// ------------------------------------------------------------

#[pymodule]
fn _polars_ols_reg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
