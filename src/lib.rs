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
// Regression calculation
// ------------------------------------------------------------

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

fn calculate_ols(
    inputs: &[Series],
    nw_lags: usize,
) -> PolarsResult<OlsResult> {
    if inputs.len() < 2 {
        polars_bail!(InvalidOperation:
            "OLS requires a dependent variable and at least one regressor"
        );
    }

    // Cast all input series to Float64
    let mut columns = Vec::with_capacity(inputs.len());
    for input in inputs {
        let casted = input.cast(&DataType::Float64)?;
        let ca = casted.f64()?.clone();
        columns.push(ca);
    }

    let n_rows = inputs[0].len();
    let n_cols = inputs.len(); // y + regressors -> total regressors with const = n_cols
    let k = n_cols;

    // Filter complete-case finite observations (drop nulls and non-finite)
    let mut observations: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        let mut valid = true;

        for col in &columns {
            match col.get(i) {
                Some(v) if v.is_finite() => row.push(v),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            observations.push(row);
        }
    }

    let n = observations.len();

    // Insufficient degrees of freedom for this group -> return NaNs safely
    if n <= k {
        return Ok(OlsResult::nan(k));
    }

    // Construct design matrix X (column 0 = 1.0 for intercept) and vector y
    let mut x = DMatrix::<f64>::zeros(n, k);
    let mut y = DVector::<f64>::zeros(n);

    for i in 0..n {
        x[(i, 0)] = 1.0;
        y[i] = observations[i][0];

        for j in 1..k {
            x[(i, j)] = observations[i][j];
        }
    }

    // OLS normal equations
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;

    // Bread matrix (X'X)^(-1)
    let bread = match xtx.try_inverse() {
        Some(inv) => inv,
        None => return Ok(OlsResult::nan(k)), // Singular design matrix (collinear regressors)
    };

    let beta = &bread * xty;
    let residuals = &y - &x * &beta;

    // R-squared and Adjusted R-squared
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
    let rss: f64 = residuals.iter().map(|v| v.powi(2)).sum();

    let adjr2 = if tss <= 0.0 || !tss.is_finite() {
        f64::NAN
    } else {
        let r2 = 1.0 - rss / tss;
        1.0 - (1.0 - r2) * (n - 1) as f64 / (n - k) as f64
    };

    // Precalculate scores S_t = e_t * x_t (for White and Newey-West)
    let mut scores: Vec<DVector<f64>> = Vec::with_capacity(n);
    for t in 0..n {
        let e = residuals[t];
        let mut st = DVector::<f64>::zeros(k);
        for j in 0..k {
            st[j] = x[(t, j)] * e;
        }
        scores.push(st);
    }

    // Meat matrix S: contemporaneous covariance S_0 (White HC0)
    let mut meat = DMatrix::<f64>::zeros(k, k);
    for t in 0..n {
        let st = &scores[t];
        meat += st * st.transpose();
    }

    // Newey-West Bartlett kernel lagged covariance
    if nw_lags > 0 {
        let max_lag = nw_lags.min(n.saturating_sub(1));
        for lag in 1..=max_lag {
            let weight = 1.0 - (lag as f64) / ((nw_lags + 1) as f64);
            let mut gamma = DMatrix::<f64>::zeros(k, k);
            for t in lag..n {
                let cur = &scores[t];
                let prev = &scores[t - lag];
                gamma += cur * prev.transpose();
            }
            meat += weight * (&gamma + gamma.transpose());
        }
    }

    // Sandwich covariance: (X'X)^(-1) * Meat * (X'X)^(-1)
    let covariance = &bread * meat * &bread;

    let df = (n - k) as f64;
    let student_t = match StudentsT::new(0.0, 1.0, df) {
        Ok(dist) => dist,
        Err(_) => return Ok(OlsResult::nan(k)),
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

            if t.is_nan() {
                pvalue[j] = f64::NAN;
            } else {
                let p = 2.0 * (1.0 - student_t.cdf(t.abs()));
                pvalue[j] = p.clamp(0.0, 1.0);
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
    let result = calculate_ols(inputs, kwargs.nw_lags)?;
    let values: Vec<f64> = result.beta.iter().copied().collect();
    result_to_struct(inputs, &kwargs.names, &values, "b_")
}

#[polars_expr(output_type_func_with_kwargs=tvalue_schema)]
fn ols_tvalue(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols(inputs, kwargs.nw_lags)?;
    let values: Vec<f64> = result.tvalue.iter().copied().collect();
    result_to_struct(inputs, &kwargs.names, &values, "t_")
}

#[polars_expr(output_type_func_with_kwargs=pvalue_schema)]
fn ols_pvalue(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols(inputs, kwargs.nw_lags)?;
    let values: Vec<f64> = result.pvalue.iter().copied().collect();
    result_to_struct(inputs, &kwargs.names, &values, "p_")
}

#[polars_expr(output_type=Float64)]
fn ols_adjr2(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols(inputs, kwargs.nw_lags)?;
    Ok(Series::new(
        "adjr2".into(),
        &[result.adjr2],
    ))
}

#[polars_expr(output_type_func_with_kwargs=all_schema)]
fn ols_all(inputs: &[Series], kwargs: OlsKwargs) -> PolarsResult<Series> {
    let result = calculate_ols(inputs, kwargs.nw_lags)?;

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
