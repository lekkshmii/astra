use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;
use nalgebra::DVector;
use std::collections::HashMap;

#[pymodule]
fn astra_monte_carlo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(monte_carlo_flash_crash, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_regime_switching, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_correlation_breakdown, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_volatility_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_portfolio_stress, m)?)?;
    Ok(())
}

#[pyfunction]
fn monte_carlo_flash_crash(
    initial_prices: Vec<f64>,
    num_simulations: usize,
    time_steps: usize,
    crash_probability: f64,
    crash_magnitude: f64,
    recovery_rate: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut prices = initial_prices.clone();
            let mut path = vec![prices[0]];
            let mut in_crash = false;
            let mut crash_remaining = 0;
            
            for _ in 1..time_steps {
                if !in_crash && rng.gen::<f64>() < crash_probability {
                    in_crash = true;
                    crash_remaining = 10;
                }
                
                let return_rate = if in_crash {
                    crash_remaining -= 1;
                    if crash_remaining <= 0 {
                        in_crash = false;
                    }
                    -crash_magnitude + rng.sample(StandardNormal) * 0.1
                } else {
                    rng.sample(StandardNormal) * 0.02
                };
                
                prices[0] *= (1.0 + return_rate);
                path.push(prices[0]);
            }
            
            path
        })
        .collect();
    
    Ok(results)
}

#[pyfunction]
fn monte_carlo_regime_switching(
    initial_prices: Vec<f64>,
    num_simulations: usize,
    time_steps: usize,
    bull_volatility: f64,
    bear_volatility: f64,
    transition_probability: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut prices = initial_prices.clone();
            let mut path = vec![prices[0]];
            let mut is_bull_market = true;
            
            for _ in 1..time_steps {
                if rng.gen::<f64>() < transition_probability {
                    is_bull_market = !is_bull_market;
                }
                
                let volatility = if is_bull_market { bull_volatility } else { bear_volatility };
                let drift = if is_bull_market { 0.0008 } else { -0.0002 };
                
                let return_rate = drift + rng.sample(StandardNormal) * volatility;
                prices[0] *= (1.0 + return_rate);
                path.push(prices[0]);
            }
            
            path
        })
        .collect();
    
    Ok(results)
}

#[pyfunction]
fn monte_carlo_correlation_breakdown(
    initial_prices: Vec<f64>,
    num_simulations: usize,
    time_steps: usize,
    normal_correlation: f64,
    stress_correlation: f64,
    stress_threshold: f64,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let num_assets = initial_prices.len();
    
    let results: Vec<Vec<Vec<f64>>> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut prices = initial_prices.clone();
            let mut paths: Vec<Vec<f64>> = (0..num_assets).map(|i| vec![prices[i]]).collect();
            
            for _ in 1..time_steps {
                let market_shock = rng.sample(StandardNormal).abs();
                let correlation = if market_shock > stress_threshold {
                    stress_correlation
                } else {
                    normal_correlation
                };
                
                let common_factor = rng.sample(StandardNormal) * correlation.sqrt();
                
                for i in 0..num_assets {
                    let idiosyncratic = rng.sample(StandardNormal) * (1.0 - correlation).sqrt();
                    let return_rate = (common_factor + idiosyncratic) * 0.02;
                    prices[i] *= (1.0 + return_rate);
                    paths[i].push(prices[i]);
                }
            }
            
            paths
        })
        .collect();
    
    Ok(results)
}

#[pyfunction]
fn monte_carlo_volatility_clustering(
    initial_price: f64,
    num_simulations: usize,
    time_steps: usize,
    alpha: f64,
    beta: f64,
    omega: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut price = initial_price;
            let mut path = vec![price];
            let mut variance = 0.0004;
            let mut last_return = 0.0;
            
            for _ in 1..time_steps {
                variance = omega + alpha * last_return.powi(2) + beta * variance;
                let volatility = variance.sqrt();
                
                let return_rate = rng.sample(StandardNormal) * volatility;
                price *= (1.0 + return_rate);
                path.push(price);
                last_return = return_rate;
            }
            
            path
        })
        .collect();
    
    Ok(results)
}

#[pyfunction]
fn monte_carlo_portfolio_stress(
    weights: Vec<f64>,
    expected_returns: Vec<f64>,
    covariance_matrix: Vec<Vec<f64>>,
    num_simulations: usize,
    time_horizon: usize,
    confidence_levels: Vec<f64>,
) -> PyResult<HashMap<String, f64>> {
    let num_assets = weights.len();
    
    let portfolio_returns: Vec<f64> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut portfolio_value = 1.0;
            
            for _ in 0..time_horizon {
                let mut asset_returns = vec![0.0; num_assets];
                
                for i in 0..num_assets {
                    let mut return_val = expected_returns[i];
                    for j in 0..num_assets {
                        return_val += rng.sample(StandardNormal) * covariance_matrix[i][j].sqrt();
                    }
                    asset_returns[i] = return_val;
                }
                
                let portfolio_return: f64 = weights.iter().zip(asset_returns.iter())
                    .map(|(w, r)| w * r)
                    .sum();
                
                portfolio_value *= (1.0 + portfolio_return);
            }
            
            (portfolio_value - 1.0) * 100.0
        })
        .collect();
    
    let mut sorted_returns = portfolio_returns.clone();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut results = HashMap::new();
    
    for &confidence_level in &confidence_levels {
        let index = ((1.0 - confidence_level) * num_simulations as f64) as usize;
        let var = sorted_returns[index.min(sorted_returns.len() - 1)];
        results.insert(format!("VaR_{}", (confidence_level * 100.0) as u8), var);
    }
    
    let expected_return = portfolio_returns.iter().sum::<f64>() / num_simulations as f64;
    results.insert("expected_return".to_string(), expected_return);
    
    let variance = portfolio_returns.iter()
        .map(|r| (r - expected_return).powi(2))
        .sum::<f64>() / (num_simulations - 1) as f64;
    results.insert("volatility".to_string(), variance.sqrt());
    
    Ok(results)
}
