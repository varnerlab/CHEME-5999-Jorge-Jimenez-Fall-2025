# %% [markdown]
# # Credit Risk Models – Monte Carlo (Glasserman v1)

# %%
#!/usr/bin/env python
# coding: utf-8

"""
Credit Risk Models - Monte Carlo Methods
Based on Glasserman's "Monte Carlo Methods in Financial Engineering"
Section 9.4: Credit Risk Models

INSTRUCTIONS:
1. Save this file as: credit_risk_models.py
2. Install packages: pip install numpy pandas matplotlib scipy seaborn
3. Run: python credit_risk_models.py

This implements:
- Section 9.4.1: Default Times and Valuation
- Section 9.4.2: Dependent Defaults (Copula Models)  
- Section 9.4.3: Portfolio Credit Risk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("CREDIT RISK MODELS - MONTE CARLO SIMULATION")
print("=" * 80)

# %% [markdown]
# ## PART 1: REAL CORPORATE DATA
# %%
# PART 1: REAL CORPORATE DATA
print("\nPART 1: REAL CORPORATE DATA")
print("-" * 80)

companies = {
    'AAPL': {'name': 'Apple Inc.', 'rating': 'AA+', 'sector': 'Technology', 'notional': 10e6},
    'MSFT': {'name': 'Microsoft Corp.', 'rating': 'AAA', 'sector': 'Technology', 'notional': 10e6},
    'JPM': {'name': 'JPMorgan Chase', 'rating': 'A+', 'sector': 'Financials', 'notional': 8e6},
    'BAC': {'name': 'Bank of America', 'rating': 'A-', 'sector': 'Financials', 'notional': 8e6},
    'XOM': {'name': 'Exxon Mobil', 'rating': 'AA', 'sector': 'Energy', 'notional': 7e6},
    'CVX': {'name': 'Chevron Corp.', 'rating': 'AA-', 'sector': 'Energy', 'notional': 7e6},
    'JNJ': {'name': 'Johnson & Johnson', 'rating': 'AAA', 'sector': 'Healthcare', 'notional': 9e6},
    'PFE': {'name': 'Pfizer Inc.', 'rating': 'A+', 'sector': 'Healthcare', 'notional': 6e6},
    'T': {'name': 'AT&T Inc.', 'rating': 'BBB', 'sector': 'Telecom', 'notional': 5e6},
    'VZ': {'name': 'Verizon', 'rating': 'BBB+', 'sector': 'Telecom', 'notional': 5e6}
}

default_rates = {
    'AAA': 0.0001, 'AA+': 0.0002, 'AA': 0.0003, 'AA-': 0.0005,
    'A+': 0.0008, 'A': 0.0015, 'A-': 0.0025,
    'BBB+': 0.0050, 'BBB': 0.0100, 'BBB-': 0.0200,
    'BB+': 0.0400, 'BB': 0.0800, 'BB-': 0.1200
}

portfolio_df = pd.DataFrame([
    {'Ticker': k, 'Company': v['name'], 'Rating': v['rating'], 
     'Sector': v['sector'], 'Notional': v['notional'],
     'Default_Rate': default_rates[v['rating']], 'Recovery_Rate': 0.50}
    for k, v in companies.items()
])

print(portfolio_df.to_string(index=False))
print(f"\nTotal Portfolio: ${portfolio_df['Notional'].sum()/1e6:.1f}M\n")

# %% [markdown]
# ## PART 2: DEFAULT TIME MODELS (Section 9.4.1)
# %%
# PART 2: DEFAULT TIME MODELS (Section 9.4.1)
# PART 2: DEFAULT TIME MODELS (Section 9.4.1)
# This section implements two stochastic models for simulating default times
print("PART 2: DEFAULT TIME MODELS (Section 9.4.1)")
print("-" * 80)

class DefaultTimeModel:
    """
    Models the time at which a credit event (default) occurs.
    Implements both constant hazard rate and time-varying intensity models.
    """
    def __init__(self, hazard_rate, T=5.0):
        """
        Initialize the default time model.
        
        Args:
            hazard_rate: Constant default intensity (probability per unit time)
            T: Time horizon for simulation (default 5 years)
        """
        self.hazard_rate = hazard_rate
        self.T = T
    
    def simulate_exponential(self, n_sims=10000):
        """
        Simulate default times using exponential distribution (constant hazard rate).
        
        This is the simplest model: default time τ ~ Exponential(λ)
        where λ is the hazard rate. The probability of default by time t is 1 - e^(-λt).
        
        Args:
            n_sims: Number of Monte Carlo simulations
            
        Returns:
            Array of simulated default times
        """
        return np.random.exponential(1/self.hazard_rate, n_sims)
    
    def simulate_cir(self, n_sims=10000, kappa=0.5, theta=None, sigma=0.2, n_steps=500):
        """
        Simulate default times using Cox-Ingersoll-Ross (CIR) stochastic intensity model.
        
        The hazard rate λ(t) follows: dλ = κ(θ - λ)dt + σ√λ dW
        where:
            κ (kappa): Mean reversion speed
            θ (theta): Long-run mean of hazard rate
            σ (sigma): Volatility of hazard rate
        
        This allows for time-varying default probability, more realistic than constant hazard.
        
        Args:
            n_sims: Number of Monte Carlo simulations
            kappa: Speed of mean reversion
            theta: Long-run mean (defaults to initial hazard_rate)
            sigma: Volatility parameter
            n_steps: Number of time steps for discretization
            
        Returns:
            Array of simulated default times
        """
        if theta is None:
            theta = self.hazard_rate
        
        # Time step size for discretization
        dt = self.T / n_steps
        
        # Initialize hazard rate paths - all paths start at initial hazard_rate
        lambda_t = np.ones((n_sims, n_steps + 1)) * self.hazard_rate
        
        # Simulate hazard rate paths using Euler-Maruyama discretization
        for i in range(n_steps):
            # Generate Brownian motion increments
            dW = np.random.normal(0, np.sqrt(dt), n_sims)
            
            # Update hazard rate using CIR dynamics
            # Mean reversion term: κ(θ - λ)dt pulls towards long-run mean
            # Diffusion term: σ√λ dW adds stochastic volatility
            lambda_t[:, i+1] = (lambda_t[:, i] + 
                               kappa * (theta - lambda_t[:, i]) * dt +
                               sigma * np.sqrt(np.maximum(lambda_t[:, i], 0)) * dW)
            
            # Ensure hazard rate stays non-negative (CIR boundary condition)
            lambda_t[:, i+1] = np.maximum(lambda_t[:, i+1], 0)
        
        # Calculate integrated intensity: ∫₀ᵀ λ(t)dt using trapezoidal approximation
        # This represents the cumulative hazard over the time horizon
        integrated_intensity = np.sum(lambda_t[:, :-1], axis=1) * dt
        
        # Generate default times using inverse transform method
        # For intensity models: τ = inf{t: ∫₀ᵗ λ(s)ds ≥ E} where E ~ Exp(1)
        # Equivalently: τ = -log(U) / (∫λdt / T) × T, where U ~ Uniform(0,1)
        U = np.random.uniform(0, 1, n_sims)
        return -np.log(U) / integrated_intensity * self.T

# Create model for Apple Inc. with its AA+ rating hazard rate
apple_model = DefaultTimeModel(hazard_rate=default_rates['AA+'], T=5.0)

# Generate 10,000 default times using exponential (constant hazard) model
default_times_exp = apple_model.simulate_exponential(10000)

# Generate 10,000 default times using CIR (stochastic intensity) model
default_times_cir = apple_model.simulate_cir(10000)

print(f"Apple Inc. (AA+):")
print(f"  Exponential: 5yr default prob = {np.mean(default_times_exp <= 5):.4f}")
print(f"  CIR Model:   5yr default prob = {np.mean(default_times_cir <= 5):.4f}\n")

# %% [markdown]
# ## PART 3: CREDIT VALUATION
# %%
# PART 3: CREDIT VALUATION
print("PART 3: DEFAULTABLE BOND PRICING")
print("-" * 80)

def price_defaultable_bond(notional, coupon_rate, maturity, rf_rate, 
                          hazard_rate, recovery_rate, n_sims=10000):
    default_times = np.random.exponential(1/hazard_rate, n_sims)
    payment_dates = np.arange(0.5, maturity + 0.5, 0.5)
    coupon_payment = notional * coupon_rate / 2
    bond_values = np.zeros(n_sims)
    
    for sim in range(n_sims):
        tau = default_times[sim]
        pv = 0
        if tau > maturity:
            for t in payment_dates:
                pv += coupon_payment * np.exp(-rf_rate * t)
            pv += notional * np.exp(-rf_rate * maturity)
        else:
            for t in payment_dates:
                if t < tau:
                    pv += coupon_payment * np.exp(-rf_rate * t)
            pv += notional * recovery_rate * np.exp(-rf_rate * tau)
        bond_values[sim] = pv
    
    return np.mean(bond_values), np.std(bond_values)

bond_price, bond_std = price_defaultable_bond(1000, 0.04, 5, 0.03, 
                                               default_rates['AA+'], 0.50, 10000)
risk_free_price = sum([20 * np.exp(-0.03 * t) for t in np.arange(0.5, 5.5, 0.5)]) + 1000 * np.exp(-0.15)
credit_spread = -np.log(bond_price / risk_free_price) / 5

print(f"Apple 5Y 4% Bond:")
print(f"  Price: ${bond_price:.2f}")
print(f"  Credit Spread: {credit_spread * 10000:.2f} bps\n")

# %% [markdown]
# ## PART 4: DEPENDENT DEFAULTS (Section 9.4.2)
# %%
# PART 4: DEPENDENT DEFAULTS (Section 9.4.2)
print("PART 4: DEPENDENT DEFAULTS - COPULA MODEL (Section 9.4.2)")
print("-" * 80)

class GaussianCopulaModel:
    """
    Implements Gaussian Copula model for simulating correlated defaults.
    This is a key technique from Section 9.4.2 of Glasserman for modeling
    dependent defaults using a factor structure.
    """
    def __init__(self, default_probs, correlation_matrix):
        self.default_probs = np.array(default_probs)
        self.n_obligors = len(default_probs)
        self.correlation_matrix = correlation_matrix
    
    def simulate_factor_model(self, T, n_sims=10000):
        """
        Simulate correlated default times using single-factor Gaussian copula.
        
        The model structure is: X_i = √β·F + √(1-β)·ε_i
        where F is a common market factor and ε_i are idiosyncratic shocks.
        This creates correlation in defaults through the shared factor F.
        """
        # Beta controls the strength of correlation through the common factor
        # Higher beta = more systematic risk, lower beta = more idiosyncratic risk
        beta = 0.3
        
        # F: Common market factor affecting all obligors (systematic risk)
        F = np.random.normal(0, 1, (n_sims, 1))
        
        # epsilon: Idiosyncratic shocks specific to each obligor
        epsilon = np.random.normal(0, 1, (n_sims, self.n_obligors))
        
        # X: Latent variable combining systematic and idiosyncratic components
        # This creates correlation structure: Corr(X_i, X_j) = β
        X = np.sqrt(beta) * F + np.sqrt(1 - beta) * epsilon
        
        # U: Transform to uniform variables via standard normal CDF
        # This is the copula transformation that preserves correlation
        U = norm.cdf(X)
        
        default_times = np.zeros((n_sims, self.n_obligors))
        
        # Convert uniform variables to default times using inverse transform
        # For each obligor, use their individual hazard rate
        for i in range(self.n_obligors):
            # Calculate constant hazard rate from default probability
            hazard_rate = -np.log(1 - self.default_probs[i]) / T
            
            # Generate default time: τ = -log(1-U) / λ
            # This preserves the marginal default probability while creating dependence
            default_times[:, i] = -np.log(1 - U[:, i]) / hazard_rate
        
        return default_times

# Build correlation matrix with sector-based clustering
# Companies in the same sector have higher correlation (30%)
# Companies in different sectors have base correlation (15%)
n_companies = len(portfolio_df)
correlation_matrix = np.ones((n_companies, n_companies)) * 0.15
sectors = portfolio_df['Sector'].values

for i in range(n_companies):
    for j in range(i, n_companies):
        # Same-sector pairs get higher correlation
        if sectors[i] == sectors[j]:
            correlation_matrix[i, j] = 0.3
            correlation_matrix[j, i] = 0.3
        # Diagonal elements must be 1 (perfect self-correlation)
        if i == j:
            correlation_matrix[i, j] = 1.0

# Calculate 5-year cumulative default probabilities from annual hazard rates
# Formula: P(τ ≤ T) = 1 - exp(-λT)
T = 5.0
default_probs = [1 - np.exp(-default_rates[r] * T) for r in portfolio_df['Rating']]

# Initialize copula model and simulate 10,000 correlated default scenarios
copula_model = GaussianCopulaModel(default_probs, correlation_matrix)
default_times = copula_model.simulate_factor_model(T, 10000)

# Convert default times to binary default indicators (1 = default, 0 = no default)
defaults = (default_times <= T).astype(int)

print(f"Factor Model (10,000 sims):")
print(f"  Mean defaults: {np.mean(np.sum(defaults, axis=1)):.2f}")
print(f"  P(no defaults): {np.mean(np.sum(defaults, axis=1) == 0):.4f}")
print(f"  P(2+ defaults): {np.mean(np.sum(defaults, axis=1) >= 2):.4f}\n")

print("FINANCIAL INTERPRETATION:")
print("-" * 80)
print("Why Copulas and Factor Models Matter:")
print("  • COPULAS allow us to model dependent defaults while preserving individual")
print("    default probabilities. Without copulas, we'd assume independence, which")
print("    severely underestimates the risk of multiple simultaneous defaults.")
print()
print("  • FACTOR MODEL captures systematic risk through a common market factor (F).")
print("    When the market factor is negative (economic downturn), multiple companies")
print("    are more likely to default together - this is 'default clustering'.")
print()
print("  • DEFAULT CORRELATION creates 'tail risk': The probability of 2+ defaults")
print("    is much higher than if defaults were independent. This is critical for")
print("    portfolio credit risk, CDO pricing, and regulatory capital calculations.")
print()
print(f"Key Outputs Explained:")
print(f"  • Mean defaults ({np.mean(np.sum(defaults, axis=1)):.2f}): On average, we expect less than 1")
print(f"    company to default over 5 years - reflecting the high credit quality (AA/A ratings).")
print()
print(f"  • P(no defaults) ({np.mean(np.sum(defaults, axis=1) == 0):.4f}): {np.mean(np.sum(defaults, axis=1) == 0)*100:.1f}% chance of zero defaults.")
print(f"    This gives confidence but we must prepare for the {(1-np.mean(np.sum(defaults, axis=1) == 0))*100:.1f}% tail scenarios.")
print()
print(f"  • P(2+ defaults) ({np.mean(np.sum(defaults, axis=1) >= 2):.4f}): {np.mean(np.sum(defaults, axis=1) >= 2)*100:.2f}% chance of multiple defaults.")
print(f"    Though rare, these clustered default events drive extreme portfolio losses")
print(f"    and are the primary concern for risk managers and regulators.")
print()
print("Business Impact:")
print("  This analysis helps banks determine how much capital to hold, how to price")
print("  credit derivatives, and whether portfolio diversification is truly effective.")
print("  The correlation structure reveals that sector concentration (e.g., multiple")
print("  financial firms) increases risk more than the number of names suggests.\n")


# %% [markdown]
# ## PART 5: PORTFOLIO CREDIT RISK (Section 9.4.3)
# %%
# PART 5: PORTFOLIO CREDIT RISK (Section 9.4.3)
print("PART 5: PORTFOLIO CREDIT RISK (Section 9.4.3)")
print("-" * 80)

class PortfolioCreditRisk:
    def __init__(self, portfolio_df, correlation_matrix, T=5.0):
        self.portfolio_df = portfolio_df
        self.correlation_matrix = correlation_matrix
        self.T = T
    
    def simulate_portfolio_loss(self, n_sims=10000):
        default_probs = [1 - np.exp(-row['Default_Rate'] * self.T) 
                        for _, row in self.portfolio_df.iterrows()]
        copula = GaussianCopulaModel(default_probs, self.correlation_matrix)
        default_times = copula.simulate_factor_model(self.T, n_sims)
        
        losses = np.zeros(n_sims)
        defaults_matrix = (default_times <= self.T).astype(int)
        
        for sim in range(n_sims):
            for i, (_, obligor) in enumerate(self.portfolio_df.iterrows()):
                if defaults_matrix[sim, i] == 1:
                    losses[sim] += obligor['Notional'] * (1 - obligor['Recovery_Rate'])
        
        return losses, defaults_matrix
    
    def calculate_risk_metrics(self, losses):
        return {
            'expected_loss': np.mean(losses),
            'unexpected_loss': np.std(losses),
            'var_95': np.percentile(losses, 95),
            'var_99': np.percentile(losses, 99),
            'var_999': np.percentile(losses, 99.9),
            'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
            'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)])
        }

portfolio_model = PortfolioCreditRisk(portfolio_df, correlation_matrix, T=5.0)
losses, defaults_matrix = portfolio_model.simulate_portfolio_loss(50000)
risk_metrics = portfolio_model.calculate_risk_metrics(losses)

total_exposure = portfolio_df['Notional'].sum()

print(f"Portfolio Risk (5yr, 50,000 sims):")
print(f"  Total Exposure: ${total_exposure/1e6:.1f}M")
print(f"  Expected Loss: ${risk_metrics['expected_loss']/1e6:.3f}M")
print(f"  Unexpected Loss: ${risk_metrics['unexpected_loss']/1e6:.3f}M")
print(f"  95% VaR: ${risk_metrics['var_95']/1e6:.3f}M")
print(f"  99% VaR: ${risk_metrics['var_99']/1e6:.3f}M")
print(f"  99.9% VaR: ${risk_metrics['var_999']/1e6:.3f}M")
print(f"  99% CVaR: ${risk_metrics['cvar_99']/1e6:.3f}M")

economic_capital = risk_metrics['var_999'] - risk_metrics['expected_loss']
print(f"  Economic Capital: ${economic_capital/1e6:.3f}M\n")

# %% [markdown]
# ## PART 6: STRESS TESTING
# 1. Takes the original portfolio and downgrades every company's credit rating by one or two notches
# - AAA → AA
# - AA+ → A+
# - A+ → BBB+, etc.
# 
# 2. Recalculates default rates based on the new (worse) ratings
# - Lower ratings = higher probability of default
# 
# 3. Re-runs the Monte Carlo simulation with these stressed parameters
# 
# 4. Compares the risk metrics between normal and stressed scenarios
# %%
# PART 6: STRESS TESTING
print("PART 6: STRESS TESTING")
print("-" * 80)

# Create a copy of the original portfolio to avoid modifying the base case
stressed_portfolio = portfolio_df.copy()

# Define rating downgrade mapping (each rating drops by 1-2 notches)
# This simulates a credit quality deterioration scenario
rating_map = {'AAA': 'AA', 'AA+': 'A+', 'AA': 'A', 'AA-': 'A-',
              'A+': 'BBB+', 'A': 'BBB', 'A-': 'BBB-',
              'BBB+': 'BB+', 'BBB': 'BB'}

# Apply the rating downgrades to all companies in the stressed portfolio
# Companies not in the map keep their original rating
stressed_portfolio['Rating'] = stressed_portfolio['Rating'].map(lambda r: rating_map.get(r, r))

# Update default rates based on the new (lower) credit ratings
# Lower ratings have higher default probabilities
stressed_portfolio['Default_Rate'] = stressed_portfolio['Rating'].map(default_rates)

# Create a new portfolio risk model with the stressed parameters
stressed_model = PortfolioCreditRisk(stressed_portfolio, correlation_matrix, T=5.0)

# Run Monte Carlo simulation with 20,000 scenarios to calculate stressed losses
# Each simulation generates correlated defaults and calculates portfolio loss
stressed_losses, _ = stressed_model.simulate_portfolio_loss(20000)

# Calculate risk metrics (VaR, CVaR, Expected Loss, etc.) under stress scenario
stressed_metrics = stressed_model.calculate_risk_metrics(stressed_losses)

print(f"Rating Downgrade Scenario:")
print(f"  Expected Loss: ${stressed_metrics['expected_loss']/1e6:.3f}M")
print(f"  99% VaR: ${stressed_metrics['var_99']/1e6:.3f}M")
print(f"  VaR Increase: {(stressed_metrics['var_99']/risk_metrics['var_99']-1)*100:.1f}%\n")

# %%
