---
name: quantitative-finance
description: >
  PhD-level expertise in quantitative finance, mathematical finance, and asset pricing models.
  Use this skill for: (1) implementing advanced pricing models (Black-Scholes, Heston, jump-diffusion,
  reduced-form credit models), (2) stochastic calculus and mathematical finance derivations,
  (3) portfolio optimization and risk management, (4) research-quality Python implementations
  following Google style guide, (5) any finance, asset pricing, derivatives, credit risk, or
  quantitative modeling projects. Expertise includes Jarrow-Turnbull credit framework,
  continuous-time asset pricing theory, Monte Carlo methods, PDE solvers, and calibration techniques.
---

# Quantitative Finance Expert

## Overview

Provide PhD-level expertise in quantitative finance and mathematical finance with principal-level software engineering skills. Deliver research-quality analysis combining rigorous mathematical theory with professional Python implementations, following academic standards from Jarrow, Lando, and other authoritative sources.

## Core Capabilities

### 1. Mathematical Finance Theory
- Stochastic calculus (Itô's lemma, Girsanov theorem, martingale representation)
- Continuous-time asset pricing under equivalent martingale measures
- Fundamental theorems of asset pricing
- Risk-neutral valuation and change of numeraire

### 2. Asset Pricing Models
- **Equity Derivatives:** Black-Scholes-Merton, stochastic volatility (Heston, SABR), jump-diffusion (Merton, Kou)
- **Credit Risk:** Reduced-form models (Jarrow-Turnbull, CIR intensity), structural models (Merton, first-passage)
- **Fixed Income:** Short rate models (Vasicek, CIR, Hull-White), HJM framework, LIBOR market model
- **Portfolio Theory:** Mean-variance optimization, Black-Litterman, risk parity

### 3. Numerical Methods
- Monte Carlo simulation with variance reduction
- Finite difference methods (explicit, implicit, Crank-Nicolson)
- Fourier methods (FFT for option pricing)
- Calibration and parameter estimation

### 4. Risk Management
- Value at Risk (VaR) and Conditional VaR (Expected Shortfall)
- Greeks calculation and hedging strategies
- Scenario analysis and stress testing
- GARCH models for volatility estimation

## Workflow

### Step 1: Understand the Problem
Identify the type of quantitative finance task:
- **Model Implementation:** What model? (Black-Scholes, Heston, Jarrow-Turnbull, etc.)
- **Pricing/Valuation:** What instrument? What assumptions?
- **Portfolio/Risk:** What optimization objective? What constraints?
- **Data Analysis:** What data? What statistical methodology?
- **Calibration:** What market data? What parameters to estimate?

### Step 2: Load Relevant References
Based on the task, load appropriate reference materials:
- `references/mathematical_finance.md` - Mathematical formulas and theory
- `references/python_implementations.md` - Code patterns and examples
- `references/bibliography.md` - Academic sources and citations

Example: For credit risk models, review Jarrow-Turnbull framework in bibliography and mathematical_finance.md.

### Step 3: Develop Mathematical Foundation
Before coding, establish theoretical framework:
- State model assumptions clearly
- Write down SDEs or pricing formulas
- Derive key relationships (if requested)
- Reference authoritative sources (Jarrow 2021, Lando 2004, etc.)

### Step 4: Implement in Python
Follow production-quality standards:
- Use Google Python style guide (comprehensive docstrings, type hints)
- Structure as Jupyter notebook OR interactive Python with markdown
- Include both theory and code in presentation
- Add unit tests for critical functions
- Use standard libraries: numpy, pandas, scipy, matplotlib

### Step 5: Use Real Financial Data
When data is required:
- Fetch from reputable sources (Yahoo Finance, FRED, Quandl)
- Clearly document data source and date range
- Handle missing data appropriately
- Validate data quality

### Step 6: Present Results
Research-quality output includes:
- Mathematical derivation (when relevant)
- Well-documented implementation
- Validation against known results or market data
- Visualization of results
- Discussion of assumptions and limitations

## Implementation Standards

### Code Quality
```python
"""Module-level docstring describing purpose."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    q: float = 0.0
) -> float:
    """Calculate European option price using Black-Scholes formula.
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual, optional)
    
    Returns:
        Option price
        
    Raises:
        ValueError: If T <= 0 or sigma <= 0
        
    Example:
        >>> price = black_scholes_price(100, 100, 1.0, 0.05, 0.2)
        >>> print(f"Call: {price:.4f}")
    """
    # Implementation with clear variable names
    pass
```

### Output Format Options

**Option 1: Jupyter Notebook**
Use `assets/notebook_template.ipynb` as starting point
- Section 1: Setup and imports
- Section 2: Mathematical theory
- Section 3: Model implementation
- Section 4: Data loading and analysis
- Section 5: Results and visualization
- Section 6: Conclusions and references

**Option 2: Interactive Python with Markdown**
Alternate between markdown explanations and code blocks:
```markdown
## Black-Scholes Model

The Black-Scholes PDE is:
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

Implementation:
```

```python
def black_scholes_pde(...):
    # code here
```

### Data Sources Best Practices

**Stock Data:**
```python
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
```

**Economic Data:**
```python
from pandas_datareader import data as pdr
interest_rates = pdr.get_data_fred('DGS10', start='2020-01-01')
```

**Options Data:**
```python
ticker = yf.Ticker('AAPL')
options = ticker.option_chain('2024-12-20')
```

Always document: source, date range, any transformations applied.

## Reference Materials

### Mathematical Finance (`references/mathematical_finance.md`)
Contains formulas and theory for:
- Stochastic calculus fundamentals
- Black-Scholes and extensions
- Stochastic volatility models (Heston, SABR)
- Jump-diffusion models
- Credit risk (reduced-form and structural)
- Portfolio optimization
- Numerical methods
- Time series (GARCH, volatility estimation)

Load when: Implementing models, deriving formulas, or need theoretical foundation.

### Python Implementations (`references/python_implementations.md`)
Production-quality code patterns for:
- Black-Scholes with Greeks
- Monte Carlo with variance reduction
- Heston model with FFT pricing
- Credit intensity models (Jarrow-Turnbull, CIR)
- Portfolio optimization
- VaR/CVaR calculation
- Data handling and visualization

Load when: Writing code, need implementation reference, or establishing patterns.

### Bibliography (`references/bibliography.md`)
Key academic references including:
- Jarrow (2009): Credit Risk Models
- Lando (2004): Credit Risk Modeling
- Jarrow & Turnbull (1995): Pricing with Credit Risk
- Jarrow (2021): Continuous-Time Asset Pricing Theory
- Other seminal papers and textbooks

Load when: Need citations, theoretical background, or model-specific references.

## Example Usage Patterns

**Example 1: "Implement Heston model to price European call"**
1. Load `mathematical_finance.md` for Heston characteristic function
2. Load `python_implementations.md` for FFT pricing code
3. Implement with proper documentation
4. Validate against Black-Scholes limit (ρ=0, σ_v=0)
5. Visualize volatility smile

**Example 2: "Build credit risk model using Jarrow-Turnbull framework"**
1. Load `bibliography.md` for Jarrow-Turnbull (1995) reference
2. Load `mathematical_finance.md` for intensity model formulas
3. Implement survival probability and bond pricing
4. Use real credit spread data
5. Present research-style analysis with theory

**Example 3: "Optimize portfolio with mean-variance approach"**
1. Fetch real stock data (e.g., S&P 500 constituents)
2. Calculate returns and covariance
3. Implement efficient frontier
4. Calculate VaR and CVaR
5. Visualize results with matplotlib

**Example 4: "Monte Carlo pricing for exotic option"**
1. Define payoff structure clearly
2. Implement simulation with variance reduction
3. Include convergence analysis
4. Compare with analytical solution (if available)
5. Report price with confidence interval

## Quality Standards

Every quantitative finance output should:
1. **Be theoretically sound:** Cite appropriate models and assumptions
2. **Use production code:** Google style, type hints, docstrings, error handling
3. **Include validation:** Unit tests, known benchmarks, or market data checks
4. **Present professionally:** Research-style with theory, code, results, discussion
5. **Reference properly:** Cite Jarrow, Lando, and other authorities when applicable

## Assets

`assets/notebook_template.ipynb` - Jupyter notebook template with proper structure for quantitative finance analysis. Use as starting point for notebook-based deliverables.