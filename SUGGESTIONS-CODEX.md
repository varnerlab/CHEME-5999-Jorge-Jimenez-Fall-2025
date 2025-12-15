# Credit_Risk Code Issues and Suggested Fixes

1) `Credit_Risk/1_Advanced_Reduced_Form.py` – Stress test results only keep the last scenario because `stress_results.append(...)` sits outside the scenario loop (lines ~983-1007).  
   Suggested fix: move the append inside the `for scenario_name, shocks in scenarios.items():` loop so each scenario is recorded before moving to the next one.

2) `Credit_Risk/1_Advanced_Reduced_Form.py` – Monte Carlo bond pricer flags default with a single uniform draw against `exp(-∫λ dt)`, then discounts recovery to maturity using the full rate path (lines ~432-463). This understates losses and ignores default timing.  
   Suggested fix: simulate default time by accumulating hazard until the exponential threshold is hit (or sample τ from the path) and, if default occurs before T, discount recovery to τ, not to T.

3) `Credit_Risk/1_Advanced_Reduced_Form.py` – The “theoretical” CDS spread used for calibration collapses the premium and protection legs to simple averages (lines ~273-295). It cannot fit a term structure and mis-calibrates CIR parameters.  
   Suggested fix: implement the standard CDS pricing integrals: premium leg `∫_0^T P(0,t) S Q(τ>t) dt` and protection leg `∫_0^T P(0,t) (1-δ) λ(t) Q(τ>t) dt`, using either semi-analytic CIR survival or numerical quadrature, then minimize squared errors to market spreads.

4) `Credit_Risk/1_Advanced_Reduced_Form.py` – `VasicekModel.zero_coupon_bond` always discounts off `r0` (line ~144), ignoring the short rate at pricing time.  
   Suggested fix: take the current rate as an argument (`r_t`) and use it instead of `r0`, or add an overload `zero_coupon_bond_from_rate(r_t, t, T)` so path-dependent pricing is consistent with simulated rates.

5) `Credit_Risk/2_Glasserman_Credit Risk_interactive_v2.py` – CIR default-time simulation computes ∫λ dt over the path and treats it as a single exponential draw (lines ~139-168), effectively using an average intensity and distorting timing/default probabilities.  
   Suggested fix: simulate arrival time via cumulative hazard: draw E~Exp(1) and step through λ(t)·Δt until the running sum exceeds E; default at that step. Alternatively, use exact CIR sampling for τ when available.

6) `Credit_Risk/2_Glasserman_Credit Risk_interactive_v2.py` – The Gaussian copula ignores the supplied correlation matrix; defaults are driven by a hard-coded β=0.3 factor model (lines ~235-260).  
   Suggested fix: use the provided `correlation_matrix` via Cholesky (or eigen) decomposition to correlate the latent normals, or calibrate β from that matrix rather than replacing it.

7) `Credit_Risk/Merton_Model_(MULTIPLE FIRMS).ipynb` – In `MertonParameterEstimator.estimate_asset_parameters_iterative`, the early break when σ or T ≤ 0 exits without setting `converged`, causing an exception on return.  
   Suggested fix: initialize `converged = False` before the loop (and ensure it is returned) or set it explicitly in the edge-case branch.

8) `Credit_Risk/Merton_Model_(MULTIPLE FIRMS).ipynb` – Risk tiers in `analyze_portfolio` use `pd.cut` with bins (0,0.001,…); PD values of exactly 0 fall outside bins and become NaN (later cells apply a manual fix).  
   Suggested fix: make bins left-closed at 0 (e.g., `bins=[-np.inf, 0.001, 0.01, 0.05, np.inf]`) or add `include_lowest=True`/`right=True` to catch zero PDs.

9) `Credit_Risk/Merton_gmn_BASE_v1.ipynb` and `Credit_Risk/Merton_Model_(MULTIPLE FIRMS).ipynb` – Notebooks install packages and pull live data (yfinance/FRED) at runtime, making them non-reproducible in restricted/offline environments.  
   Suggested fix: move installs/data fetches to a setup script with pinned versions and cache sample data locally (or gate network calls behind a flag) so the notebooks run deterministically without internet access.
