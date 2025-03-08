# Geometric Brownian Motion Simulation Analysis

## Results Summary

| Parameter Type | Parameter Values | Theoretical Probability | Empirical Probability | Absolute Error | Relative Error (%) |
|----------------|------------------|------------------------|----------------------|----------------|-------------------|
| **Drift (μ)** | μ = -0.10, σ = 0.2 | 0.3778 | 0.3688 | 0.0090 | 2.39 |
|  | μ = -0.05, σ = 0.2 | 0.4378 | 0.4332 | 0.0046 | 1.06 |
|  | μ = 0.00, σ = 0.2 | 0.5000 | 0.4944 | 0.0056 | 1.12 |
|  | μ = 0.05, σ = 0.2 | 0.5624 | 0.5626 | 0.0002 | 0.03 |
|  | μ = 0.10, σ = 0.2 | 0.6231 | 0.6261 | 0.0030 | 0.47 |
| **Volatility (σ)** | μ = 0.05, σ = 0.1 | 0.7330 | 0.7306 | 0.0024 | 0.33 |
|  | μ = 0.05, σ = 0.2 | 0.5624 | 0.5587 | 0.0037 | 0.66 |
|  | μ = 0.05, σ = 0.3 | 0.5278 | 0.5291 | 0.0013 | 0.24 |
|  | μ = 0.05, σ = 0.4 | 0.5157 | 0.5045 | 0.0112 | 2.16 |
| **Boundaries** | S₀ = 100, α = 105, β = 95 | 0.5312 | 0.5314 | 0.0002 | 0.03 |
|  | S₀ = 100, α = 110, β = 90 | 0.5624 | 0.5619 | 0.0005 | 0.09 |
|  | S₀ = 100, α = 120, β = 80 | 0.6243 | 0.6212 | 0.0031 | 0.49 |
|  | S₀ = 100, α = 120, β = 90 | 0.4171 | 0.4209 | 0.0038 | 0.91 |
|  | S₀ = 100, α = 110, β = 80 | 0.7490 | 0.7428 | 0.0062 | 0.83 |

## Insights on Geometric Brownian Motion Simulations

My simulation study of Geometric Brownian Motion (GBM) validates the theoretical framework with remarkable accuracy. The simulation results consistently demonstrate a close alignment between theoretical probabilities and their empirical counterparts, with relative errors typically below 1% and only occasionally exceeding 2%.

### Effect of Drift Parameter (μ)

The drift parameter shows a clear directional impact on hitting probabilities:

- With negative drift (μ = -0.10), the probability of hitting the upper boundary before the lower boundary is only 36.88%, as the process is naturally biased downward.
- At zero drift (μ = 0.00), we observe a near 50/50 split (49.44%) between hitting upper and lower boundaries.
- With positive drift (μ = 0.10), the probability increases substantially to 62.61% for hitting the upper boundary first.

This confirms the intuitive expectation that drift significantly influences the directional tendency of the process, with each 0.05 increment in μ increasing the upper boundary hitting probability by approximately 6-7 percentage points.

### Impact of Volatility (σ)

Volatility demonstrates a non-linear relationship with boundary hitting probabilities:

- At low volatility (σ = 0.1), with positive drift μ = 0.05, the process has a high probability (73.06%) of hitting the upper boundary first, as the path is more deterministic.
- As volatility increases to σ = 0.4, this probability decreases to 50.45%, approaching a 50/50 chance despite the positive drift.

This reveals that higher volatility diminishes the influence of drift, causing the process to behave more randomly even when there's a directional bias. Essentially, as σ increases, randomness increasingly dominates over the drift's directional force.

### Boundary Placement Effects

The distance and symmetry of boundaries significantly impact hitting probabilities:

- With symmetric but narrow boundaries (α = 105, β = 95), the probability of hitting the upper boundary is 53.14%.
- As boundaries widen symmetrically (α = 110, β = 90), this probability increases to 56.19%.
- With asymmetric boundaries, the effects are more pronounced:
  - When the upper boundary is more distant (α = 120, β = 90), the upper hitting probability decreases to 42.09%.
  - When the lower boundary is more distant (α = 110, β = 80), the upper hitting probability increases substantially to 74.28%.

These findings highlight that both the absolute and relative distances of boundaries from the starting point are crucial factors in determining hitting probabilities.

### Statistical Validity

The simulation demonstrates excellent agreement between theoretical and empirical probabilities across all parameter sets:

- The maximum absolute error is only 0.0112 (for μ = 0.05, σ = 0.4)
- The maximum relative error is 2.39% (for μ = -0.10, σ = 0.2)
- The consistency of these results across 10,000 simulation runs per parameter set (implied from hit counts) provides strong confidence in the statistical validity of both the theoretical formulas and simulation methodology.