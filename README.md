# lau-landauer-meter

> Landauer's principle for agents: compute cost = thermodynamic work = Fisher-Rao distance², lower-bounded by kT ln 2 per bit erased

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## What This Does

This crate measures the **thermodynamic cost of learning** for self-improving agents. It connects three physical principles:

1. **Landauer's principle**: Erasing one bit of information dissipates at least kT ln 2 ≈ 2.87 × 10⁻²¹ J at room temperature
2. **Fisher-Rao distance**: The geodesic distance on a statistical manifold measures how far beliefs move during an update
3. **Varadhan's formula**: The thermodynamic work of a belief update equals the squared Fisher-Rao distance scaled by temperature

Together, these form the **Opus Emergent Theorem D**: the compute cost of one self-modeling loop equals the thermodynamic work of the belief update, which equals the squared Fisher-Rao distance moved, lower-bounded by Landauer dissipation kT ln 2 per bit erased.

The crate provides tools to count FLOPs, measure Fisher-Rao distances, compute thermodynamic costs, audit self-improvement efficiency, and detect wasteful learning phases.

## Key Idea

When an agent updates its beliefs, it moves on a statistical manifold (the space of probability distributions parameterized by its model). The **Fisher-Rao distance** measures the geodesic length of this movement. By Varadhan's formula, the thermodynamic work done is proportional to the *square* of this distance.

Meanwhile, Landauer's principle provides a fundamental lower bound: each bit of information erased costs at least kT ln 2 joules. The ratio of actual compute cost to Landauer minimum gives a **thermodynamic efficiency** metric — how close is the agent to the physical limits of efficient learning?

This crate lets you track this efficiency in real-time via the `PlatoMonitor`, which detects when an agent is spending many FLOPs for tiny belief updates (wasteful learning) versus efficiently converting computation into genuine belief change.

## Install

```toml
[dependencies]
lau-landauer-meter = "0.1"
```

### Dependencies

- **nalgebra** 0.33 — vectors and matrices for Fisher information matrices
- **serde** 1 (with `derive`) — serialization of all types

Dev dependency: **serde_json** 1 for test roundtrips.

## Quick Start

### Landauer Bit Energy

```rust
use lau_landauer_meter::*;

// Energy to erase one bit at room temperature (300K)
let energy = landauer_bit(300.0);
// ≈ 2.87 × 10⁻²¹ joules
println!("One bit erased: {:.2e} J", energy);
```

### Fisher-Rao Distances

```rust
// Between two 1D Gaussians
let d = FisherRao::distance_1d(0.0, 1.0, 1.0); // μ₁=0, μ₂=1, σ=1

// Between two categorical distributions
let d = FisherRao::distance_categorical(&[0.5, 0.5], &[0.99, 0.01]);

// Using a Fisher information matrix (Mahalanobis-like)
let theta1 = DVector::from_vec(vec![0.0, 0.0]);
let theta2 = DVector::from_vec(vec![1.0, 1.0]);
let fim = DMatrix::identity(2, 2);
let d = FisherRao::distance_fim(&theta1, &theta2, &fim);
```

### FLOP Counting

```rust
let mut counter = FlopCounter::new();
counter.record_matmul(128, 64, 256);   // 2×128×64×256 = 4,194,304 FLOPs
counter.record_matrix_inverse(100);     // ~⅔ × 100³ ≈ 666,667 FLOPs
counter.record_cholesky(50);            // ~50³/3 ≈ 41,667 FLOPs
println!("Total: {} FLOPs", counter.total_flops());
println!("Average per update: {:.0}", counter.avg_flops());
```

### Full Self-Improvement Audit

```rust
let temp = 300.0;
let mut flops = FlopCounter::new();
let mut varadhan = VaradhanBridge::new(temp);
let mut landauer = LandauerBound::new(temp);
let mut audit = SelfImprovementAudit::new(temp);

// Simulate a belief update
let compute_flops = 1_000_000u64;
let fisher_rao_distance = 0.5;
let bits_erased = 3u64;

flops.record(compute_flops);
let thermo_cost = varadhan.record_step(fisher_rao_distance); // d² × T = 0.25 × 300 = 75
landauer.erase(bits_erased);
audit.record_update(compute_flops, bits_erased, fisher_rao_distance, 1);

// Get the report
let report = audit.report();
println!("Efficiency: {:.6}", report.overall_efficiency);
println!("Cost per bit: {:.2}", report.cost_per_bit);
```

### PLATO Monitor (Real-Time Efficiency)

```rust
let mut monitor = PlatoMonitor::new(300.0, 0.01); // temp=300K, wasteful threshold=1%

// Monitor each learning step
let result = monitor.monitor_step(
    1_000_000,          // FLOPs spent
    &[0.5, 0.5],        // old parameters
    &[0.6, 0.4],        // new parameters
    &[4.0, 4.0],        // FIM diagonal
    42,                  // timestamp
);

println!("Fisher-Rao distance: {:.6}", result.fisher_rao_distance);
println!("Efficiency: {:.6}", result.efficiency);
println!("Wasteful: {}", result.is_wasteful);

// Check for alerts
for alert in monitor.alerts() {
    println!("⚠️ {}", alert);
}
```

## API Reference

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BOLTZMANN` | 1.380649 × 10⁻²³ J/K | Boltzmann constant |
| `DEFAULT_TEMPERATURE` | 300.0 K | Room temperature |
| `LN2` | ln(2) | Natural log of 2 |
| `landauer_bit(T)` | kT ln 2 | Energy to erase one bit at temperature T |

### FLOP Counter (`FlopCounter`)

| Method | FLOPs Counted |
|--------|---------------|
| `.record(n)` | n FLOPs |
| `.record_matmul(m, k, n)` | 2mnk |
| `.record_dot(d)` | 2d |
| `.record_matrix_inverse(n)` | ~⅔n³ |
| `.record_cholesky(n)` | ~n³/3 |
| `.record_gradient(n)` | n |

### Fisher-Rao Distances (`FisherRao`)

| Method | Formula |
|--------|---------|
| `.distance_1d(μ₁, μ₂, σ)` | √2 · \|arctan(μ₁/σ) − arctan(μ₂/σ)\| |
| `.distance_categorical(p, q)` | 2 · arccos(Σ √(pᵢqᵢ)) |
| `.distance_fim(θ₁, θ₂, F)` | √((θ₂−θ₁)ᵀ F (θ₂−θ₁)) |
| `.fim_gaussian_known_cov(Σ)` | Σ⁻¹ |
| `.fim_bernoulli(p)` | diag(1/(pᵢ(1−pᵢ))) |

### Varadhan Bridge (`VaradhanBridge`)

| Method | Description |
|--------|-------------|
| `.new(T)` | Create at temperature T |
| `.step_cost(d)` | d² × T — thermodynamic cost of Fisher-Rao step |
| `.record_step(d)` | Record step and accumulate cost |
| `.compute_cost_from_flops(n)` | n × kT — Landauer bound from FLOPs |

### Landauer Bound (`LandauerBound`)

| Method | Description |
|--------|-------------|
| `.new(T)` | Create at temperature T |
| `.erase(n)` | Record n bits erased, return dissipation |
| `.erase_from_entropy(H₁, H₂)` | Bits = max(⌈H₁−H₂⌉, 1) |
| `.min_dissipation(n)` | n × kT ln 2 |

### Thermodynamic Efficiency (`ThermodynamicEfficiency`)

| Method | Description |
|--------|-------------|
| `.efficiency()` | Landauer minimum / actual cost (0→1) |
| `.is_wasteful(threshold)` | efficiency < threshold |
| `.record(flops, bits, distance)` | Log one update step |

### Curvature Exchange Rate (`CurvatureExchange`)

| Method | Description |
|--------|-------------|
| `.ricci_from_fim(F)` | Approximate Ricci scalar from FIM |
| `.set_curvature(R)` | Set curvature and compute cost rate |
| `.flops_for_step(d)` | FLOPs needed for geodesic step of length d |

### Self-Improvement Audit (`SelfImprovementAudit`)

| Method | Description |
|--------|-------------|
| `.record_update(flops, bits, distance, ts)` | Log complete step |
| `.overall_efficiency()` | Cumulative Landauer / actual |
| `.cost_per_bit()` | Thermodynamic cost per bit erased |
| `.is_wasteful_phase(window, threshold)` | Detect sustained wasteful learning |
| `.report()` | Full `AuditReport` struct |

### PLATO Monitor (`PlatoMonitor`)

| Method | Description |
|--------|-------------|
| `.new(T, threshold)` | Create monitor with wasteful threshold |
| `.monitor_step(flops, old, new, fim, ts)` | Complete step analysis |
| `.alerts()` | Wasteful learning alert log |
| `.audit_report()` | Full audit summary |

## How It Works

1. **FLOP counting**: The `FlopCounter` records floating-point operations with standard operation counts (matmul = 2mnk, inverse ≈ ⅔n³, etc.).

2. **Fisher-Rao distance**: For 1D Gaussians, the exact closed-form geodesic distance is √2 · |arctan(μ₁/σ) − arctan(μ₂/σ)|. For categorical distributions, d(p,q) = 2 arccos(Σ √(pᵢqᵢ)). For general parameter vectors with a Fisher information matrix, the Mahalanobis-like metric √(δᵀFδ) approximates the geodesic.

3. **Varadhan bridge**: Each belief update has thermodynamic cost = d² × T where d is the Fisher-Rao distance. This accumulates over an agent's lifetime.

4. **Landauer bound**: Every bit erased costs at least kT ln 2 ≈ 2.87 × 10⁻²¹ J at 300K. The ratio of actual energy (flops × kT) to Landauer minimum gives thermodynamic efficiency.

5. **Curvature exchange rate**: The Ricci scalar curvature of the statistical manifold determines how expensive learning is in different regions of belief space. Higher curvature = more FLOPs per geodesic step.

6. **PLATO monitoring**: The `PlatoMonitor` combines all of the above in a single interface. Each learning step computes Fisher-Rao distance, counts bits erased, records efficiency, and alerts when learning is wasteful (high FLOPs, tiny belief change).

## The Math

### Landauer's Principle

```
E_min = kT ln 2  (per bit erased)
```

At T = 300K: E_min ≈ 2.87 × 10⁻²¹ J. This is the fundamental thermodynamic minimum for irreversible computation.

### Fisher-Rao Distance

The Fisher-Rao metric is the unique Riemannian metric on the space of probability distributions that is invariant under sufficient statistics reparameterization.

For 1D Gaussians with shared σ:
```
d(μ₁, μ₂) = √2 · |arctan(μ₁/σ) − arctan(μ₂/σ)|
```

For categorical distributions:
```
d(p, q) = 2 · arccos(Σᵢ √(pᵢ qᵢ))  ∈ [0, π]
```

### Fisher Information Matrix

For a multivariate Gaussian with known covariance Σ:
```
F = Σ⁻¹
```

For independent Bernoulli parameters:
```
F_ii = 1 / (pᵢ(1 − pᵢ))
```

### Varadhan's Formula

The thermodynamic work of moving from θ₁ to θ₂ on the statistical manifold:
```
W = d_Fisher-Rao(θ₁, θ₂)² × T
```

This connects information geometry to thermodynamics: learning costs energy proportional to the square of the distance moved in belief space.

### Thermodynamic Efficiency

```
η = E_Landauer / E_actual = (bits_erased × kT ln 2) / (FLOPs × kT)
```

Perfect efficiency (η = 1) means every FLOP erases exactly one bit at the Landauer minimum. In practice, η ≪ 1 due to algorithmic overhead.

### Ricci Curvature as Cost Rate

```
R ≈ −tr(F⁻¹ · H[ln det F])
```

Higher |Ricci| means the manifold curves more sharply, making geodesic steps more expensive. The cost rate = 1 + |R| gives FLOPs per unit geodesic distance.

## Tests

75 unit tests covering:
- Landauer bit energy at physical constants
- FLOP counter: matmul, dot product, matrix inverse, Cholesky, gradient
- Fisher-Rao distances: 1D Gaussian, categorical, FIM-based (symmetry, identity, bounds)
- Fisher information matrices: Gaussian known covariance, Bernoulli
- Varadhan bridge: step cost, cumulative cost, zero distance, FLOP-based cost
- Landauer bound: single/multiple erasure, entropy-based erasure
- Thermodynamic efficiency: perfect/wasteful detection
- Curvature exchange: Ricci from FIM, cost rate
- Bit erasure counting: parameter change, KL-based, entropy-based
- Temperature calibration from FLOP/distance ratio
- Self-improvement audit: full pipeline, cost per bit, wasteful phase detection
- PLATO monitor: step monitoring, alerts, audit reports
- Serde roundtrips for all serializable types

Run with: `cargo test`

## License

MIT
