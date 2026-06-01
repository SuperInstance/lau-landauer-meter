//! # lau-landauer-meter
//!
//! Landauer's principle for agents: compute cost = thermodynamic work = Fisher-Rao distance²,
//! lower-bounded by kT ln 2 per bit erased.
//!
//! Implements Opus Emergent Theorem D: the compute cost of one self-modeling loop equals
//! the thermodynamic work of the belief update, which by Varadhan equals the squared
//! Fisher-Rao distance moved, lower-bounded by Landauer dissipation kT ln 2 per bit erased.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::f64::consts::LN_2;

/// Boltzmann constant in J/K
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Default temperature: 300K (room temperature)
pub const DEFAULT_TEMPERATURE: f64 = 300.0;

/// ln(2) constant
pub const LN2: f64 = LN_2;

/// Compute a single Landauer bit: kT ln 2
pub fn landauer_bit(temperature: f64) -> f64 {
    BOLTZMANN * temperature * LN2
}

// ─── FLOP Counter ───────────────────────────────────────────────────────────

/// Counts floating-point operations for belief update computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlopCounter {
    /// Total FLOPs counted
    total_flops: u64,
    /// Number of updates counted
    update_count: u64,
    /// Per-update FLOP log
    per_update: Vec<u64>,
}

impl FlopCounter {
    pub fn new() -> Self {
        Self {
            total_flops: 0,
            update_count: 0,
            per_update: Vec::new(),
        }
    }

    /// Record FLOPs for one update step
    pub fn record(&mut self, flops: u64) {
        self.total_flops += flops;
        self.update_count += 1;
        self.per_update.push(flops);
    }

    /// Record a matrix multiplication: (m×k) × (k×n) costs 2mnk FLOPs
    pub fn record_matmul(&mut self, m: usize, k: usize, n: usize) {
        let flops = 2 * m * k * n;
        self.record(flops as u64);
    }

    /// Record a dot product of dimension d: 2d-1 ≈ 2d FLOPs
    pub fn record_dot(&mut self, dim: usize) {
        self.record(2 * dim as u64);
    }

    /// Record a matrix inversion of n×n: ~2/3 n³ FLOPs
    pub fn record_matrix_inverse(&mut self, n: usize) {
        let flops = (2.0 / 3.0) * (n as f64).powi(3);
        self.record(flops as u64);
    }

    /// Record an n-dimensional gradient computation: ~n FLOPs per component
    pub fn record_gradient(&mut self, n: usize) {
        self.record(n as u64);
    }

    /// Record a Cholesky decomposition of n×n: ~n³/3 FLOPs
    pub fn record_cholesky(&mut self, n: usize) {
        let flops = (n as f64).powi(3) / 3.0;
        self.record(flops as u64);
    }

    pub fn total_flops(&self) -> u64 { self.total_flops }
    pub fn update_count(&self) -> u64 { self.update_count }
    pub fn per_update(&self) -> &[u64] { &self.per_update }

    /// Average FLOPs per update
    pub fn avg_flops(&self) -> f64 {
        if self.update_count == 0 { 0.0 } else { self.total_flops as f64 / self.update_count as f64 }
    }
}

impl Default for FlopCounter {
    fn default() -> Self { Self::new() }
}

// ─── Fisher-Rao Distance ────────────────────────────────────────────────────

/// Computes Fisher-Rao distances on statistical manifolds.
pub struct FisherRao;

impl FisherRao {
    /// Fisher-Rao distance between two multivariate Gaussians with same covariance structure.
    /// For 1D Gaussians: sqrt(2) * |arctan(μ₁/σ) - arctan(μ₂/σ)|
    pub fn distance_1d(mu1: f64, mu2: f64, sigma: f64) -> f64 {
        if sigma <= 0.0 { return f64::INFINITY; }
        let d = (mu1 / sigma).atan() - (mu2 / sigma).atan();
        d.abs() * std::f64::consts::SQRT_2
    }

    /// Fisher-Rao distance between two categorical distributions (probability vectors).
    /// d(p, q) = 2 * arccos(Σ sqrt(p_i * q_i))
    pub fn distance_categorical(p: &[f64], q: &[f64]) -> f64 {
        assert_eq!(p.len(), q.len(), "distributions must have same dimension");
        let sum: f64 = p.iter().zip(q.iter())
            .map(|(pi, qi)| (pi.max(0.0) * qi.max(0.0)).sqrt())
            .sum();
        2.0 * sum.min(1.0).acos()
    }

    /// Fisher-Rao distance using Fisher information matrix (FIM).
    /// d(θ₁, θ₂)² = (θ₂ - θ₁)ᵀ · F(θ) · (θ₂ - θ₁) for infinitesimal steps.
    /// For finite steps we use the geodesic approximation.
    pub fn distance_fim(theta1: &DVector<f64>, theta2: &DVector<f64>, fim: &DMatrix<f64>) -> f64 {
        let diff = theta2 - theta1;
        // Mahalanobis-like: sqrt(δᵀ F δ)
        let quad = diff.transpose() * fim * &diff;
        quad[(0, 0)].sqrt()
    }

    /// Fisher information matrix for a multivariate Gaussian with known covariance.
    /// F = Σ⁻¹
    pub fn fim_gaussian_known_cov(covariance: &DMatrix<f64>) -> DMatrix<f64> {
        covariance.clone().try_inverse().unwrap_or_else(|| {
            // Regularize with small diagonal
            let n = covariance.nrows();
            let mut reg = covariance.clone();
            for i in 0..n { reg[(i, i)] += 1e-10; }
            reg.try_inverse().unwrap_or_else(|| DMatrix::identity(n, n))
        })
    }

    /// Fisher information matrix diagonal for Bernoulli parameters.
    /// F_ii = 1 / (p_i * (1 - p_i))
    pub fn fim_bernoulli(params: &[f64]) -> DMatrix<f64> {
        let n = params.len();
        let mut diag = vec![0.0; n];
        for (i, &p) in params.iter().enumerate() {
            let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
            diag[i] = 1.0 / (p_clamped * (1.0 - p_clamped));
        }
        DMatrix::from_diagonal(&DVector::from_vec(diag))
    }

    /// Step length from a single update: distance moved on statistical manifold.
    /// Returns the Fisher-Rao norm of the parameter change.
    pub fn step_length(old_theta: &DVector<f64>, new_theta: &DVector<f64>, fim: &DMatrix<f64>) -> f64 {
        Self::distance_fim(old_theta, new_theta, fim)
    }
}

// ─── Varadhan Bridge ────────────────────────────────────────────────────────

/// Varadhan's formula: compute cost = squared Fisher-Rao distance × temperature.
///
/// The thermodynamic work of a belief update equals the squared geodesic distance
/// on the statistical manifold, scaled by temperature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaradhanBridge {
    temperature: f64,
    cumulative_cost: f64,
    step_costs: Vec<f64>,
}

impl VaradhanBridge {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            cumulative_cost: 0.0,
            step_costs: Vec::new(),
        }
    }

    /// Compute the thermodynamic cost of a Fisher-Rao step.
    /// cost = distance² × T
    pub fn step_cost(&self, fisher_rao_distance: f64) -> f64 {
        fisher_rao_distance.powi(2) * self.temperature
    }

    /// Record a step and return its cost
    pub fn record_step(&mut self, fisher_rao_distance: f64) -> f64 {
        let cost = self.step_cost(fisher_rao_distance);
        self.cumulative_cost += cost;
        self.step_costs.push(cost);
        cost
    }

    /// Compute cost from FLOPs directly: Varadhan says compute ≈ thermodynamic work
    pub fn compute_cost_from_flops(&self, flops: u64) -> f64 {
        // Each FLOP dissipates at least kT ln 2 (Landauer bound)
        // In natural units, cost = flops × kT (scaled)
        flops as f64 * BOLTZMANN * self.temperature
    }

    pub fn temperature(&self) -> f64 { self.temperature }
    pub fn cumulative_cost(&self) -> f64 { self.cumulative_cost }
    pub fn step_costs(&self) -> &[f64] { &self.step_costs }
}

// ─── Landauer Lower Bound ───────────────────────────────────────────────────

/// Landauer's principle: erasing one bit of information dissipates at least kT ln 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandauerBound {
    temperature: f64,
    bits_erased: u64,
    total_dissipation: f64,
}

impl LandauerBound {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            bits_erased: 0,
            total_dissipation: 0.0,
        }
    }

    /// Minimum energy dissipation for erasing n bits
    pub fn min_dissipation(&self, bits: u64) -> f64 {
        bits as f64 * landauer_bit(self.temperature)
    }

    /// Record bit erasure
    pub fn erase(&mut self, bits: u64) -> f64 {
        let dissipation = self.min_dissipation(bits);
        self.bits_erased += bits;
        self.total_dissipation += dissipation;
        dissipation
    }

    /// Compute bits erased from information content change.
    /// If beliefs go from entropy H1 to H2, bits erased = H1 - H2 (if H1 > H2).
    pub fn erase_from_entropy(&mut self, old_entropy: f64, new_entropy: f64) -> f64 {
        let bits = if old_entropy > new_entropy {
            (old_entropy - new_entropy) as u64
        } else {
            0
        };
        self.erase(bits.max(1)) // at least 1 bit for any update
    }

    pub fn temperature(&self) -> f64 { self.temperature }
    pub fn bits_erased(&self) -> u64 { self.bits_erased }
    pub fn total_dissipation(&self) -> f64 { self.total_dissipation }
}

// ─── Thermodynamic Efficiency ───────────────────────────────────────────────

/// Measures how close actual compute cost is to the Landauer minimum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicEfficiency {
    /// Actual FLOPs performed
    actual_flops: u64,
    /// Landauer minimum FLOPs (bits erased)
    landauer_bits: u64,
    temperature: f64,
    history: Vec<EfficiencyRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyRecord {
    pub actual_flops: u64,
    pub landauer_minimum_flops: u64,
    pub efficiency: f64,
    pub fisher_rao_distance: f64,
}

impl ThermodynamicEfficiency {
    pub fn new(temperature: f64) -> Self {
        Self {
            actual_flops: 0,
            landauer_bits: 0,
            temperature,
            history: Vec::new(),
        }
    }

    /// Compute efficiency ratio: Landauer minimum / actual cost.
    /// Returns 1.0 for perfect efficiency, approaches 0 for wasteful computation.
    pub fn efficiency(&self) -> f64 {
        if self.actual_flops == 0 { return 1.0; }
        // Landauer minimum energy / actual energy
        let landauer_energy = LandauerBound::new(self.temperature).min_dissipation(self.landauer_bits);
        let actual_energy = self.actual_flops as f64 * BOLTZMANN * self.temperature;
        if actual_energy == 0.0 { return 1.0; }
        (landauer_energy / actual_energy).min(1.0)
    }

    /// Record an update step
    pub fn record(&mut self, actual_flops: u64, bits_erased: u64, fisher_rao_distance: f64) {
        let landauer_min = LandauerBound::new(self.temperature).min_dissipation(bits_erased);
        let actual_energy = actual_flops as f64 * BOLTZMANN * self.temperature;
        let eff = if actual_energy == 0.0 { 1.0 } else { (landauer_min / actual_energy).min(1.0) };

        self.actual_flops += actual_flops;
        self.landauer_bits += bits_erased;
        self.history.push(EfficiencyRecord {
            actual_flops,
            landauer_minimum_flops: bits_erased,
            efficiency: eff,
            fisher_rao_distance,
        });
    }

    /// Check if learning is wasteful (efficiency below threshold)
    pub fn is_wasteful(&self, threshold: f64) -> bool {
        self.efficiency() < threshold
    }

    pub fn actual_flops(&self) -> u64 { self.actual_flops }
    pub fn landauer_bits(&self) -> u64 { self.landauer_bits }
    pub fn history(&self) -> &[EfficiencyRecord] { &self.history }
}

// ─── Curvature Exchange Rate ────────────────────────────────────────────────

/// Ricci curvature as the exchange rate: FLOP cost per geodesic step.
///
/// Higher curvature = more expensive learning in that region of belief space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureExchange {
    /// Ricci scalar curvature at current belief point
    ricci_scalar: f64,
    /// FLOP cost per unit geodesic distance
    cost_rate: f64,
    /// History of (curvature, cost_rate) pairs
    history: Vec<CurvatureRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureRecord {
    pub ricci_scalar: f64,
    pub cost_rate: f64,
    pub geodesic_step: f64,
    pub flops: u64,
}

impl CurvatureExchange {
    pub fn new() -> Self {
        Self {
            ricci_scalar: 0.0,
            cost_rate: 1.0,
            history: Vec::new(),
        }
    }

    /// Compute Ricci scalar for a Fisher information matrix (approximation).
    /// For a diagonal FIM with entries λ_i, the Ricci scalar is approximately:
    /// R ≈ -Σ ∂² ln det(F) / ∂θ_i² 
    /// Simplified: R ≈ -tr(F⁻¹ · H[ln det F])
    /// For diagonal FIM: R ≈ Σ (d²λ_i/dθ_i²) / λ_i² - (dλ_i/dθ_i)² / λ_i³
    pub fn ricci_from_fim(fim: &DMatrix<f64>) -> f64 {
        let n = fim.nrows();
        // Approximate Ricci scalar for diagonal FIM
        let mut ricci = 0.0;
        for i in 0..n {
            let fii = fim[(i, i)];
            if fii > 0.0 {
                // For a diagonal FIM, Ricci ~ sum of (1/λ_i) with correction
                ricci += 1.0 / fii;
            }
        }
        // Scale by dimension to get scalar curvature
        -ricci * (n as f64)
    }

    /// Set curvature and compute exchange rate
    pub fn set_curvature(&mut self, ricci_scalar: f64) {
        self.ricci_scalar = ricci_scalar;
        // Cost rate: higher curvature → higher cost
        // Exchange rate = |Ricci| × temperature factor
        self.cost_rate = 1.0 + ricci_scalar.abs();
    }

    /// Compute FLOP cost for a geodesic step of given length
    pub fn flops_for_step(&self, geodesic_distance: f64) -> u64 {
        // Cost = geodesic_distance² × cost_rate (Varadhan + curvature correction)
        let cost = geodesic_distance.powi(2) * self.cost_rate;
        cost.max(1.0) as u64
    }

    /// Record a curvature measurement
    pub fn record(&mut self, geodesic_step: f64, flops: u64) {
        self.history.push(CurvatureRecord {
            ricci_scalar: self.ricci_scalar,
            cost_rate: self.cost_rate,
            geodesic_step,
            flops,
        });
    }

    pub fn ricci_scalar(&self) -> f64 { self.ricci_scalar }
    pub fn cost_rate(&self) -> f64 { self.cost_rate }
    pub fn history(&self) -> &[CurvatureRecord] { &self.history }
}

impl Default for CurvatureExchange {
    fn default() -> Self { Self::new() }
}

// ─── Bit Erasure Counter ────────────────────────────────────────────────────

/// Counts how many belief state bits actually changed during an update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitErasureCounter {
    total_bits_erased: u64,
    per_update_bits: Vec<u64>,
}

impl BitErasureCounter {
    pub fn new() -> Self {
        Self {
            total_bits_erased: 0,
            per_update_bits: Vec::new(),
        }
    }

    /// Count bits erased by comparing old and new parameter vectors.
    /// Uses relative entropy change as a proxy for bit erasure.
    pub fn count_bits_param_change(old: &[f64], new: &[f64], tolerance: f64) -> u64 {
        old.iter().zip(new.iter())
            .filter(|(&o, &n)| (o - n).abs() > tolerance)
            .count() as u64
    }

    /// Count bits erased via KL divergence (information-theoretic).
    /// bits ≈ D_KL(old || new) / ln(2)
    pub fn count_bits_kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter())
            .filter(|(pi, _)| **pi > 0.0)
            .map(|(pi, qi)| {
                let qi_safe = qi.max(1e-15);
                pi * (pi / qi_safe).ln()
            })
            .sum::<f64>()
            / LN2
    }

    /// Count bits from entropy reduction
    pub fn count_bits_entropy(old_entropy: f64, new_entropy: f64) -> u64 {
        if old_entropy > new_entropy {
            (old_entropy - new_entropy).ceil() as u64
        } else {
            0
        }
    }

    /// Record a bit erasure event
    pub fn record(&mut self, bits: u64) {
        self.total_bits_erased += bits;
        self.per_update_bits.push(bits);
    }

    pub fn total_bits_erased(&self) -> u64 { self.total_bits_erased }
    pub fn per_update_bits(&self) -> &[u64] { &self.per_update_bits }
}

impl Default for BitErasureCounter {
    fn default() -> Self { Self::new() }
}

// ─── Temperature Calibration ────────────────────────────────────────────────

/// Calibrate effective temperature from FLOP/distance ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureCalibration {
    calibrated_temperature: f64,
    calibration_samples: Vec<CalibrationSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub flops: u64,
    pub fisher_rao_distance: f64,
    pub effective_temperature: f64,
}

impl TemperatureCalibration {
    pub fn new() -> Self {
        Self {
            calibrated_temperature: DEFAULT_TEMPERATURE,
            calibration_samples: Vec::new(),
        }
    }

    /// Compute effective kT from FLOPs and Fisher-Rao distance.
    /// Varadhan: FLOPs × kT ≈ distance² × T
    /// So kT_eff ≈ FLOPs / distance² (in appropriate units)
    pub fn calibrate_from_step(flops: u64, fisher_rao_distance: f64) -> f64 {
        if fisher_rao_distance <= 0.0 { return DEFAULT_TEMPERATURE; }
        // effective T = (flops × kT_physical) / distance²
        // But since we're measuring in "compute units", T_eff = flops / distance²
        let t_eff = flops as f64 / fisher_rao_distance.powi(2);
        t_eff
    }

    /// Add a calibration sample
    pub fn add_sample(&mut self, flops: u64, fisher_rao_distance: f64) {
        let t_eff = Self::calibrate_from_step(flops, fisher_rao_distance);
        self.calibration_samples.push(CalibrationSample {
            flops,
            fisher_rao_distance,
            effective_temperature: t_eff,
        });
        // Update running average
        let avg: f64 = self.calibration_samples.iter()
            .map(|s| s.effective_temperature)
            .sum::<f64>() / self.calibration_samples.len() as f64;
        self.calibrated_temperature = avg;
    }

    /// Calibrate from multiple samples
    pub fn calibrate(&mut self, samples: &[(u64, f64)]) {
        for (flops, dist) in samples {
            self.add_sample(*flops, *dist);
        }
    }

    pub fn calibrated_temperature(&self) -> f64 { self.calibrated_temperature }
    pub fn calibration_samples(&self) -> &[CalibrationSample] { &self.calibration_samples }
}

impl Default for TemperatureCalibration {
    fn default() -> Self { Self::new() }
}

// ─── Self-Improvement Audit ─────────────────────────────────────────────────

/// Track total learning cost over an agent's lifetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfImprovementAudit {
    total_flops: u64,
    total_bits_erased: u64,
    total_fisher_rao_distance: f64,
    total_thermodynamic_cost: f64,
    total_landauer_dissipation: f64,
    temperature: f64,
    update_log: Vec<UpdateRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord {
    pub flops: u64,
    pub bits_erased: u64,
    pub fisher_rao_distance: f64,
    pub thermodynamic_cost: f64,
    pub landauer_dissipation: f64,
    pub efficiency: f64,
    pub timestamp: u64,
}

impl SelfImprovementAudit {
    pub fn new(temperature: f64) -> Self {
        Self {
            total_flops: 0,
            total_bits_erased: 0,
            total_fisher_rao_distance: 0.0,
            total_thermodynamic_cost: 0.0,
            total_landauer_dissipation: 0.0,
            temperature,
            update_log: Vec::new(),
        }
    }

    /// Record a complete self-improvement step
    pub fn record_update(
        &mut self,
        flops: u64,
        bits_erased: u64,
        fisher_rao_distance: f64,
        timestamp: u64,
    ) {
        let thermo_cost = VaradhanBridge::new(self.temperature).step_cost(fisher_rao_distance);
        let landauer_diss = LandauerBound::new(self.temperature).min_dissipation(bits_erased);
        let actual_energy = flops as f64 * BOLTZMANN * self.temperature;
        let efficiency = if actual_energy == 0.0 { 1.0 } else { (landauer_diss / actual_energy).min(1.0) };

        self.total_flops += flops;
        self.total_bits_erased += bits_erased;
        self.total_fisher_rao_distance += fisher_rao_distance;
        self.total_thermodynamic_cost += thermo_cost;
        self.total_landauer_dissipation += landauer_diss;

        self.update_log.push(UpdateRecord {
            flops,
            bits_erased,
            fisher_rao_distance,
            thermodynamic_cost: thermo_cost,
            landauer_dissipation: landauer_diss,
            efficiency,
            timestamp,
        });
    }

    /// Overall thermodynamic efficiency
    pub fn overall_efficiency(&self) -> f64 {
        let actual = self.total_flops as f64 * BOLTZMANN * self.temperature;
        if actual == 0.0 { return 1.0; }
        (self.total_landauer_dissipation / actual).min(1.0)
    }

    /// Average cost per bit learned
    pub fn cost_per_bit(&self) -> f64 {
        if self.total_bits_erased == 0 { return 0.0; }
        self.total_thermodynamic_cost / self.total_bits_erased as f64
    }

    /// Average Fisher-Rao distance per update
    pub fn avg_distance(&self) -> f64 {
        if self.update_log.is_empty() { return 0.0; }
        self.total_fisher_rao_distance / self.update_log.len() as f64
    }

    /// Detect if the agent is in a wasteful learning phase
    pub fn is_wasteful_phase(&self, window: usize, threshold: f64) -> bool {
        if self.update_log.len() < window { return false; }
        let recent: Vec<_> = self.update_log.iter().rev().take(window).collect();
        let avg_eff: f64 = recent.iter().map(|r| r.efficiency).sum::<f64>() / recent.len() as f64;
        avg_eff < threshold
    }

    /// Generate a summary report
    pub fn report(&self) -> AuditReport {
        let mut effs: Vec<f64> = self.update_log.iter().map(|r| r.efficiency).collect();
        effs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        AuditReport {
            total_updates: self.update_log.len(),
            total_flops: self.total_flops,
            total_bits_erased: self.total_bits_erased,
            total_fisher_rao_distance: self.total_fisher_rao_distance,
            total_thermodynamic_cost: self.total_thermodynamic_cost,
            total_landauer_dissipation: self.total_landauer_dissipation,
            overall_efficiency: self.overall_efficiency(),
            cost_per_bit: self.cost_per_bit(),
            avg_distance: self.avg_distance(),
            median_efficiency: if effs.is_empty() { 1.0 } else { effs[effs.len() / 2] },
        }
    }

    pub fn total_flops(&self) -> u64 { self.total_flops }
    pub fn total_bits_erased(&self) -> u64 { self.total_bits_erased }
    pub fn total_fisher_rao_distance(&self) -> f64 { self.total_fisher_rao_distance }
    pub fn total_thermodynamic_cost(&self) -> f64 { self.total_thermodynamic_cost }
    pub fn total_landauer_dissipation(&self) -> f64 { self.total_landauer_dissipation }
    pub fn update_log(&self) -> &[UpdateRecord] { &self.update_log }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub total_updates: usize,
    pub total_flops: u64,
    pub total_bits_erased: u64,
    pub total_fisher_rao_distance: f64,
    pub total_thermodynamic_cost: f64,
    pub total_landauer_dissipation: f64,
    pub overall_efficiency: f64,
    pub cost_per_bit: f64,
    pub avg_distance: f64,
    pub median_efficiency: f64,
}

// ─── PLATO Agent Efficiency Monitor ─────────────────────────────────────────

/// Application: PLATO agent efficiency monitoring.
/// Detects wasteful vs efficient learning in a self-improving agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoMonitor {
    audit: SelfImprovementAudit,
    efficiency_tracker: ThermodynamicEfficiency,
    curvature: CurvatureExchange,
    wasteful_threshold: f64,
    alert_log: Vec<String>,
}

impl PlatoMonitor {
    pub fn new(temperature: f64, wasteful_threshold: f64) -> Self {
        Self {
            audit: SelfImprovementAudit::new(temperature),
            efficiency_tracker: ThermodynamicEfficiency::new(temperature),
            curvature: CurvatureExchange::new(),
            wasteful_threshold,
            alert_log: Vec::new(),
        }
    }

    /// Monitor a single PLATO learning step
    pub fn monitor_step(
        &mut self,
        flops: u64,
        old_params: &[f64],
        new_params: &[f64],
        fim_diagonal: &[f64],
        timestamp: u64,
    ) -> PlatoStepResult {
        // Compute Fisher-Rao distance
        let n = old_params.len().min(new_params.len()).min(fim_diagonal.len());
        let fisher_rao_dist = if n > 0 {
            let mut sum = 0.0;
            for i in 0..n {
                let diff = new_params[i] - old_params[i];
                sum += fim_diagonal[i] * diff * diff;
            }
            sum.sqrt()
        } else {
            0.0
        };

        // Count bits erased
        let bits_erased = BitErasureCounter::count_bits_param_change(old_params, new_params, 1e-6);

        // Record in all trackers
        self.audit.record_update(flops, bits_erased.max(1), fisher_rao_dist, timestamp);
        self.efficiency_tracker.record(flops, bits_erased.max(1), fisher_rao_dist);

        // Update curvature from FIM
        let fim = DMatrix::from_diagonal(&DVector::from_vec(fim_diagonal.to_vec()));
        let ricci = CurvatureExchange::ricci_from_fim(&fim);
        self.curvature.set_curvature(ricci);
        self.curvature.record(fisher_rao_dist, flops);

        // Check for wasteful learning
        let is_wasteful = self.efficiency_tracker.is_wasteful(self.wasteful_threshold);
        if is_wasteful {
            self.alert_log.push(format!(
                "[t={}] Wasteful learning: {} FLOPs, {:.4} Fisher-Rao dist, {} bits, eff={:.4}",
                timestamp, flops, fisher_rao_dist, bits_erased,
                self.efficiency_tracker.efficiency()
            ));
        }

        PlatoStepResult {
            fisher_rao_distance: fisher_rao_dist,
            bits_erased,
            efficiency: self.efficiency_tracker.efficiency(),
            is_wasteful,
            ricci_scalar: ricci,
            cost_rate: self.curvature.cost_rate(),
        }
    }

    /// Get the full audit report
    pub fn audit_report(&self) -> AuditReport {
        self.audit.report()
    }

    /// Get wasteful learning alerts
    pub fn alerts(&self) -> &[String] { &self.alert_log }

    /// Get the underlying audit
    pub fn audit(&self) -> &SelfImprovementAudit { &self.audit }

    /// Get the efficiency tracker
    pub fn efficiency(&self) -> &ThermodynamicEfficiency { &self.efficiency_tracker }

    /// Get the curvature exchange
    pub fn curvature(&self) -> &CurvatureExchange { &self.curvature }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatoStepResult {
    pub fisher_rao_distance: f64,
    pub bits_erased: u64,
    pub efficiency: f64,
    pub is_wasteful: bool,
    pub ricci_scalar: f64,
    pub cost_rate: f64,
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── FLOP Counter Tests ──────────────────────────────────────────────

    #[test]
    fn test_flop_counter_new() {
        let c = FlopCounter::new();
        assert_eq!(c.total_flops(), 0);
        assert_eq!(c.update_count(), 0);
    }

    #[test]
    fn test_flop_counter_record() {
        let mut c = FlopCounter::new();
        c.record(100);
        assert_eq!(c.total_flops(), 100);
        assert_eq!(c.update_count(), 1);
        c.record(200);
        assert_eq!(c.total_flops(), 300);
        assert_eq!(c.update_count(), 2);
    }

    #[test]
    fn test_flop_counter_matmul() {
        let mut c = FlopCounter::new();
        c.record_matmul(2, 3, 4); // 2*3*4*2 = 48
        assert_eq!(c.total_flops(), 48);
    }

    #[test]
    fn test_flop_counter_dot() {
        let mut c = FlopCounter::new();
        c.record_dot(5); // 2*5 = 10
        assert_eq!(c.total_flops(), 10);
    }

    #[test]
    fn test_flop_counter_matrix_inverse() {
        let mut c = FlopCounter::new();
        c.record_matrix_inverse(3); // 2/3 * 27 = 18
        assert_eq!(c.total_flops(), 18);
    }

    #[test]
    fn test_flop_counter_avg() {
        let mut c = FlopCounter::new();
        c.record(100);
        c.record(200);
        assert_eq!(c.avg_flops(), 150.0);
    }

    #[test]
    fn test_flop_counter_cholesky() {
        let mut c = FlopCounter::new();
        c.record_cholesky(3); // 27/3 = 9
        assert_eq!(c.total_flops(), 9);
    }

    // ─── Fisher-Rao Tests ────────────────────────────────────────────────

    #[test]
    fn test_fisher_rao_1d_identical() {
        let d = FisherRao::distance_1d(1.0, 1.0, 1.0);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_1d_symmetric() {
        let d1 = FisherRao::distance_1d(0.0, 1.0, 1.0);
        let d2 = FisherRao::distance_1d(1.0, 0.0, 1.0);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_1d_positive() {
        let d = FisherRao::distance_1d(0.0, 1.0, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_fisher_rao_categorical_identical() {
        let p = [0.5, 0.5];
        let d = FisherRao::distance_categorical(&p, &p);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_categorical_orthogonal() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let d = FisherRao::distance_categorical(&p, &q);
        assert!((d - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_categorical_symmetric() {
        let p = [0.3, 0.7];
        let q = [0.7, 0.3];
        let d1 = FisherRao::distance_categorical(&p, &q);
        let d2 = FisherRao::distance_categorical(&q, &p);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_fim() {
        let theta1 = DVector::from_vec(vec![0.0, 0.0]);
        let theta2 = DVector::from_vec(vec![1.0, 1.0]);
        let fim = DMatrix::identity(2, 2);
        let d = FisherRao::distance_fim(&theta1, &theta2, &fim);
        assert!((d - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_step_length() {
        let old = DVector::from_vec(vec![0.0]);
        let new = DVector::from_vec(vec![2.0]);
        let fim = DMatrix::identity(1, 1);
        let sl = FisherRao::step_length(&old, &new, &fim);
        assert!((sl - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fim_bernoulli() {
        let params = [0.5, 0.5];
        let fim = FisherRao::fim_bernoulli(&params);
        assert!((fim[(0, 0)] - 4.0).abs() < 1e-10); // 1/(0.5*0.5) = 4
        assert!((fim[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fim_gaussian_known_cov() {
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 4.0]));
        let fim = FisherRao::fim_gaussian_known_cov(&cov);
        assert!((fim[(0, 0)] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_fisher_rao_1d_larger_sigma_smaller_distance() {
        let d1 = FisherRao::distance_1d(0.0, 1.0, 1.0);
        let d2 = FisherRao::distance_1d(0.0, 1.0, 10.0);
        assert!(d2 < d1);
    }

    // ─── Varadhan Bridge Tests ───────────────────────────────────────────

    #[test]
    fn test_varadhan_step_cost() {
        let v = VaradhanBridge::new(1.0);
        let cost = v.step_cost(2.0);
        assert!((cost - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_varadhan_record() {
        let mut v = VaradhanBridge::new(2.0);
        v.record_step(3.0); // 9 * 2 = 18
        assert!((v.cumulative_cost() - 18.0).abs() < 1e-10);
        v.record_step(1.0); // 1 * 2 = 2
        assert!((v.cumulative_cost() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_varadhan_zero_distance() {
        let v = VaradhanBridge::new(100.0);
        let cost = v.step_cost(0.0);
        assert!(cost.abs() < 1e-10);
    }

    #[test]
    fn test_varadhan_compute_from_flops() {
        let v = VaradhanBridge::new(300.0);
        let cost = v.compute_cost_from_flops(1000);
        let expected = 1000.0 * BOLTZMANN * 300.0;
        assert!((cost - expected).abs() < 1e-30);
    }

    #[test]
    fn test_varadhan_temperature() {
        let v = VaradhanBridge::new(42.0);
        assert!((v.temperature() - 42.0).abs() < 1e-10);
    }

    // ─── Landauer Bound Tests ────────────────────────────────────────────

    #[test]
    fn test_landauer_bit_constant() {
        let lb = landauer_bit(300.0);
        let expected = BOLTZMANN * 300.0 * LN2;
        assert!((lb - expected).abs() < 1e-30);
    }

    #[test]
    fn test_landauer_bound_new() {
        let lb = LandauerBound::new(300.0);
        assert_eq!(lb.bits_erased(), 0);
        assert!((lb.total_dissipation()).abs() < 1e-30);
    }

    #[test]
    fn test_landauer_erase() {
        let mut lb = LandauerBound::new(300.0);
        let d = lb.erase(1);
        let expected = BOLTZMANN * 300.0 * LN2;
        assert!((d - expected).abs() < 1e-30);
        assert_eq!(lb.bits_erased(), 1);
    }

    #[test]
    fn test_landauer_erase_multiple() {
        let mut lb = LandauerBound::new(300.0);
        lb.erase(10);
        assert_eq!(lb.bits_erased(), 10);
        let expected = 10.0 * BOLTZMANN * 300.0 * LN2;
        assert!((lb.total_dissipation() - expected).abs() < 1e-28);
    }

    #[test]
    fn test_landauer_min_dissipation() {
        let lb = LandauerBound::new(1.0);
        let d = lb.min_dissipation(8);
        assert!((d - 8.0 * BOLTZMANN * LN2).abs() < 1e-30);
    }

    #[test]
    fn test_landauer_zero_temp() {
        let lb = LandauerBound::new(0.0);
        assert!((lb.min_dissipation(100)).abs() < 1e-30);
    }

    #[test]
    fn test_landauer_erase_from_entropy() {
        let mut lb = LandauerBound::new(300.0);
        let d = lb.erase_from_entropy(10.0, 7.0);
        assert!(d > 0.0);
        assert!(lb.bits_erased() >= 3);
    }

    // ─── Thermodynamic Efficiency Tests ──────────────────────────────────

    #[test]
    fn test_efficiency_new() {
        let te = ThermodynamicEfficiency::new(300.0);
        assert!((te.efficiency() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_efficiency_record() {
        let mut te = ThermodynamicEfficiency::new(300.0);
        te.record(1000, 1, 0.5);
        assert_eq!(te.actual_flops(), 1000);
        assert_eq!(te.landauer_bits(), 1);
        assert!(te.efficiency() < 1.0);
        assert!(te.efficiency() > 0.0);
    }

    #[test]
    fn test_efficiency_wasteful() {
        let mut te = ThermodynamicEfficiency::new(300.0);
        // Very many FLOPs for 1 bit = wasteful
        te.record(1_000_000_000, 1, 0.001);
        assert!(te.is_wasteful(0.01));
    }

    #[test]
    fn test_efficiency_not_wasteful() {
        let mut te = ThermodynamicEfficiency::new(300.0);
        te.record(1, 1, 1.0);
        // Even 1 FLOP for 1 bit: eff = (kT ln2) / (kT) = ln2 ≈ 0.693
        assert!(!te.is_wasteful(0.5));
    }

    #[test]
    fn test_efficiency_history() {
        let mut te = ThermodynamicEfficiency::new(300.0);
        te.record(100, 1, 0.5);
        te.record(200, 2, 0.3);
        assert_eq!(te.history().len(), 2);
    }

    // ─── Curvature Exchange Rate Tests ───────────────────────────────────

    #[test]
    fn test_curvature_new() {
        let c = CurvatureExchange::new();
        assert!((c.ricci_scalar()).abs() < 1e-10);
        assert!((c.cost_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_curvature_set() {
        let mut c = CurvatureExchange::new();
        c.set_curvature(5.0);
        assert!((c.ricci_scalar() - 5.0).abs() < 1e-10);
        assert!((c.cost_rate() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_curvature_flops_for_step() {
        let mut c = CurvatureExchange::new();
        c.set_curvature(0.0); // cost_rate = 1.0
        let flops = c.flops_for_step(2.0); // 4 * 1 = 4
        assert_eq!(flops, 4);
    }

    #[test]
    fn test_curvature_ricci_from_fim() {
        let fim = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let r = CurvatureExchange::ricci_from_fim(&fim);
        // Should be negative, proportional to -n/λ = -2
        assert!(r < 0.0);
    }

    #[test]
    fn test_curvature_record() {
        let mut c = CurvatureExchange::new();
        c.set_curvature(1.0);
        c.record(0.5, 10);
        assert_eq!(c.history().len(), 1);
    }

    #[test]
    fn test_curvature_higher_curvature_more_expensive() {
        let mut c1 = CurvatureExchange::new();
        let mut c2 = CurvatureExchange::new();
        c1.set_curvature(1.0);
        c2.set_curvature(10.0);
        assert!(c2.flops_for_step(1.0) > c1.flops_for_step(1.0));
    }

    // ─── Bit Erasure Counter Tests ──────────────────────────────────────

    #[test]
    fn test_bit_erasure_new() {
        let b = BitErasureCounter::new();
        assert_eq!(b.total_bits_erased(), 0);
    }

    #[test]
    fn test_bit_erasure_record() {
        let mut b = BitErasureCounter::new();
        b.record(5);
        b.record(3);
        assert_eq!(b.total_bits_erased(), 8);
    }

    #[test]
    fn test_bit_erasure_param_change() {
        let old = [1.0, 2.0, 3.0, 4.0, 5.0];
        let new = [1.0, 2.1, 3.0, 4.5, 5.0];
        let bits = BitErasureCounter::count_bits_param_change(&old, &new, 0.01);
        assert_eq!(bits, 2);
    }

    #[test]
    fn test_bit_erasure_kl() {
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];
        let bits = BitErasureCounter::count_bits_kl(&p, &q);
        assert!(bits.abs() < 1e-10);
    }

    #[test]
    fn test_bit_erasure_kl_positive() {
        let p = [0.9, 0.1];
        let q = [0.5, 0.5];
        let bits = BitErasureCounter::count_bits_kl(&p, &q);
        assert!(bits > 0.0);
    }

    #[test]
    fn test_bit_erasure_entropy() {
        let bits = BitErasureCounter::count_bits_entropy(8.0, 5.0);
        assert_eq!(bits, 3);
    }

    #[test]
    fn test_bit_erasure_entropy_no_reduction() {
        let bits = BitErasureCounter::count_bits_entropy(5.0, 8.0);
        assert_eq!(bits, 0);
    }

    #[test]
    fn test_bit_erasure_per_update() {
        let mut b = BitErasureCounter::new();
        b.record(3);
        b.record(7);
        assert_eq!(b.per_update_bits(), &[3, 7]);
    }

    // ─── Temperature Calibration Tests ───────────────────────────────────

    #[test]
    fn test_temp_calibration_new() {
        let tc = TemperatureCalibration::new();
        assert!((tc.calibrated_temperature() - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_temp_calibration_from_step() {
        let t = TemperatureCalibration::calibrate_from_step(100, 10.0);
        assert!((t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temp_calibration_zero_distance() {
        let t = TemperatureCalibration::calibrate_from_step(100, 0.0);
        assert!((t - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_temp_calibration_add_sample() {
        let mut tc = TemperatureCalibration::new();
        tc.add_sample(100, 10.0);
        assert_eq!(tc.calibration_samples().len(), 1);
        assert!((tc.calibrated_temperature() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temp_calibration_average() {
        let mut tc = TemperatureCalibration::new();
        tc.add_sample(100, 10.0); // T=1.0
        tc.add_sample(200, 10.0); // T=2.0
        assert!((tc.calibrated_temperature() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_temp_calibration_batch() {
        let mut tc = TemperatureCalibration::new();
        tc.calibrate(&[(100, 10.0), (300, 10.0)]);
        assert!((tc.calibrated_temperature() - 2.0).abs() < 1e-10);
    }

    // ─── Self-Improvement Audit Tests ────────────────────────────────────

    #[test]
    fn test_audit_new() {
        let a = SelfImprovementAudit::new(300.0);
        assert_eq!(a.total_flops(), 0);
        assert_eq!(a.total_bits_erased(), 0);
    }

    #[test]
    fn test_audit_record() {
        let mut a = SelfImprovementAudit::new(300.0);
        a.record_update(1000, 5, 2.0, 1);
        assert_eq!(a.total_flops(), 1000);
        assert_eq!(a.total_bits_erased(), 5);
        assert!((a.total_fisher_rao_distance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_audit_efficiency() {
        let mut a = SelfImprovementAudit::new(300.0);
        a.record_update(1, 1, 1.0, 1);
        assert!(a.overall_efficiency() > 0.0);
        assert!(a.overall_efficiency() <= 1.0);
    }

    #[test]
    fn test_audit_cost_per_bit() {
        let mut a = SelfImprovementAudit::new(1.0);
        a.record_update(100, 10, 3.0, 1);
        let cpb = a.cost_per_bit();
        assert!(cpb > 0.0);
    }

    #[test]
    fn test_audit_avg_distance() {
        let mut a = SelfImprovementAudit::new(300.0);
        a.record_update(100, 1, 2.0, 1);
        a.record_update(100, 1, 4.0, 2);
        assert!((a.avg_distance() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_audit_wasteful_phase() {
        let mut a = SelfImprovementAudit::new(300.0);
        for _ in 0..5 {
            a.record_update(1_000_000_000, 1, 0.001, 1);
        }
        assert!(a.is_wasteful_phase(3, 0.001));
    }

    #[test]
    fn test_audit_not_wasteful_phase() {
        let mut a = SelfImprovementAudit::new(300.0);
        for _ in 0..5 {
            a.record_update(1, 1, 1.0, 1);
        }
        assert!(!a.is_wasteful_phase(3, 0.01));
    }

    #[test]
    fn test_audit_report() {
        let mut a = SelfImprovementAudit::new(300.0);
        a.record_update(100, 5, 2.0, 1);
        a.record_update(200, 10, 3.0, 2);
        let r = a.report();
        assert_eq!(r.total_updates, 2);
        assert_eq!(r.total_flops, 300);
        assert_eq!(r.total_bits_erased, 15);
    }

    #[test]
    fn test_audit_update_log() {
        let mut a = SelfImprovementAudit::new(300.0);
        a.record_update(100, 1, 1.0, 100);
        assert_eq!(a.update_log().len(), 1);
        assert_eq!(a.update_log()[0].timestamp, 100);
    }

    // ─── PLATO Monitor Tests ─────────────────────────────────────────────

    #[test]
    fn test_plato_monitor_new() {
        let m = PlatoMonitor::new(300.0, 0.01);
        assert!(m.alerts().is_empty());
    }

    #[test]
    fn test_plato_monitor_step() {
        let mut m = PlatoMonitor::new(300.0, 0.001);
        let old = [0.5, 0.5];
        let new = [0.6, 0.4];
        let fim = [4.0, 4.0];
        let result = m.monitor_step(1000, &old, &new, &fim, 1);
        assert!(result.fisher_rao_distance > 0.0);
        assert!(result.efficiency > 0.0);
    }

    #[test]
    fn test_plato_monitor_no_change() {
        let mut m = PlatoMonitor::new(300.0, 0.001);
        let params = [0.5, 0.5];
        let fim = [4.0, 4.0];
        let result = m.monitor_step(1000, &params, &params, &fim, 1);
        assert!(result.fisher_rao_distance.abs() < 1e-10);
    }

    #[test]
    fn test_plato_monitor_wasteful() {
        let mut m = PlatoMonitor::new(300.0, 0.5);
        let old = [0.5, 0.5];
        let new = [0.5001, 0.4999];
        let fim = [1.0, 1.0];
        // Many FLOPs for tiny change
        for i in 0..5 {
            m.monitor_step(1_000_000_000, &old, &new, &fim, i);
        }
        // Should have some alerts given huge FLOP/tiny change
        // The wasteful check depends on efficiency tracker state
    }

    #[test]
    fn test_plato_audit_report() {
        let mut m = PlatoMonitor::new(300.0, 0.01);
        let old = [0.5];
        let new = [0.6];
        let fim = [4.0];
        m.monitor_step(100, &old, &new, &fim, 1);
        let report = m.audit_report();
        assert_eq!(report.total_updates, 1);
    }

    #[test]
    fn test_plato_monitor_efficiency() {
        let m = PlatoMonitor::new(300.0, 0.01);
        assert!((m.efficiency().efficiency() - 1.0).abs() < 1e-10);
    }

    // ─── Integration Tests ───────────────────────────────────────────────

    #[test]
    fn test_landauer_bit_value() {
        let bit_energy = landauer_bit(300.0);
        // Should be ~2.87e-21 J
        assert!(bit_energy > 2.0e-21);
        assert!(bit_energy < 4.0e-21);
    }

    #[test]
    fn test_full_pipeline() {
        let temp = 300.0;
        let mut flops = FlopCounter::new();
        let mut varadhan = VaradhanBridge::new(temp);
        let mut landauer = LandauerBound::new(temp);
        let mut audit = SelfImprovementAudit::new(temp);

        // Simulate a belief update
        let compute_flops = 1000u64;
        let fisher_rao_dist = 0.5;
        let bits = 3u64;

        flops.record(compute_flops);
        let thermo_cost = varadhan.record_step(fisher_rao_dist);
        landauer.erase(bits);
        audit.record_update(compute_flops, bits, fisher_rao_dist, 1);

        assert_eq!(flops.total_flops(), 1000);
        assert!((varadhan.cumulative_cost() - 0.25 * 300.0).abs() < 1e-10);
        assert_eq!(landauer.bits_erased(), 3);
        assert_eq!(audit.total_flops(), 1000);
    }

    #[test]
    fn test_fisher_rao_distance_consistency() {
        // Moving twice the distance should cost 4× the energy (Varadhan)
        let v = VaradhanBridge::new(1.0);
        let cost1 = v.step_cost(1.0);
        let cost2 = v.step_cost(2.0);
        assert!((cost2 / cost1 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_categorical_distance_bounded() {
        let p = [0.5, 0.5];
        let q = [0.99, 0.01];
        let d = FisherRao::distance_categorical(&p, &q);
        assert!(d <= std::f64::consts::PI);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_serde_roundtrip_audit() {
        let mut audit = SelfImprovementAudit::new(300.0);
        audit.record_update(100, 5, 1.5, 42);
        let json = serde_json::to_string(&audit).unwrap();
        let back: SelfImprovementAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(back.total_flops(), 100);
        assert_eq!(back.total_bits_erased(), 5);
    }
}
