//! Neighbourhood estimation kernels: nearest neighbour, inverse distance and
//! ordinary (point or block) kriging.
//!
//! The caller (Python, `minecore.estimation.kriging`) finds candidate samples
//! with a KD-tree in anisotropy-normalised space and passes them here sorted
//! by that distance. This module applies the selection rules (exclusion of a
//! hole, maximum per hole, maximum count, minimum count) and solves each
//! target. Targets run in parallel on all cores; results do not depend on the
//! number of threads.
//!
//! Conventions match `minecore/estimation/variogram.py`: practical ranges,
//! `gamma = nugget*(h > 0) + sum(sill_i * model_i(|A_i h|))` where the rows of
//! `A_i` are the ellipsoid axes divided by their ranges, and
//! `covariance = total_sill - gamma`.

#![allow(clippy::many_single_char_names)] // kriging notation: a, b, c, m, n, p, r, s, t, v, w

use std::thread;

/// Variogram structure shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// `1.5r - 0.5r^3` for `r <= 1`, then 1.
    Spherical,
    /// `1 - exp(-3r)`.
    Exponential,
    /// `1 - exp(-3r^2)`.
    Gaussian,
}

/// One nested structure: shape, sill and anisotropy transform.
#[derive(Debug, Clone, Copy)]
pub struct Structure {
    /// Shape of the structure.
    pub shape: Shape,
    /// Partial sill.
    pub sill: f64,
    /// Rows: axis unit vectors divided by the ranges (lag -> reduced lag).
    pub transform: [[f64; 3]; 3],
}

/// Nugget plus nested structures.
#[derive(Debug, Clone)]
pub struct Variogram {
    /// Nugget effect.
    pub nugget: f64,
    /// Nested structures.
    pub structures: Vec<Structure>,
}

impl Variogram {
    /// Total sill (nugget + partial sills).
    pub fn total_sill(&self) -> f64 {
        self.nugget + self.structures.iter().map(|s| s.sill).sum::<f64>()
    }

    /// Semivariogram of the lag vector `h`.
    pub fn gamma(&self, h: [f64; 3]) -> f64 {
        let zero = h[0] == 0.0 && h[1] == 0.0 && h[2] == 0.0;
        let mut g = if zero { 0.0 } else { self.nugget };
        for s in &self.structures {
            let mut r2 = 0.0;
            for row in &s.transform {
                let v = row[0] * h[0] + row[1] * h[1] + row[2] * h[2];
                r2 += v * v;
            }
            let r = r2.sqrt();
            g += s.sill
                * match s.shape {
                    Shape::Spherical => {
                        let rc = r.min(1.0);
                        1.5 * rc - 0.5 * rc * rc * rc
                    }
                    Shape::Exponential => 1.0 - (-3.0 * r).exp(),
                    Shape::Gaussian => 1.0 - (-3.0 * r2).exp(),
                };
        }
        g
    }

    /// Covariance `total_sill - gamma`.
    pub fn covariance(&self, h: [f64; 3]) -> f64 {
        self.total_sill() - self.gamma(h)
    }
}

/// Estimation method.
#[derive(Debug, Clone)]
pub enum Method {
    /// Value of the closest selected sample.
    Nearest,
    /// Inverse distance (anisotropic distance) to a power.
    InverseDistance {
        /// Power of the distance.
        power: f64,
    },
    /// Ordinary kriging; `offsets` discretise the block (one zero offset = point).
    OrdinaryKriging {
        /// Variogram model.
        variogram: Variogram,
        /// Discretisation points relative to the block centre.
        offsets: Vec<[f64; 3]>,
    },
    /// Simple kriging with a known stationary `mean`.
    SimpleKriging {
        /// Variogram model.
        variogram: Variogram,
        /// Discretisation points relative to the block centre.
        offsets: Vec<[f64; 3]>,
        /// Known mean of the attribute.
        mean: f64,
    },
}

/// Neighbour selection rules.
#[derive(Debug, Clone, Copy)]
pub struct Selection {
    /// Fewer selected samples than this leaves the target unestimated.
    pub min_samples: usize,
    /// At most this many samples are used.
    pub max_samples: usize,
    /// At most this many samples per hole (0 = no limit).
    pub max_per_hole: usize,
}

/// Inputs for one estimation run. `candidates` and `distances` are `m x k`
/// row-major; a candidate equal to `u32::MAX` (or an infinite distance) ends
/// the row.
pub struct Problem<'a> {
    /// Sample coordinates.
    pub samples: &'a [[f64; 3]],
    /// Sample values.
    pub values: &'a [f64],
    /// Hole number of each sample.
    pub holes: Option<&'a [i64]>,
    /// Target coordinates (block centres or points).
    pub targets: &'a [[f64; 3]],
    /// Hole to leave out for each target (cross-validation).
    pub exclude_hole: Option<&'a [i64]>,
    /// Candidate sample indices, sorted by distance within each row.
    pub candidates: &'a [u32],
    /// Anisotropic distances of the candidates.
    pub distances: &'a [f64],
    /// Candidates per target.
    pub k: usize,
}

/// Per-target results; `NaN` value where the target was not estimated.
#[derive(Debug, Clone, Default)]
pub struct Estimates {
    /// Estimated value.
    pub value: Vec<f64>,
    /// Kriging variance (NaN for other methods).
    pub variance: Vec<f64>,
    /// Number of samples used.
    pub used: Vec<i64>,
    /// Mean anisotropic distance of the samples used.
    pub mean_distance: Vec<f64>,
    /// Slope of regression of the true block value on its estimate,
    /// `Cov(Z*, Z) / Var(Z*)` (NaN for non-kriging methods).
    pub slope: Vec<f64>,
    /// Kriging efficiency `(BV - KV) / BV` with `BV` the block variance
    /// (NaN for non-kriging methods).
    pub efficiency: Vec<f64>,
}

/// Per-target kriging output of [`estimate_one`].
struct One {
    value: f64,
    variance: f64,
    slope: f64,
    efficiency: f64,
}

impl One {
    fn plain(value: f64) -> Self {
        Self {
            value,
            variance: f64::NAN,
            slope: f64::NAN,
            efficiency: f64::NAN,
        }
    }
}

fn select(p: &Problem<'_>, t: usize, rules: Selection, out: &mut Vec<(usize, f64)>) {
    out.clear();
    let mut per_hole: Vec<(i64, usize)> = Vec::new();
    for c in 0..p.k {
        let idx = p.candidates[t * p.k + c];
        let d = p.distances[t * p.k + c];
        if idx == u32::MAX || !d.is_finite() {
            break;
        }
        let i = idx as usize;
        if let (Some(holes), Some(ex)) = (p.holes, p.exclude_hole) {
            if holes[i] == ex[t] {
                continue;
            }
        }
        if rules.max_per_hole > 0 {
            if let Some(holes) = p.holes {
                let h = holes[i];
                match per_hole.iter_mut().find(|(hh, _)| *hh == h) {
                    Some((_, n)) if *n >= rules.max_per_hole => continue,
                    Some((_, n)) => *n += 1,
                    None => per_hole.push((h, 1)),
                }
            }
        }
        out.push((i, d));
        if out.len() == rules.max_samples {
            break;
        }
    }
}

/// Solves `a x = b` in place by Gaussian elimination with partial pivoting.
/// Returns false for a singular matrix.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r * n + col].abs() > a[pivot * n + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * n + col].abs() < 1e-12 {
            return false;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            b.swap(col, pivot);
        }
        for r in col + 1..n {
            let f = a[r * n + col] / a[col * n + col];
            if f != 0.0 {
                for c in col..n {
                    a[r * n + c] -= f * a[col * n + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r * n + c] * b[c];
        }
        b[r] = s / a[r * n + r];
    }
    true
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Average covariance between the discretisation points of a block.
fn block_covariance(vg: &Variogram, offsets: &[[f64; 3]]) -> f64 {
    let mut s = 0.0;
    for a in offsets {
        for b in offsets {
            s += vg.covariance(sub(*a, *b));
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = (offsets.len() * offsets.len()) as f64;
    s / n
}

/// Mean covariance between sample `i` and the discretised target `t`.
#[allow(clippy::cast_precision_loss)]
fn point_block_covariance(
    p: &Problem<'_>,
    vg: &Variogram,
    offsets: &[[f64; 3]],
    i: usize,
    t: usize,
) -> f64 {
    let mut s = 0.0;
    for o in offsets {
        let x = [
            p.targets[t][0] + o[0],
            p.targets[t][1] + o[1],
            p.targets[t][2] + o[2],
        ];
        s += vg.covariance(sub(p.samples[i], x));
    }
    s / offsets.len() as f64
}

/// Kriging statistics from the weights' `w'c`, the Lagrange multiplier `mu`
/// (0 for simple kriging) and the block variance `cbb`.
fn kriging_stats(value: f64, cbb: f64, wc: f64, mu: f64) -> One {
    // OK system: C w + mu 1 = c, so Var(Z*) = w'C w = w'c - mu and
    // Cov(Z*, Z_v) = w'c; KV = C_vv - w'c - mu.
    let variance = cbb - wc - mu;
    let var_est = wc - mu;
    One {
        value,
        variance,
        slope: if var_est.abs() > 1e-300 {
            wc / var_est
        } else {
            f64::NAN
        },
        efficiency: if cbb > 0.0 {
            (cbb - variance) / cbb
        } else {
            f64::NAN
        },
    }
}

fn estimate_one(
    p: &Problem<'_>,
    t: usize,
    method: &Method,
    cbb: f64,
    chosen: &[(usize, f64)],
    scratch: &mut (Vec<f64>, Vec<f64>),
) -> One {
    match method {
        Method::Nearest => One::plain(p.values[chosen[0].0]),
        Method::InverseDistance { power } => {
            if chosen[0].1 < 1e-12 {
                return One::plain(p.values[chosen[0].0]);
            }
            let (mut num, mut den) = (0.0, 0.0);
            for &(i, d) in chosen {
                let w = 1.0 / d.powf(*power);
                num += w * p.values[i];
                den += w;
            }
            One::plain(num / den)
        }
        Method::OrdinaryKriging { variogram, offsets } => {
            let n = chosen.len();
            let m = n + 1;
            let (a, b) = scratch;
            a.clear();
            a.resize(m * m, 1.0);
            b.clear();
            b.resize(m, 1.0);
            a[n * m + n] = 0.0;
            for (r, &(i, _)) in chosen.iter().enumerate() {
                for (c, &(j, _)) in chosen.iter().enumerate().skip(r) {
                    let cov = variogram.covariance(sub(p.samples[i], p.samples[j]));
                    a[r * m + c] = cov;
                    a[c * m + r] = cov;
                }
                b[r] = point_block_covariance(p, variogram, offsets, i, t);
            }
            let rhs: Vec<f64> = b[..n].to_vec();
            if !solve(a, b, m) {
                return One::plain(f64::NAN);
            }
            let mut value = 0.0;
            let mut wc = 0.0;
            for (r, &(i, _)) in chosen.iter().enumerate() {
                value += b[r] * p.values[i];
                wc += b[r] * rhs[r];
            }
            kriging_stats(value, cbb, wc, b[n])
        }
        Method::SimpleKriging {
            variogram,
            offsets,
            mean,
        } => {
            let n = chosen.len();
            let (a, b) = scratch;
            a.clear();
            a.resize(n * n, 0.0);
            b.clear();
            b.resize(n, 0.0);
            for (r, &(i, _)) in chosen.iter().enumerate() {
                for (c, &(j, _)) in chosen.iter().enumerate().skip(r) {
                    let cov = variogram.covariance(sub(p.samples[i], p.samples[j]));
                    a[r * n + c] = cov;
                    a[c * n + r] = cov;
                }
                b[r] = point_block_covariance(p, variogram, offsets, i, t);
            }
            let rhs: Vec<f64> = b.clone();
            if !solve(a, b, n) {
                return One::plain(f64::NAN);
            }
            let mut value = *mean;
            let mut wc = 0.0;
            for (r, &(i, _)) in chosen.iter().enumerate() {
                value += b[r] * (p.values[i] - mean);
                wc += b[r] * rhs[r];
            }
            kriging_stats(value, cbb, wc, 0.0)
        }
    }
}

/// Estimates every target. Runs on all available cores.
pub fn estimate(p: &Problem<'_>, method: &Method, rules: Selection) -> Estimates {
    let m = p.targets.len();
    let cbb = match method {
        Method::OrdinaryKriging { variogram, offsets }
        | Method::SimpleKriging {
            variogram, offsets, ..
        } => block_covariance(variogram, offsets),
        _ => f64::NAN,
    };
    let threads = thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let chunk = m.div_ceil(threads).max(1);
    let mut out = Estimates {
        value: vec![f64::NAN; m],
        variance: vec![f64::NAN; m],
        used: vec![0; m],
        mean_distance: vec![f64::NAN; m],
        slope: vec![f64::NAN; m],
        efficiency: vec![f64::NAN; m],
    };
    thread::scope(|scope| {
        let parts = out
            .value
            .chunks_mut(chunk)
            .zip(out.variance.chunks_mut(chunk))
            .zip(out.used.chunks_mut(chunk))
            .zip(out.mean_distance.chunks_mut(chunk))
            .zip(out.slope.chunks_mut(chunk))
            .zip(out.efficiency.chunks_mut(chunk))
            .enumerate();
        for (part, (((((value, variance), used), mean_distance), slope), efficiency)) in parts {
            scope.spawn(move || {
                let mut chosen = Vec::new();
                let mut scratch = (Vec::new(), Vec::new());
                for local in 0..value.len() {
                    let t = part * chunk + local;
                    select(p, t, rules, &mut chosen);
                    if chosen.is_empty() || chosen.len() < rules.min_samples {
                        continue;
                    }
                    let one = estimate_one(p, t, method, cbb, &chosen, &mut scratch);
                    value[local] = one.value;
                    variance[local] = one.variance;
                    slope[local] = one.slope;
                    efficiency[local] = one.efficiency;
                    used[local] = i64::try_from(chosen.len()).unwrap_or(i64::MAX);
                    #[allow(clippy::cast_precision_loss)]
                    let n = chosen.len() as f64;
                    mean_distance[local] = chosen.iter().map(|c| c.1).sum::<f64>() / n;
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso(range: f64) -> [[f64; 3]; 3] {
        [
            [1.0 / range, 0.0, 0.0],
            [0.0, 1.0 / range, 0.0],
            [0.0, 0.0, 1.0 / range],
        ]
    }

    fn vg() -> Variogram {
        Variogram {
            nugget: 0.0,
            structures: vec![Structure {
                shape: Shape::Spherical,
                sill: 1.0,
                transform: iso(10.0),
            }],
        }
    }

    fn problem<'a>(
        samples: &'a [[f64; 3]],
        values: &'a [f64],
        targets: &'a [[f64; 3]],
        candidates: &'a [u32],
        distances: &'a [f64],
        k: usize,
    ) -> Problem<'a> {
        Problem {
            samples,
            values,
            holes: None,
            targets,
            exclude_hole: None,
            candidates,
            distances,
            k,
        }
    }

    #[test]
    fn two_symmetric_samples_get_equal_weights() {
        // Target midway between two samples: weights 0.5 each, by symmetry.
        let s = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let v = [1.0, 3.0];
        let t = [[2.0, 0.0, 0.0]];
        let p = problem(&s, &v, &t, &[0, 1], &[0.2, 0.2], 2);
        let method = Method::OrdinaryKriging {
            variogram: vg(),
            offsets: vec![[0.0; 3]],
        };
        let r = estimate(
            &p,
            &method,
            Selection {
                min_samples: 1,
                max_samples: 8,
                max_per_hole: 0,
            },
        );
        assert!((r.value[0] - 2.0).abs() < 1e-12);
        // sigma^2 = C(0) - sum w C(x_i, x0) - mu, from the 3x3 system by hand.
        let g = |h: f64| {
            let r = (h / 10.0_f64).min(1.0);
            1.0 - (1.5 * r - 0.5 * r * r * r)
        };
        let (c01, c0t) = (g(4.0), g(2.0));
        let mu = c0t - 0.5 * (1.0 + c01);
        let expected = 1.0 - c0t - mu;
        assert!((r.variance[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn kriging_is_exact_at_a_sample() {
        let s = [[0.0, 0.0, 0.0], [3.0, 1.0, 0.0], [-2.0, 4.0, 1.0]];
        let v = [5.0, 1.0, 2.0];
        let t = [[0.0, 0.0, 0.0]];
        let p = problem(&s, &v, &t, &[0, 1, 2], &[0.0, 0.3, 0.4], 3);
        let method = Method::OrdinaryKriging {
            variogram: vg(),
            offsets: vec![[0.0; 3]],
        };
        let r = estimate(
            &p,
            &method,
            Selection {
                min_samples: 1,
                max_samples: 8,
                max_per_hole: 0,
            },
        );
        assert!((r.value[0] - 5.0).abs() < 1e-9);
        assert!(r.variance[0].abs() < 1e-9);
        // A perfect estimate: slope of regression and efficiency are 1.
        assert!((r.slope[0] - 1.0).abs() < 1e-9);
        assert!((r.efficiency[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn simple_kriging_falls_back_to_the_mean_beyond_the_range() {
        // Samples further than the range carry no covariance with the target:
        // SK returns the known mean, KV = sill, efficiency 0.
        let s = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]];
        let v = [5.0, 1.0];
        let t = [[25.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let cand = [0, 1, 0, 1];
        let dist = [2.5, 2.5, 0.0, 5.0];
        let p = problem(&s, &v, &t, &cand, &dist, 2);
        let method = Method::SimpleKriging {
            variogram: vg(),
            offsets: vec![[0.0; 3]],
            mean: 2.0,
        };
        let rules = Selection {
            min_samples: 1,
            max_samples: 8,
            max_per_hole: 0,
        };
        let r = estimate(&p, &method, rules);
        assert!((r.value[0] - 2.0).abs() < 1e-12);
        assert!((r.variance[0] - 1.0).abs() < 1e-12);
        assert!(r.efficiency[0].abs() < 1e-12);
        // At a sample SK is exact.
        assert!((r.value[1] - 5.0).abs() < 1e-9);
        assert!(r.variance[1].abs() < 1e-9);
    }

    #[test]
    fn ordinary_kriging_slope_matches_its_definition() {
        // Slope = w'c / (w'c - mu): check against an explicit 2-sample solve.
        let s = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
        let v = [1.0, 3.0];
        let t = [[9.0, 0.0, 0.0]];
        let p = problem(&s, &v, &t, &[1, 0], &[0.3, 0.9], 2);
        let method = Method::OrdinaryKriging {
            variogram: vg(),
            offsets: vec![[0.0; 3]],
        };
        let rules = Selection {
            min_samples: 1,
            max_samples: 8,
            max_per_hole: 0,
        };
        let r = estimate(&p, &method, rules);
        let g = vg();
        let c01 = g.covariance([6.0, 0.0, 0.0]);
        let c0t = g.covariance([9.0, 0.0, 0.0]);
        let c1t = g.covariance([3.0, 0.0, 0.0]);
        // [1 c01 1; c01 1 1; 1 1 0] closed form for two samples.
        let w0 = 0.5 + (c0t - c1t) / (2.0 * (1.0 - c01));
        let w1 = 1.0 - w0;
        let mu = c0t - w0 - w1 * c01;
        let wc = w0 * c0t + w1 * c1t;
        assert!((r.value[0] - (w0 * 1.0 + w1 * 3.0)).abs() < 1e-9);
        assert!((r.slope[0] - wc / (wc - mu)).abs() < 1e-9);
        assert!((r.variance[0] - (1.0 - wc - mu)).abs() < 1e-9);
        assert!((r.efficiency[0] - (1.0 - r.variance[0])).abs() < 1e-9);
    }

    #[test]
    fn selection_rules() {
        let s = [[0.0; 3]; 4];
        let v = [1.0, 2.0, 3.0, 4.0];
        let holes = [7, 7, 7, 8];
        let t = [[0.0; 3], [0.0; 3]];
        let cand = [0, 1, 2, 3, 0, 1, 2, 3];
        let dist = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4];
        let ex = [7, 0];
        let p = Problem {
            samples: &s,
            values: &v,
            holes: Some(&holes),
            targets: &t,
            exclude_hole: Some(&ex),
            candidates: &cand,
            distances: &dist,
            k: 4,
        };
        let rules = Selection {
            min_samples: 1,
            max_samples: 3,
            max_per_hole: 2,
        };
        let r = estimate(&p, &Method::Nearest, rules);
        // Target 0 leaves hole 7 out: only sample 3 remains.
        assert_eq!(r.used, vec![1, 3]);
        assert!((r.value[0] - 4.0).abs() < 1e-12);
        // Target 1: two from hole 7 (limit), then hole 8.
        assert!((r.mean_distance[1] - (0.1 + 0.2 + 0.4) / 3.0).abs() < 1e-12);
        let strict = Selection {
            min_samples: 2,
            ..rules
        };
        assert!(estimate(&p, &Method::Nearest, strict).value[0].is_nan());
    }
}
