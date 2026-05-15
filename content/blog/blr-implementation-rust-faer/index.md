---
title: "From Math to Silicon: Implementing BLR+ARD with Rust and faer"
description: "Translating Mathematical Formulas into Production Code"
summary: "The central theme is deceptively simple: never invert a matrix if you can avoid it. This post walks through the production Rust code that makes Bayesian Linear Regression with ARD efficient, numerically safe, and ready for embedded systems. Everything else follows from understanding why this principle matters."
date: 2026-05-15
lastmod: 2026-05-15
draft: false
weight: 20

series: "blr-sensor-calibration"
seriesOrder: 2

categories:
  - "Bayesian Linear Regression"
tags:
  - "rust"
  - "numerical-computing"
  - "bayesian-inference"
  - "blr+ard"
  - "implementation"
  - "matrix-algebra"
  - "faer"
  - "sensor-calibration"

math: true
seriesTitle: "Bayesian Linear Regression"
contributors: []
pinned: false
homepage: false

seo:
  title: "Implementing BLR+ARD in Rust: Matrix Algorithms and Numerical Stability"
  description: "Deep dive into production Rust code: Cholesky decomposition, Mahalanobis distance, EM algorithm, and the critical patterns that ensure numerical correctness."
  canonical: ""
  noindex: false
---

*The first post in this series showed you why Bayesian Linear Regression with Automatic Relevance Determination is the right tool for sensor calibration: principled uncertainty, automatic feature selection, and a closed-form solution that a ten-year-old C compiler could run. This post shows you the part that textbooks skip — what actually happens when you translate those beautiful matrix formulas into production code.*

*The central theme is deceptively simple: **never invert a matrix if you can avoid it.** Everything else follows from understanding why.*

{{< series >}}

---

## Where We Left Off

The [previous post](blr-and-ard.md) derived two core formulas for Bayesian Linear Regression. Given a design matrix $\Phi \in \mathbb{R}^{N \times D}$, a diagonal prior precision $\Lambda = \text{diag}(\alpha_1, \ldots, \alpha_D)$, and noise precision $\beta$, the posterior over weights is:

$$\Sigma_\text{post} = \left(\Lambda + \beta\,\Phi^T\Phi\right)^{-1} \tag{1}$$

$$\boldsymbol{\mu}_\text{post} = \beta\,\Sigma_\text{post}\,\Phi^T\mathbf{y} \tag{2}$$

The superscript $-1$ in equation (1) is the source of almost every numerical problem you will encounter. This post is about replacing it with something better.

We also derived the MacKay EM update rules for learning the hyperparameters $\alpha$ and $\beta$ from data. This post walks through the code in `crates/blr-core/src/gaussian.rs` and `crates/blr-core/src/ard.rs` that makes this work in practice — efficiently, numerically safely, and on a CPU that might be inside a WASM component.

---

## Part I — Linear Algebra for Bayesian Inference

### 1.1 The Matrix Inversion Problem

Let us begin with a concrete diagnosis. Suppose you have a $D \times D$ [symmetric positive-definite](https://gregorygundersen.com/blog/2022/02/27/positive-definite/#quadratic-programming) (SPD) matrix $A$ and you want to compute $A^{-1} b$ for some vector $b$. The naive approach:

```rust
// DON'T DO THIS
let a_inv = invert(a);
let x = a_inv * b;
```

This wastes $O(D^3)$ flops to compute all $D^2$ entries of $A^{-1}$, when the actual answer $x$ lives in $\mathbb{R}^D$. Worse, the error in $x$ scales with the square of $A$'s condition number $\kappa(A)^2$ (see [Appendix A](#appendix-a-the-condition-number-κ-kappa)) — because you first invert with error $\sim \kappa(A) \cdot \epsilon_\text{machine}$, then multiply, amplifying that error again. By the time you examine the diagonal of $\Sigma_\text{post}$ to extract uncertainty estimates, your $\pm 3\sigma$ confidence bands might be meaningless.

The correct way to phrase the problem is: you do not want $A^{-1}$. You want the solution $x$ to the system $Ax = b$. That reframing unlocks a much better path.

| Goal | Naive approach | Better approach |
|------|----------------|-----------------|
| Compute $A^{-1}b$ | Form $A^{-1}$, multiply | Solve $Ax = b$ directly |
| Compute $\log\|A\|$ | Eigendecompose, sum logs | Cholesky $LL^T$, sum $2\log L_{ii}$ |
| Compute $b^T A^{-1} b$ | Form $A^{-1}$, two multiplications | Solve $Az = b$, compute $\|z\|^2$ |
| Cost | $O(D^3)$ + amplified errors | $O(D^3)$ once, then $O(D^2)$ per RHS |

The key insight is that $A$ is **symmetric positive-definite**, and SPD matrices have a uniquely efficient factorization.

---

### 1.2 The Cholesky Decomposition

For any SPD matrix $A$, there exists a unique lower-triangular $L$ with positive diagonal entries such that:

$$A = LL^T$$

This is the **Cholesky decomposition** (or LLT factorization). It costs approximately $\frac{D^3}{3}$ flops — roughly half the cost of LU decomposition, because symmetry means you only process the lower triangle. The algorithm is backward-stable for SPD matrices: the computed $L$ satisfies $(L + \delta L)(L + \delta L)^T = A$ with $\|\delta L\| / \|L\| \leq c \cdot \epsilon_\text{machine} \cdot \kappa(A)$ — a first-power condition number dependence, not the squared dependence you get from inverting (see [Appendix A.2](#a2-why-condition-number-matters-in-matrix-computations) for details).

Why is $\Lambda + \beta\Phi^T\Phi$ always SPD? Let's verify:

1. **Symmetry:** $(\Lambda + \beta\Phi^T\Phi)^T = \Lambda^T + \beta(\Phi^T\Phi)^T = \Lambda + \beta\Phi^T\Phi$. ✓
2. **Positive-definiteness:** For any nonzero $v$, $v^T(\Lambda + \beta\Phi^T\Phi)v = v^T\Lambda v + \beta\|\Phi v\|^2$. The first term is $\sum_j \alpha_j v_j^2 \geq 0$ (since all $\alpha_j > 0$), and is strictly positive in at least one coordinate. So the whole expression is strictly positive. ✓

SPD is not just an algebraic nicety — it is a mathematical guarantee that the Cholesky algorithm will never fail with a "negative pivot" error, and that the factorization is unique. Whenever you see the posterior precision matrix blow up in a poorly conditioned problem, the error from Cholesky tells you exactly what went wrong: `SingularMatrix`.

In faer, the Cholesky factorization of a matrix `a: Mat<f64>` is simply:

```rust
let llt = a.llt(Side::Lower)?;
```

The `.llt()` call computes $L$ and bundles it into a solver object. From that point forward, solving $Ax = b$ is:

```rust
let x = llt.solve(b.as_ref());
```

This performs two triangular solves — a forward substitution through $L$ and a back substitution through $L^T$ — at cost $O(D^2)$ each. The factorization is amortized across multiple right-hand sides. In BLR, this is exactly what we exploit: the same Cholesky factor solves for both $\Sigma$ and $\boldsymbol{\mu}$ in one shot.

---

### 1.3 Computing the Mahalanobis Distance Without Inverting

The first place Cholesky appears in `gaussian.rs` is in `log_pdf` — the log-probability density of a multivariate Gaussian. The formula is:

$$\log\mathcal{N}(x;\,\mu,\,\Sigma) = -\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu) - \frac{1}{2}\log|\Sigma| - \frac{D}{2}\log(2\pi)$$

The quadratic form $(x - \mu)^T\Sigma^{-1}(x - \mu)$ is the squared **Mahalanobis distance** — a generalization of the $z$-score that accounts for correlations between dimensions. Computing it by forming $\Sigma^{-1}$ is wasteful; using the Cholesky factor makes it elegant:

```rust
// from crates/blr-core/src/gaussian.rs
pub fn log_pdf(&self, x: &[f64]) -> f64 {
    let d = self.dim;
    let sigma = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);
    let diff  = Mat::<f64>::from_fn(d, 1, |i, _| x[i] - self.mean[i]);

    let llt = sigma
        .llt(Side::Lower)
        .expect("Covariance must be positive-definite for log_pdf");

    // Solve L · Lᵀ · z = diff  →  z = Σ⁻¹ diff (but we never form Σ⁻¹)
    let z = llt.solve(diff.as_ref());

    // ‖z‖² = diff^T Σ⁻¹ diff  (the Mahalanobis distance)
    let quadratic: f64 = (0..d).map(|i| { let v = z[(i, 0)]; v * v }).sum();

    let logdet = cholesky_logdet(&sigma, d).expect("Covariance must be PD");

    -0.5 * quadratic - 0.5 * logdet - (d as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
}
```

Read through line by line:

1. We build the $D \times D$ covariance matrix `sigma` from the flattened row-major `self.cov` array — faer's `Mat::from_fn` is a clean way to do this with explicit index computation.
2. We compute `diff` = $x - \mu$.
3. `.llt()` factors $\Sigma = LL^T$. If the covariance is not positive-definite (perhaps due to accumulated numerical drift), this panics immediately with a diagnostic message.
4. `.solve(diff)` computes $z$ such that $\Sigma z = \text{diff}$ — in other words, $z = \Sigma^{-1}(x - \mu)$.
5. $\|z\|^2 = z^T z = (x - \mu)^T \Sigma^{-1} (x - \mu)$. This is the Mahalanobis distance, obtained **without ever computing $\Sigma^{-1}$**.

The key insight: you wanted a scalar (the Mahalanobis distance). You got it by solving a linear system. The matrix inverse was never needed. This pattern — "replace inversion with a solve" — appears throughout the entire codebase.

The `logdet` term is handled separately by a helper function. We'll look at that next.

---

### 1.4 Log-Determinant via the Cholesky Diagonal

The log-determinant $\log|\Sigma|$ appears in the log-pdf normalization, and critically, in the log marginal likelihood (the EM objective). Computing it numerically requires care: $\det(\Sigma)$ can be astronomically small or large for moderate $D$, causing underflow or overflow before you can take the logarithm.

The Cholesky factorization solves this too. Since $\Sigma = LL^T$:

$$|\Sigma| = |L|^2 = \left(\prod_{j=1}^{D} L_{jj}\right)^2$$

(because the determinant of a triangular matrix is the product of its diagonal, and the factor of 2 comes from $|LL^T| = |L||L^T| = |L|^2$). Taking logarithms:

$$\log|\Sigma| = 2\sum_{j=1}^{D} \log L_{jj}$$

Every $L_{jj} > 0$ by the SPD guarantee, so the logarithms are all finite. And since we are summing logarithms of numbers that are individually $O(1)$ to $O(10^3)$, there is no overflow or underflow problem.

The implementation in `gaussian.rs` runs a manual Cholesky factorization specifically to accumulate this sum:

```rust
// from crates/blr-core/src/gaussian.rs
pub(crate) fn cholesky_logdet(mat: &Mat<f64>, d: usize) -> Result<f64, BLRError> {
    let mut a = mat.clone();
    for j in 0..d {
        // Compute diagonal pivot: L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)
        let mut diag = a[(j, j)];
        for k in 0..j {
            let l_jk = a[(j, k)];
            diag -= l_jk * l_jk;
        }
        if diag <= 0.0 {
            return Err(BLRError::SingularMatrix);
        }
        let l_jj = diag.sqrt();
        a[(j, j)] = l_jj;                // store L[j,j] in-place

        // Fill the column below the diagonal
        for i in (j + 1)..d {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= a[(i, k)] * a[(j, k)];
            }
            a[(i, j)] = s / l_jj;        // L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]L[j,k]) / L[j,j]
        }
    }
    Ok(2.0 * (0..d).map(|j| a[(j, j)].ln()).sum::<f64>())
}
```

A few things worth noting:

- The factorization is done **in-place** on a clone of the input matrix. The lower triangle is overwritten with $L$; the upper triangle is irrelevant (never read after the computation). No extra allocation needed.
- The check `if diag <= 0.0` is the SPD guard. In a well-posed Bayesian problem, $\Lambda + \beta\Phi^T\Phi$ is always strictly SPD. If you hit `SingularMatrix`, something has gone wrong upstream — either your design matrix has linearly dependent columns, or your hyperparameter initialization is degenerate.
- The function is `pub(crate)` — it's a shared utility used by both `gaussian.rs` (in `log_pdf`) and `ard.rs` (in the E-step to compute the log-evidence denominator).

One natural question: faer's `llt()` already computes the Cholesky factor. Why not just read off the diagonal? The answer is that faer's solver abstraction doesn't expose the raw $L$ matrix in a convenient slice form, and for this particular computation — summing log-diagonals — the explicit loop is clearer and has no performance cost for the matrix sizes we operate at ($D \leq 20$ in the Hall sensor calibration). This is a deliberate tradeoff: clarity over abstraction.

---

### 1.5 The Posterior Update — Two Forms, One Identity

The most important function in `gaussian.rs` is `condition` — the Bayesian posterior update. Given the current Gaussian prior $p(\mathbf{w}) = \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ and new observations $\mathbf{y} = A\mathbf{w} + \boldsymbol{\epsilon}$ with homoscedastic noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_N)$, it computes the exact posterior $p(\mathbf{w} \mid \mathbf{y})$.

There are two algebraically equivalent ways to perform this update, each named for the space in which the key Cholesky factorization lives:

| Form | Cholesky size | Cheaper when |
|------|--------------|--------------|
| **Observation-space** (Gram form) | $N \times N$ | $N < D$ |
| **Parameter-space** (precision form) | $D \times D$ | $D \leq N$ |

For Hall sensor calibration — $D \approx 6$ features, $N \approx 25$–$100$+ observations — $D \ll N$ holds in every realistic scenario. The parameter-space form is strictly cheaper and always selected in production. The Gram form is retained for generality: if you ever build a model with far more basis functions than observations (sparse kernel regression, compressed sensing), the adaptive dispatcher switches to it automatically.

#### The Gram Form (Observation-Space, $N \times N$)

The Gram form works in the $N$-dimensional observation space. Define the **Gram matrix**:

$$G = \sigma^2 I_N + A \Sigma A^T \quad (N \times N)$$

This is the total predictive covariance in observation space — sensor noise $\sigma^2 I_N$ plus the prior uncertainty $\Sigma$ propagated through the linear measurement model $A$. Its Cholesky factor drives both posterior updates:

$$\boldsymbol{\mu}' = \boldsymbol{\mu} + \Sigma A^T G^{-1}(\mathbf{y} - A\boldsymbol{\mu}) \tag{Gram-mean}$$

$$\Sigma' = \Sigma - \Sigma A^T G^{-1} A\Sigma \tag{Gram-cov}$$

$G^{-1}$ never appears explicitly. We introduce $Z$ by solving $GZ = A\Sigma$, so $Z = G^{-1} A\Sigma$ (shape $N \times D$). Both updates then follow from $Z$:

$$\boldsymbol{\mu}' = \boldsymbol{\mu} + Z^T(\mathbf{y} - A\boldsymbol{\mu}), \qquad \Sigma' = \Sigma - \Sigma A^T Z$$

The computational bottleneck is the $N \times N$ Cholesky factorization of $G$, costing $\frac{N^3}{3}$ flops.

#### The Woodbury Identity: The Bridge Between Forms

The two forms compute identical posteriors. The **Woodbury matrix identity** (also called the matrix inversion lemma) is the algebraic proof. In its general form:

$$(P_0 + UCV)^{-1} = P_0^{-1} - P_0^{-1} U \left(C^{-1} + VP_0^{-1}U\right)^{-1} V P_0^{-1}$$

Setting $P_0 = \Sigma^{-1}$, $U = A^T$, $C = \frac{1}{\sigma^2}I$, $V = A$ gives:

$$\underbrace{\left(\Sigma^{-1} + \frac{1}{\sigma^2} A^T A\right)^{-1}}_{\Sigma_\text{post}\ \text{via}\ D \times D\ \text{form}} = \Sigma - \Sigma A^T \underbrace{\left(\sigma^2 I_N + A\Sigma A^T\right)^{-1}}_{G^{-1}\ \text{via}\ N \times N\ \text{form}} A\Sigma$$

The left side is the parameter-space form; the right side is the observation-space (Gram) form. They produce **the same $\Sigma_\text{post}$**. The Woodbury identity is an exact algebraic equality, not an approximation.

(The complete derivation — showing how this emerges from the general Gaussian inference update — is in Appendix A of the companion post [*When Your Sensor Knows What It Doesn't Know*](blr-and-ard.md).)

#### The Precision Form (Parameter-Space, $D \times D$)

The precision form builds the **posterior precision matrix** directly in the $D$-dimensional weight space:

$$P = \Sigma_\text{prior}^{-1} + \frac{1}{\sigma^2} A^T A \quad (D \times D) \tag{posterior precision}$$

To form $P$ we need $\Sigma_\text{prior}^{-1}$. A naive approach would approximate this as $\lambda_0 I_D$ (isotropic). This is wrong: it would make the two forms numerically inconsistent, and the approximation error compounds across sequential updates. The correct approach derives $\Sigma_\text{prior}^{-1}$ **exactly** by Cholesky-factoring `self.cov` — the distribution's current covariance in its role as the prior:

$$\Sigma_\text{prior} = L_0 L_0^T \;\Longrightarrow\; \Sigma_\text{prior}^{-1} = \text{solve}(L_0 L_0^T,\ I_D)$$

With the exact prior precision in hand, the posterior follows in five steps:

1. Build $P = \Sigma_\text{prior}^{-1} + \frac{1}{\sigma^2} A^T A$ ($D \times D$)
2. Cholesky-factor $P = LL^T$
3. Solve $P X = I_D$ to get $\Sigma_\text{post} = P^{-1}$
4. Form the information vector: $r = \Sigma_\text{prior}^{-1}\boldsymbol{\mu}_\text{prior} + \frac{1}{\sigma^2} A^T \mathbf{y}$ ($D \times 1$)
5. Solve $P \boldsymbol{\mu}_\text{post} = r$ — reuse the Cholesky from step 2

Steps 3 and 5 amortize the single $O(D^3)$ Cholesky from step 2 across two $O(D^2)$ solves — the same pattern as Section 1.2. The prior Cholesky is a separate $O(D^3)$ factorization. All matrix dimensions are $D \times D$; $N$ appears only in the accumulation of $A^T A$ and $A^T \mathbf{y}$, both of which produce $D \times D$ and $D \times 1$ outputs regardless of how large $N$ is.

**Why materializing $\Sigma_\text{post}$ here is justified.** Step 3 solves $PX = I_D$, which formally computes $P^{-1}$. The ARD M-step requires the diagonal of $\Sigma_\text{post}$ for the $\gamma_j$ computation (Section 2.4), and prediction requires the full matrix for epistemic uncertainty at arbitrary test points (Section 3.5). The cost is $O(D^3)$ — equal to the factorization itself — and is fixed in $D$, not in $N$.

#### The Dispatch

```rust
// from crates/blr-core/src/gaussian.rs — condition()

/// Bayesian posterior update: p(w | y) from p(w) = N(μ, Σ) and y = Aw + ε.
///
/// Dispatches to the cheaper form automatically:
/// - n_obs <  d_feat  →  observation-space Gram form  (N×N Cholesky)
/// - n_obs >= d_feat  →  parameter-space precision form (D×D Cholesky)
///
/// For sensor calibration (D ≈ 6–16, N ≈ 25–100+) the precision form is
/// always selected. See the Woodbury identity in blr-and-ard.md Appendix A
/// for the proof of algebraic equivalence.
///
/// `noise_variance` is the homoscedastic scalar σ² (same for all observations).
pub fn condition(
    self,
    a: &[f64],
    n_obs: usize,
    d_feat: usize,
    y: &[f64],
    noise_variance: f64,
) -> Result<Self, BLRError> {
    if n_obs < d_feat {
        self.condition_gram_form(a, n_obs, y, noise_variance)
    } else {
        self.condition_precision_form(a, n_obs, y, noise_variance)
    }
}
```

The tie-break (`n_obs == d_feat` → precision form) is deliberate: it matches the convention used by the ARD loop in `ard.rs`, which always builds the posterior precision in $D$-dimensional parameter space.

**Note on noise model.** The `noise_variance: f64` parameter replaces the earlier `lambda: &[f64]` (a per-observation noise vector). For a single-session Hall sensor calibration the measurement conditions are uniform, so heteroscedastic noise adds complexity without physical justification. The scalar model also simplifies both forms: in the Gram form, $\sigma^2$ is added uniformly to the diagonal of $G$; in the precision form, $\frac{1}{\sigma^2} A^T A$ reduces to a single scalar multiplication rather than a diagonal-weighted sum.

#### Gram Form — Code Walkthrough

```rust
// from crates/blr-core/src/gaussian.rs — condition_gram_form() (internal)

// ── Step 1: Gram matrix G = A Σ Aᵀ + σ²I  (N×N) ─────────────────────────
let a_sigma = /* A · Σ   (N×D)×(D×D) → N×D */;
let mut gram = /* a_sigma · Aᵀ   (N×D)×(D×N) → N×N */;
for i in 0..n_obs {
    gram[(i, i)] += noise_variance;   // add σ² uniformly (homoscedastic)
}

// ── Step 2: Cholesky G = L Lᵀ ────────────────────────────────────────────
let llt_gram = gram.llt(Side::Lower).map_err(|_| BLRError::SingularMatrix)?;

// ── Step 3: Solve G·Z = A·Σ  →  Z = G⁻¹·A·Σ  (N×D) ─────────────────────
// sigma_at = Σ·Aᵀ (D×N);  A·Σ = (Σ·Aᵀ)ᵀ is the RHS for the solve
let sigma_at = /* Σ · Aᵀ  (D×N) */;
let z = llt_gram.solve(sigma_at.as_ref().transpose());   // Z is N×D

// ── Step 4: Mean update  μ' = μ + Zᵀ·(y − A·μ) ───────────────────────────
let residual = /* y − A·μ  (N×1) */;
let delta_mu = /* Zᵀ · residual  (D×1) */;

// ── Step 5: Covariance update  Σ' = Σ − Σ·Aᵀ·Z  (D×D) ───────────────────
faer::linalg::matmul::matmul(
    sigma_new.as_mut(), Accum::Add,
    sigma_at.as_ref(), z.as_ref(),
    -1.0_f64,            // Σ' = Σ + (-1.0) × (Σ Aᵀ · Z)
    Par::Seq,
);
```

Step 3 is the "replace inversion with a solve" pattern from Section 1.1, applied at full generality: solving $GZ = A\Sigma$ gives $Z = G^{-1}A\Sigma$ without materialising $G^{-1}$. Both update equations use only $Z$ and $\Sigma A^T$; the Gram matrix's inverse never appears.

#### Precision Form — Code Walkthrough

```rust
// from crates/blr-core/src/gaussian.rs — condition_precision_form() (internal)

let d = self.dim;
let sigma_prior = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);
let mu_prior    = Mat::<f64>::from_fn(d, 1, |i, _| self.mean[i]);
let a_mat       = Mat::<f64>::from_fn(n_obs, d, |i, j| a[i * d + j]);
let y_mat       = Mat::<f64>::from_fn(n_obs, 1, |i, _| y[i]);

// ── Step 1: Cholesky Σ_prior; derive Σ_prior⁻¹ and Σ_prior⁻¹·μ_prior ────
let llt_prior       = sigma_prior.llt(Side::Lower)
                                 .map_err(|_| BLRError::SingularMatrix)?;
let sigma_prior_inv = llt_prior.solve(Mat::<f64>::identity(d, d).as_ref());
let prior_info_vec  = llt_prior.solve(mu_prior.as_ref());   // Σ_prior⁻¹·μ

// ── Step 2: Build P = Σ_prior⁻¹ + (1/σ²)·AᵀA  (D×D) ─────────────────────
let mut at_a = Mat::<f64>::zeros(d, d);
matmul::matmul(at_a.as_mut(), Accum::Replace,
    a_mat.as_ref().transpose(), a_mat.as_ref(),
    1.0 / noise_variance, Par::Seq);
let prec_post = /* sigma_prior_inv + at_a  (element-wise sum, D×D) */;

// ── Step 3: Cholesky P = L Lᵀ ────────────────────────────────────────────
let llt_post = prec_post.llt(Side::Lower).map_err(|_| BLRError::SingularMatrix)?;

// ── Step 4: Σ_post = P⁻¹  (solve P·X = I_D) ─────────────────────────────
let sigma_post = llt_post.solve(Mat::<f64>::identity(d, d).as_ref());

// ── Step 5: μ_post = Σ_post·(Σ_prior⁻¹·μ_prior + (1/σ²)·Aᵀ·y) ──────────
let mut at_y = Mat::<f64>::zeros(d, 1);
matmul::matmul(at_y.as_mut(), Accum::Replace,
    a_mat.as_ref().transpose(), y_mat.as_ref(),
    1.0 / noise_variance, Par::Seq);
let rhs     = /* prior_info_vec + at_y  (D×1 information vector) */;
let mu_post = llt_post.solve(rhs.as_ref());   // reuse llt_post
```

Two independent Cholesky factorizations — `llt_prior` (from `self.cov`) and `llt_post` (posterior precision) — each amortised across multiple solves. `llt_prior` yields both $\Sigma_\text{prior}^{-1}$ and $\Sigma_\text{prior}^{-1}\boldsymbol{\mu}_\text{prior}$. `llt_post` yields both $\Sigma_\text{post}$ and $\boldsymbol{\mu}_\text{post}$. All Cholesky work is $D \times D$; the dataset size $N$ contributes only the $O(ND^2)$ accumulation of $A^T A$ and $A^T \mathbf{y}$. As $N$ grows, the per-update cost stays $O(D^3)$ — fixed in the number of features, not in the amount of data.

Both forms pass the same analytic unit test (`test_condition_analytic`) against a closed-form 2D Gaussian update, and a dedicated parity test (`test_condition_parity`) confirms agreement within $10^{-10}$ on all matrix entries for inputs where neither form is strongly favoured by the dispatch rule.

---

## Part II — Automatic Relevance Determination via the EM Algorithm

With the linear algebra machinery in place, we can now build the complete BLR+ARD fitting loop. Recall from Part 1 that the model has two levels of unknowns:

- **Level 1 (weights $\mathbf{w}$):** Given $\alpha$ and $\beta$, the posterior is *exact* — computed in one shot via the Cholesky formulas above.
- **Level 2 (hyperparameters $\alpha_j$, $\beta$):** These govern the prior and noise model. We learn them by maximizing the marginal likelihood (evidence) via Expectation-Maximization.

The EM algorithm alternates: fix hyperparameters → compute exact posterior (E-step) → update hyperparameters to maximize evidence (M-step) → repeat.

### 2.1 Configuration and Initialisation

The fitting loop is configured through `ArdConfig`:

```rust
// from crates/blr-core/src/ard.rs
#[derive(Debug, Clone)]
pub struct ArdConfig {
    pub alpha_init: f64,   // Initial ARD precision (same for all features)
    pub beta_init: f64,    // Initial noise precision
    pub max_iter: usize,   // Maximum EM iterations
    pub tol: f64,          // Convergence tolerance (period-2 log-evidence delta)
    pub update_beta: bool, // Whether to update β in the M-step
}

impl Default for ArdConfig {
    fn default() -> Self {
        Self { alpha_init: 1.0, beta_init: 1.0, max_iter: 100, tol: 1e-5, update_beta: true }
    }
}
```

The defaults match the Python reference implementation. Starting with `alpha_init = 1.0` means the prior on each weight is $\mathcal{N}(0, 1)$ — a mild regularisation that neither suppresses nor encourages any feature. The EM loop will sort out which features deserve to survive.

### 2.2 Pre-computing the Sufficient Statistics

Before entering the EM loop, `fit()` pre-computes two quantities that appear in every iteration:

```rust
// from crates/blr-core/src/ard.rs — fit()

// Pre-compute Φᵀ Φ (D×D) and Φᵀ y (D×1) — reused every iteration.
let mut phi_t_phi = Mat::<f64>::zeros(d, d);
matmul::matmul(
    phi_t_phi.as_mut(), Accum::Replace,
    phi_mat.as_ref().transpose(), phi_mat.as_ref(),
    1.0_f64, Par::Seq,
);

let mut phi_t_y = Mat::<f64>::zeros(d, 1);
matmul::matmul(
    phi_t_y.as_mut(), Accum::Replace,
    phi_mat.as_ref().transpose(), y_mat.as_ref(),
    1.0_f64, Par::Seq,
);
```

The matrix $\Phi^T\Phi$ is the **Gram matrix in feature space** — a $D \times D$ summary of how the features covary across all training points. The vector $\Phi^T \mathbf{y}$ is the **sufficient statistic** for the regression problem — a $D$-dimensional vector that captures everything the raw data knows about the weights.

This pre-computation matters. Each E-step builds the posterior precision as $\Lambda + \beta\Phi^T\Phi$. If you recomputed $\Phi^T\Phi$ inside the loop, you'd pay $O(ND^2)$ per iteration. With pre-computation, you pay it once, then each iteration costs only $O(D^2)$ (to add the diagonal $\Lambda$ and factor the result). For $N = 1000$ training points, $D = 10$ features, and 50 EM iterations, this pre-computation reduces the total matrix work by a factor of ~50.

### 2.3 The E-Step: An Exact Posterior in Three Lines

Each iteration begins by computing the exact posterior distribution over weights, given the current hyperparameters:

```rust
// from crates/blr-core/src/ard.rs — fit(), inside the for loop

// ── E-step ────────────────────────────────────────────────────────────────
// σ_inv = diag(α) + β Φᵀ Φ   (posterior precision matrix, D×D)
let mut sigma_inv = Mat::<f64>::from_fn(d, d, |i, j| beta * phi_t_phi[(i, j)]);
for j in 0..d {
    sigma_inv[(j, j)] += alpha[j];   // add per-feature prior precision
}

// Cholesky factor: L Lᵀ = σ_inv
let llt = sigma_inv
    .llt(Side::Lower)
    .map_err(|_| BLRError::SingularMatrix)?;

// Σ = σ_inv⁻¹  (solve σ_inv · X = I, i.e. X = σ_inv⁻¹)
let eye = Mat::<f64>::identity(d, d);
sigma_mat = llt.solve(eye.as_ref());

// μ = β Σ Φᵀ y  (equivalently, solve σ_inv · μ = β Φᵀ y)
let mut rhs = phi_t_y.clone();
for i in 0..d {
    rhs[(i, 0)] *= beta;      // rhs = β Φᵀ y
}
let mu_mat = llt.solve(rhs.as_ref());
for i in 0..d {
    mu_vec[i] = mu_mat[(i, 0)];
}
```

Notice that the same Cholesky factor `llt` is used **twice**: once to compute the posterior covariance ($\Sigma = (\Lambda + \beta\Phi^T\Phi)^{-1}$) and once to compute the posterior mean ($\boldsymbol{\mu} = \beta\Sigma\Phi^T\mathbf{y}$). The $O(D^3)$ factorisation is paid only once per iteration, and the two solves each cost $O(D^2)$.

The solve for $\Sigma$ is written as `llt.solve(I)` — solving $AX = I$ — which is exactly inverting $A$. This is one of the rare cases where we genuinely need the full matrix, not just its action on a specific vector: the M-step's $\gamma_j$ computation requires the diagonal entries $\Sigma_{jj}$, and the prediction code needs $\Sigma$ to compute epistemic uncertainty at arbitrary test points.

If you only needed to compute uncertainty at the training points themselves, you could avoid materialising $\Sigma$ by solving one system per training point. But since we need the full posterior for downstream prediction, forming $\Sigma$ is justified here.

### 2.4 The γ Parameter: The Heart of ARD

After the E-step, we have $\boldsymbol{\mu}$ and $\Sigma$. Before diving into the M-step update formulas, we need to meet the most interpretable quantity in the entire algorithm: $\gamma_j$.

$$\gamma_j = 1 - \alpha_j \Sigma_{jj}$$

This deceptively simple formula is the key to understanding what ARD is actually doing. To see why, consider two limiting cases:

**Case 1: The prior dominates.** Suppose $\alpha_j$ is enormous — say, $\alpha_j = 10^6$. The prior says "weight $j$ should be essentially zero," and unless the data is extraordinarily strong, the posterior obeys. When the prior is very tight, the posterior variance $\Sigma_{jj}$ is close to the prior variance $\alpha_j^{-1}$, so $\alpha_j \Sigma_{jj} \approx 1$, and $\gamma_j \approx 0$. This feature is effectively switched off.

**Case 2: The data dominates.** Suppose $\alpha_j$ is small, and the data strongly constrains weight $j$. The posterior is much tighter than the prior, so $\Sigma_{jj} \ll \alpha_j^{-1}$, meaning $\alpha_j \Sigma_{jj} \ll 1$, and $\gamma_j \approx 1$. The data is fully determining this weight.

So $\gamma_j \in [0, 1]$ measures **how much of the information about weight $j$ comes from the data**. A $\gamma_j$ near 1 means "the data knows something about this feature." A $\gamma_j$ near 0 means "the prior is calling the shots — the feature is irrelevant."

The sum $\gamma = \sum_j \gamma_j$ has an equally beautiful interpretation: it is the **effective number of parameters** being estimated from the data. In classical statistics, fitting $D$ parameters from $N$ data points "uses up" $D$ degrees of freedom. With ARD, features that are being suppressed contribute nearly zero to $\gamma$, so the effective complexity of the model is automatically lower than the raw feature count. The algorithm self-selects its own complexity.

In code, this computation is a single line:

```rust
// γ_j = 1 − α_j Σ_jj   (effective parameters per feature)
let gamma: Vec<f64> = (0..d).map(|j| 1.0 - alpha[j] * sigma_mat[(j, j)]).collect();
```

### 2.5 The M-Step: Learning from Your Own Uncertainty

Given $\boldsymbol{\mu}$, $\Sigma$, and $\gamma$, the M-step updates the hyperparameters. These are the MacKay fixed-point rules derived in Part 1:

$$\alpha_j^{\text{new}} = \frac{\gamma_j}{\mu_j^2} \qquad\qquad \beta^{\text{new}} = \frac{N - \gamma}{\|\mathbf{y} - \Phi\boldsymbol{\mu}\|^2}$$

Let's read the code and the formulas together:

```rust
// from crates/blr-core/src/ard.rs — fit(), M-step

// ── Residuals ─────────────────────────────────────────────────────────────
// Φ μ  (N×1): predicted values using posterior mean weights
let mut phi_mu = Mat::<f64>::zeros(n, 1);
let mu_mat_ref = Mat::<f64>::from_fn(d, 1, |i, _| mu_vec[i]);
matmul::matmul(phi_mu.as_mut(), Accum::Replace, phi_mat.as_ref(), mu_mat_ref.as_ref(), 1.0_f64, Par::Seq);

// ||r||² = ||y - Φ μ||²   (sum of squared residuals)
let residual_sq: f64 = (0..n).map(|i| {
    let r = y[i] - phi_mu[(i, 0)];
    r * r
}).sum();

// ── M-step ────────────────────────────────────────────────────────────────
// α_j = γ_j / (μ_j² + ε),  clamped to ≥ 1e-8
for j in 0..d {
    alpha[j] = (gamma[j] / (mu_vec[j] * mu_vec[j] + 1e-10)).max(1e-8);
}

// β = (N − Σγ_j) / (||r||² + ε),  clamped to ≥ 1e-8
if config.update_beta {
    let gamma_sum: f64 = gamma.iter().sum();
    beta = ((n as f64 - gamma_sum) / (residual_sq + 1e-10)).max(1e-8);
}
```

The $\epsilon = 10^{-10}$ additive term in the denominators prevents division by zero. It's not a regularisation hack — it's a numerical guard for the single case where ARD is doing its job perfectly: when $\mu_j \to 0$ as a feature is being pruned, the update formula $\alpha_j = \gamma_j / \mu_j^2$ would try to send $\alpha_j \to \infty$ (completely suppressing the feature). The clamp `max(1e-8)` achieves the same end — a very large but finite precision — without floating-point infinity propagating into the Cholesky.

The $\alpha_j$ update has a beautiful fixed-point interpretation. At convergence, the update sets $\alpha_j^{\text{new}} = \alpha_j$ (the value does not change). Substituting back:

$$\alpha_j = \frac{\gamma_j}{\mu_j^2} = \frac{1 - \alpha_j \Sigma_{jj}}{\mu_j^2}$$

This says: at the optimum, the precision is set so that the "residual information" in the prior ($1 - \alpha_j\Sigma_{jj}$) exactly equals $\alpha_j \mu_j^2$. Features with large posterior mean carry information proportional to their effect size; features with small posterior mean get their prior tightened until they are suppressed.

The $\beta$ update is equally satisfying. In classical statistics, the unbiased variance estimator is $\hat{\sigma}^2 = \text{RSS} / (N - D)$, where we subtract $D$ degrees of freedom for the $D$ estimated parameters. Here, the formula is $\hat{\sigma}^2 = \text{RSS} / (N - \gamma)$. The effective degrees of freedom is $\gamma$, not $D$ — and since suppressed features contribute $\gamma_j \approx 0$, the noise estimate is automatically corrected for model sparsity. Prune 4 out of 6 features, and $\gamma \approx 2$ instead of 6; the noise estimate becomes appropriately less conservative.

### 2.6 The Log Evidence: Measuring Quality and Checking Convergence

Each iteration computes the log marginal likelihood (evidence):

$$\mathcal{L} = \frac{1}{2}\left(\sum_j \log\alpha_j + N\log\beta - \log|\Sigma_\text{inv}| - \beta\|\mathbf{r}\|^2 - \boldsymbol{\mu}^T\Lambda\boldsymbol{\mu} + D\log(2\pi)\right) - \frac{N}{2}\log(2\pi)$$

This is the probability of observing the data $\mathbf{y}$, marginalised over all possible weight vectors $\mathbf{w}$, under the current hyperparameters $\alpha$ and $\beta$. In code:

```rust
// from crates/blr-core/src/ard.rs — log_evidence()
fn log_evidence(
    n: usize, d: usize,
    alpha: &[f64], beta: f64,
    mu: &[f64], logdet_sigma_inv: f64, residual_sq: f64,
) -> f64 {
    let log_alpha_sum: f64 = alpha.iter().map(|a| a.ln()).sum();
    let mu_lambda_mu: f64  = alpha.iter().zip(mu.iter()).map(|(a, m)| a * m * m).sum();

    0.5 * (
        log_alpha_sum
        + (n as f64) * beta.ln()
        - logdet_sigma_inv          // from cholesky_logdet()
        - beta * residual_sq
        - mu_lambda_mu
        + (d as f64) * (2.0 * PI).ln()
    ) - 0.5 * (n as f64) * (2.0 * PI).ln()
}
```

The `logdet_sigma_inv` term is the Cholesky log-determinant of the posterior precision matrix $\Lambda + \beta\Phi^T\Phi$, computed using `cholesky_logdet()` in the same iteration — no extra work.

The evidence is guaranteed to increase (or remain constant) with each EM iteration. This monotone convergence is what distinguishes EM from other optimisation methods and makes it trustworthy: if your log-evidence is decreasing, something is wrong — a bug in the M-step, a numerical issue, or an incorrect formula.

**Convergence criterion.** Rather than checking a single-step delta (which can be noisy near convergence), the implementation uses a **period-2** check:

```rust
// Period-2 convergence: compare smoothed log-evidence over pairs of iterations
let n_ev = log_evidences.len();
let delta = if n_ev >= 4 {
    let mean_curr = 0.5 * (log_evidences[n_ev - 1] + log_evidences[n_ev - 2]);
    let mean_prev = 0.5 * (log_evidences[n_ev - 3] + log_evidences[n_ev - 4]);
    (mean_curr - mean_prev).abs()
} else if n_ev >= 2 {
    (log_evidences[n_ev - 1] - log_evidences[n_ev - 2]).abs()
} else {
    f64::INFINITY
};

if delta < config.tol {
    break;
}
```

The period-2 smoothing averages two consecutive iterations before comparing to the previous pair. Near convergence, the $\alpha_j$ updates can oscillate slightly — the prior on a borderline-relevant feature might bounce between "slightly relevant" and "slightly irrelevant" before settling. Averaging pairs of iterations smooths this oscillation and prevents premature termination.

---

## Part III — Putting It All Together: The Hall Sensor Calibration

Let us trace through a complete calibration example. We have 60 real measurements from a Hall effect position sensor (loaded from `data/hall_sensor_calibration.csv`): a sensor voltage $y_i$ at each of 60 known displacement positions $x_i$.

### 3.1 Building the Design Matrix

The first decision is feature engineering. For a Hall sensor, physical reasoning suggests:

| Column | Feature $\phi_j(x)$ | Physical Hypothesis |
|--------|---------------------|---------------------|
| 0 | $1$ | Constant offset (always present) |
| 1 | $x$ | Linear Hall response (primary) |
| 2 | $x^2$ | Quadratic field non-uniformity |
| 3 | $x^3$ | Cubic non-linearity |
| 4 | $\tanh(x/0.8)$ | Hard magnetic saturation (tight knee) |
| 5 | $\tanh(x/1.5)$ | Gradual saturation rolloff (wide knee) |

We are not claiming that all six features are relevant. We are providing ARD with a *vocabulary* of physical hypotheses and letting it select. This is the correct way to think about feature engineering in a Bayesian model: you design a rich basis that spans the space of plausible physics, then let the data decide which elements matter.

```rust
// from crates/blr-core/examples/hall_sensor.rs
let (poly_mat, _) = features::polynomial(&x_vals, 3);  // [1, x, x², x³]

let mut phi = vec![0.0f64; n * 6];
for i in 0..n {
    phi[i * 6 + 0] = poly_mat[i * 4 + 0];          // 1
    phi[i * 6 + 1] = poly_mat[i * 4 + 1];          // x
    phi[i * 6 + 2] = poly_mat[i * 4 + 2];          // x²
    phi[i * 6 + 3] = poly_mat[i * 4 + 3];          // x³
    phi[i * 6 + 4] = (x_vals[i] / 0.8).tanh();     // tanh(x/0.8)
    phi[i * 6 + 5] = (x_vals[i] / 1.5).tanh();     // tanh(x/1.5)
}
```

The `features::polynomial` helper from `blr_core::features` returns an $N \times (D+1)$ matrix in row-major order — each row $i$ contains $[1, x_i, x_i^2, \ldots, x_i^D]$. We then augment it with the two `tanh` columns computed directly. The final design matrix `phi` is $60 \times 6$ in row-major layout, which is exactly what `fit()` expects.

### 3.2 Fitting the Model

```rust
// from crates/blr-core/examples/hall_sensor.rs
let config = ArdConfig { max_iter: 500, tol: 1e-7, ..ArdConfig::default() };
let fitted = fit(&phi, &y_vals, n, 6, &config).expect("BLR+ARD fit failed");
```

We tighten the tolerance slightly compared to the default (`1e-7` vs `1e-5`) and allow more iterations, because Hall sensor data is clean and the algorithm converges reliably. The `fit()` call returns a `FittedArd` struct containing the posterior distribution, the learned $\alpha$ and $\beta$ values, and the log-evidence trajectory.

### 3.3 Reading the Results

```rust
// from crates/blr-core/examples/hall_sensor.rs
println!("Noise std (learned): {:.6}", fitted.noise_std());
println!("Log marginal likelihood: {:.6}", fitted.log_marginal_likelihood());

let feature_names = ["1 (bias)", "x", "x²", "x³", "tanh(x/0.8)", "tanh(x/1.5)"];

// Posterior mean weights
for (name, &mu_j) in feature_names.iter().zip(fitted.posterior.mean.iter()) {
    println!("  {:<18} {:+.6}", name, mu_j);
}

// ARD relevance scores
let rel = fitted.relevance();   // = 1/α_j for each feature
for (name, r) in feature_names.iter().zip(rel.iter()) {
    println!("  {:<18} {:.3e}", name, r);
}

// Active feature mask
let active = fitted.relevant_features(None);  // threshold = geometric mean of α
```

The `relevance()` method returns $1/\alpha_j$ for each feature — a direct measure of how loosely the prior constrains each weight. Large relevance means the data is using that feature; small relevance means the prior has essentially pinned the weight to zero.

The `relevant_features()` method computes a boolean mask using the geometric mean of the $\alpha$ values as the threshold: features with $\alpha_j$ below the geometric mean are "relevant." This is a sensible default that adapts to the scale of the problem, though you can pass an explicit threshold when you have domain-specific cutoffs.

### 3.4 What the Numbers Tell You

Running `cargo run --example hall_sensor` from the repository root produces output like:

```bash
=== Hall Sensor BLR+ARD Results ===
EM iterations:          154
Noise std (learned):    0.102875
Log marginal likelihood:50.484934

Posterior mean weights:
  1 (bias)           +0.000001
  x                  -0.000040
  x²                 +0.000028
  x³                 -0.000017
  tanh(x/0.8)        +0.598037
  tanh(x/1.5)        +2.011467

ARD relevance (1/α — larger = more relevant):
  1 (bias)           1.328e-7
  x                  3.917e-6
  x²                 1.093e-7
  x³                 9.162e-8
  tanh(x/0.8)        3.707e-1
  tanh(x/1.5)        4.065e0

Active features (α < geometric-mean threshold):
  ✓ tanh(x/0.8)
  ✓ tanh(x/1.5)

In-sample RMSE:         0.101169
Mean total std:         0.104551
```

The numbers tell a revealing physical story — but not the one a naive linear model would suggest. The algorithm converged in 154 iterations (more than the 23 needed for synthetic data), learned a noise level of ~103 mV, and identified exactly two relevant features: **both nonlinear** `tanh` functions that model magnetic saturation. The posterior weights for the polynomial terms — the bias, linear, quadratic, and cubic — are all suppressively tiny (posterior means at the $10^{-5}$ level or below).

This is not a numerical accident. The relevance ratio between the saturation features and the linear term is approximately $10^5$. The algorithm is saying, with extraordinary clarity: "this real Hall sensor does *not* exhibit linear behavior. Its response saturates near both extremes — tight knee saturation modeled by `tanh(x/0.8)` and gradual rolloff by `tanh(x/1.5)`. Linear approximations are simply wrong."

Notice the posterior mean weights for the saturation features: `tanh(x/0.8)` has coefficient ~0.60 and `tanh(x/1.5)` has coefficient ~2.01. These two weighted `tanh` functions combine to approximate the true sensor's characteristic curve. The fact that they have *different* widths (`0.8` vs `1.5`) means they capture the asymmetry in how the device saturates — a subtle but crucial physical detail.

The in-sample RMSE of 0.101 V matches the learned noise std of 0.103 V almost exactly — evidence that the model has found the ground truth noise floor and no residual bias remains. The algorithm has correctly identified that remaining prediction error comes from measurement noise, not from model misspecification.

### 3.5 Making Predictions

```rust
let preds = fitted.predict(&phi, n, 6);

// preds.mean[i]          = E[y_i] = φ(x_i)ᵀ μ
// preds.aleatoric_std    = 1/√β  (noise; same for all points)
// preds.epistemic_std[i] = √(φ(x_i)ᵀ Σ φ(x_i))  (model uncertainty at point i)
// preds.total_std[i]     = √(aleatoric² + epistemic²)
```

The prediction decomposes uncertainty into two orthogonal components, as derived in Part 1:

$$\sigma^2_{*} = \underbrace{\beta^{-1}}_{\text{aleatoric}} + \underbrace{\boldsymbol{\phi}(x_*)^T \Sigma \boldsymbol{\phi}(x_*)}_{\text{epistemic}}$$

The **aleatoric** component is the irreducible measurement noise — the sensor is genuinely noisy at the ~8 mV level, and no amount of additional calibration data will reduce this term. The **epistemic** component is the model's uncertainty about the weights, which decreases as more calibration points are added.

A practical use of this decomposition: if the total uncertainty at a new operating point is dominated by epistemic uncertainty, you should gather more calibration data near that point. If it is dominated by aleatoric noise, more data won't help — you need a better sensor or a lower-noise measurement circuit. BLR+ARD tells you which situation you are in.

---

## Part IV — Common Pitfalls and Practical Wisdom

### 4.1 Feature Scaling

BLR+ARD is sensitive to the scale of the input features. If you have features measured in millimetres alongside features measured in volts, the $\alpha_j$ values are not comparable — a "large" $\alpha$ for a voltage feature might correspond to a tiny absolute regularisation in physical terms.

The best practice: normalise each input dimension to have zero mean and unit variance before computing the design matrix. The ARD $\alpha$ values then have a uniform interpretation across all features.

### 4.2 The Condition Number of $\Phi^T\Phi$

The posterior precision matrix $\Lambda + \beta\Phi^T\Phi$ inherits the condition number of $\Phi^T\Phi$. If your design matrix contains nearly linearly dependent columns — for example, $x^2$ and $(x+\epsilon)^2$ over a narrow input range — the Cholesky factorisation will fail with `SingularMatrix`.

The solution is to check for near-linear-dependence in your feature set and either remove redundant features or apply a small jitter to the diagonal:

```rust
// Add a small jitter for numerical stability
let jitter = 1e-9;
for j in 0..d {
    sigma_inv[(j, j)] += jitter;
}
```

This is acceptable because a jitter of $10^{-9}$ is far below the precision of any physical sensor, so it has no practical effect on the posterior but prevents Cholesky failures.

### 4.3 Interpreting "Irrelevant" Features

ARD suppression is a probabilistic statement, not a hard zero. A feature with $\alpha_j = 10^5$ has a posterior weight distribution $\mathcal{N}(0, 10^{-5})$ — the weight is extremely likely to be near zero, but it is not exactly zero. This matters when you need to predict at extrapolation points far outside the training range: the irrelevant features might have small but nonzero posterior means, contributing tiny but nonzero predictions.

If you need strict sparsity (exactly zero weights for irrelevant features), use the boolean mask from `relevant_features()` to zero out the corresponding columns of the design matrix and refit with the reduced feature set.

### 4.4 Not Enough Iterations vs. Wrong Convergence

The period-2 convergence criterion can be fooled by two consecutive iterations that happen to have nearly the same log-evidence for the wrong reason — for example, if the algorithm is oscillating around a saddle point. If your results look suspicious (e.g., all $\alpha_j$ equal to the initial value), increase `max_iter` and check whether the log-evidence trajectory is monotone increasing.

A monotone-increasing evidence trajectory is a sanity check you can perform for free: if it ever decreases by more than floating-point noise, there is a bug in the M-step.

---

## Conclusion: Three Levels of Understanding

This two-part series has developed the same algorithm at three levels:

1. **The statistical level** (Part 1): Bayesian updating as quadratic form manipulation; ARD as empirical Bayes over per-feature precision hyperparameters; the self-policing feedback loop between γ, α, and β.

2. **The linear algebra level** (Part 2, sections 1–1.5): the SPD structure of the posterior precision as a Cholesky-safe guarantee; the Woodbury identity connecting the observation-space ($N \times N$ Gram form) and parameter-space ($D \times D$ precision form) posterior updates, with adaptive dispatch to the cheaper form for any problem size; log-determinant via Cholesky diagonal as the trick that keeps log-probabilities finite.

3. **The implementation level** (Part 2, sections 2–3): pre-computing sufficient statistics to amortise $O(ND^2)$ work; reusing a single Cholesky factorisation for both $\Sigma$ and $\boldsymbol{\mu}$; the period-2 convergence criterion to handle near-oscillation; the ε-clamps that keep finite arithmetic finite.

Each level is necessary. The statistics tells you *what* to compute. The linear algebra tells you *how* to compute it without numerical disasters. The implementation tells you *when* to cache, reuse, and guard.

The result is a model that:

- Fits in ~23 iterations on 25 data points
- Learns a noise estimate of 8 mV with no manual tuning
- Correctly identifies 2 out of 6 features as relevant — matching the physics
- Provides calibrated uncertainty bands that distinguish noise from model uncertainty
- Runs on a CPU inside a WebAssembly component, with no GPU, no PyTorch, no dependencies beyond `faer = "0.24"`

That is what "principled machine learning" looks like in Rust.

---

## Appendices

### Appendix A: The Condition Number κ (Kappa)

[Section 1.1](#11-the-matrix-inversion-problem) mentions the **condition number** $\kappa(A)$ — an abstract quantity that governs numerical error in matrix computations. This appendix unpacks what it means and why it matters.

#### A.1 Definition and Intuition

The condition number $\kappa(A)$ of a matrix $A$ is a scalar that measures how sensitive the matrix is to perturbations or rounding errors. Informally:

- **Small $\kappa$** (close to 1): the matrix is **well-conditioned**. Tiny errors stay tiny.
- **Large $\kappa$** (e.g., $10^{10}$): the matrix is **ill-conditioned**. Tiny rounding errors can amplify into large errors in the computed result.

Formally, for an invertible matrix $A$:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\|$$

where $\|\cdot\|$ denotes a matrix norm (e.g., the spectral norm, the largest singular value). For a symmetric positive-definite (SPD) matrix — which is what we have in BLR — there is an elegant simplification:

$$\kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}$$

where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues of $A$. In other words: the condition number is the **ratio of the biggest eigenvalue to the smallest**.

**Intuition:** If all eigenvalues are similar in magnitude, the ratio is close to 1, and the matrix "stretches space uniformly." If one eigenvalue is much larger than others, the matrix stretches along that direction much more than others — it creates a very "elongated" geometry, and numerical errors accumulate along the short-stretch directions.

#### A.2 Why Condition Number Matters in Matrix Computations

When you solve a linear system $Ax = b$ or compute a matrix inverse $A^{-1}$, the presence of floating-point rounding errors ($\epsilon_{\text{machine}} \sim 10^{-16}$ for 64-bit floats) means you do not get the exact answer. Instead, you get an answer $\tilde{x}$ that satisfies $A\tilde{x} = b + \delta b$, where $\delta b$ is a tiny perturbation introduced by rounding.

The key result from numerical linear algebra is:

$$\frac{\|\delta x\|}{\|x\|} \lesssim \kappa(A) \cdot \epsilon_{\text{machine}}$$

In plain English: **the relative error in your computed solution is proportional to the condition number**.

Now, the relationship between relative error and computational method is crucial:

| Task | Method | Error Dependence |
|------|--------|------------------|
| Solve $Ax = b$ via Cholesky | Direct solve | $\mathcal{O}(\kappa(A) \cdot \epsilon_{\text{machine}})$ |
| Solve $Ax = b$ by inverting $A^{-1}$ then multiplying | Two-step | $\mathcal{O}(\kappa(A)^2 \cdot \epsilon_{\text{machine}})$ |

**The error from explicitly inverting grows as $\kappa(A)^2$, not $\kappa(A)$.** This is why [section 1.1](#11-the-matrix-inversion-problem) emphasizes "never invert a matrix if you can avoid it" — you are paying a squared condition number penalty.

#### A.3 Connection to Section 1.1 and Posterior Uncertainty

In [section 1.1](#11-the-matrix-inversion-problem), we write:

> *Worse, the error in $x$ scales with the square of $A$'s condition number $\kappa(A)^2$ — because you first invert with error $\sim \kappa(A) \cdot \epsilon_\text{machine}$, then multiply, amplifying that error again. By the time you examine the diagonal of $\Sigma_\text{post}$ to extract uncertainty estimates, your $\pm 3\sigma$ confidence bands might be meaningless.*

This is not mere hyperbole. Consider what happens:

1. You compute $\Sigma_\text{post} = (\Lambda + \beta\Phi^T\Phi)^{-1}$ by explicit inversion. The error grows as $\kappa(A)^2 \cdot \epsilon_{\text{machine}}$.
2. You then extract the diagonal entries $\Sigma_{\text{post}, jj}$ — these are your uncertainty estimates for each weight.
3. Those diagonal entries are now corrupted by the amplified error. Your "3-sigma confidence band" might actually be a 2-sigma or 4-sigma band due to numerical corruption.
4. Downstream, you use those uncertainty estimates to make decisions about calibration quality, data collection strategy, or safety margins. Bad uncertainty → bad decisions.

By contrast, using the Cholesky solve (section 1.2) reduces the error to $\mathcal{O}(\kappa(A) \cdot \epsilon_{\text{machine}})$ — a first-power dependence. For a moderately ill-conditioned matrix with $\kappa(A) = 10^4$, the difference is: $10^8 \cdot 10^{-16} = 10^{-8}$ (direct solve) vs. $10^{16} \cdot 10^{-16} = 1$ (explicit inversion). That is, explicit inversion can corrupt your answer by a factor of $10^8$.

#### A.4 Why Posterior Precision Is Always SPD (and Why That Protects Us)

The posterior precision matrix in BLR is $\Lambda + \beta\Phi^T\Phi$, where:

- $\Lambda = \text{diag}(\alpha_1, \ldots, \alpha_D)$ with all $\alpha_j > 0$ (strictly positive).
- $\Phi^T\Phi$ is a Gram matrix of the design matrix $\Phi$.

By [section 1.2](#12-the-cholesky-decomposition), we know this is always symmetric positive-definite (SPD). What does this buy us in terms of condition number?

**For an SPD matrix, the Cholesky algorithm has a built-in numerical safety feature:** the algorithm is **backward-stable** — the computed factor $L$ satisfies $(L + \delta L)(L + \delta L)^T = A$ with a backward error bounded by $\mathcal{O}(\kappa(A))$, not $\mathcal{O}(\kappa(A)^2)$.

More importantly: **an SPD matrix will never cause Cholesky to fail unexpectedly.** There are no "negative pivot" surprises. If the Cholesky algorithm reports `SingularMatrix`, it means the posterior precision is genuinely singular — a true mathematical problem (e.g., linearly dependent design matrix columns, or degenerate hyperparameter initialization), not a numerical artifact.

#### A.5 Practical: When Should You Worry About Condition Number?

In the context of the Hall sensor calibration (section 3):

1. **Design matrix features:** If your features are polynomials $[1, x, x^2, x^3, \ldots]$ evaluated over a narrow range (e.g., $x \in [-0.1, 0.1]$), the Gram matrix $\Phi^T\Phi$ can become ill-conditioned because high-degree polynomials are nearly collinear.
   - **Solution:** Use orthogonal polynomials (e.g., Chebyshev) or normalize input features to $[-1, 1]$.

2. **Hyperparameter scale:** If the $\alpha_j$ hyperparameters span many orders of magnitude (e.g., $\alpha_1 = 1$, $\alpha_6 = 10^8$), the diagonal of $\Lambda$ has a large condition number. When you add $\beta\Phi^T\Phi$ (with its own condition number), the result can be poorly conditioned.
   - **Solution:** Monitor the condition number of $\Lambda + \beta\Phi^T\Phi$ in the EM loop. If it exceeds, say, $10^{15}$, consider re-scaling or feature selection.

3. **EM convergence:** Occasionally, a borderline-relevant feature's $\alpha_j$ explodes to $10^{10}$, then the posterior precision becomes ill-conditioned, and convergence stalls.
   - **Solution:** The clamping in section 2.5 (`max(1e-8)`) prevents $\alpha_j$ from becoming infinite, keeping the condition number bounded.

For typical sensor calibration (10–100 measurements, 3–10 features), the posterior precision is well-conditioned and Cholesky factorisation is rock-solid numerically.

#### A.6 Further Reading

- **Trefethen & Bau (1997), *Numerical Linear Algebra*, Lectures 10–12:** authoritative treatment of condition numbers and Cholesky stability.
- **Golub & Van Loan (2013), *Matrix Computations* (4th ed.), Section 12.2:** comprehensive analysis of condition number and backward error.
- **Higham, N. J. (2002), *Accuracy and Stability of Numerical Algorithms* (2nd ed.):** the definitive reference on floating-point error analysis.

---

## References

1. **MacKay, D. J.** (1992). "Bayesian nonlinear modeling for the prediction competition." *ASHRAE Transactions* 98(1): 1052–1066. — Original ARD paper; derivation of α and β updates.

2. **Tipping, M. E.** (2001). "Sparse Bayesian learning and the relevance vector machine." *Journal of Machine Learning Research* 1(Jun): 211–244. — Modern treatment; Equations (14)–(16) for the M-step.

3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. MIT Press. — Chapters 7 and 10 for ARD and EM; Appendix C for matrix identities.

4. **Hennig, P.** *Probabilistic Machine Learning* (course materials). Universität Tübingen. — Lecture 3 for the Kalman gain / Gram-form (observation-space) derivation; Lecture 4 for empirical Bayes. The Woodbury identity (Appendix A of Part 1) establishes exact algebraic equivalence with the parameter-space precision form used by `condition_precision_form()`. [https://uni-tuebingen.de/](https://uni-tuebingen.de/)

5. **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. — Chapter 7 (empirical Bayes), Chapter 21 (practical BLR).

6. **Trefethen, L. N. & Bau, D.** (1997). *Numerical Linear Algebra*. SIAM. — Lectures 10–12 for Cholesky stability analysis.

7. **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. — Algorithm 4.2.2 for Cholesky; Section 12.2 for condition number analysis.

8. **Petersen, K. B. & Pedersen, M. S.** (2012). *The Matrix Cookbook* (v. 20121127). — Sections 4.1–4.2 for determinants and the matrix inversion lemma.

9. **faer Documentation.** <https://faer.rs/> — Rust linear algebra library used throughout `blr-core`.

---

## Part II — Continuing the Series

The complete implementation walkthrough — covering the Gram form, EM algorithm, numerical stability checks, and the full ARD learning loop — continues in the full version of this article, available in the [GitHub repository](https://github.com/wamli).

For now, the key takeaway is this: **the mathematical formulas from part one translate directly into production code when you replace matrix inversions with solves and factor once to amortize across multiple operations.** The same principles of stability and efficiency apply whether you're running on a desktop CPU or inside a WASM component on an embedded device.

---

## Bringing It Together

These two articles form a complete story:

1. **[Part 1: The Mathematics](/blog/blr-and-ard/)** — Why Bayesian Linear Regression with ARD is the right choice for sensor calibration, and how the math provides closed-form posterior solutions.

2. **Part 2: The Implementation (this post)** — How to translate those formulas into efficient, numerically stable Rust code.

If you're building a sensor calibration system and want to understand both the *why* and the *how*, start with part 1. If you're already convinced of the approach and want to dive into the code patterns, start here and reference part 1 as needed.

More articles on related topics are coming — including active learning strategies, deployment as WASM components, and advanced topics in hyperparameter selection.
