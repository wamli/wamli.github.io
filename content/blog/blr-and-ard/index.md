---
title: "When Your Sensor Knows What It Doesn't Know"
description: "The Math Behind Bayesian Linear Regression and Automatic Relevance Determination"
summary: "Bayesian Linear Regression with Automatic Relevance Determination combines principled uncertainty quantification, automatic feature selection, and a closed-form solution—all from one coherent mathematical framework. This post explains how and why."
date: 2026-05-15
lastmod: 2026-05-15
draft: false
weight: 10

series: "blr-sensor-calibration"
seriesOrder: 1

categories:
  - "Bayesian Linear Regression"
tags:
  - "bayesian-inference"
  - "sensor-calibration"
  - "mathematics"
  - "blr+ard"
  - "machine-learning"
  - "uncertainty-quantification"

math: true
seriesTitle: "Bayesian Linear Regression"
contributors: []
pinned: false
homepage: false

seo:
  title: "Bayesian Linear Regression and Automatic Relevance Determination for Sensor Calibration"
  description: "Complete mathematical exposition of BLR+ARD: from Bayes' theorem through the closed-form posterior, hyperparameter learning, and practical calibration."
  canonical: ""
  noindex: false
---

*Bayesian Linear Regression (BLR) with Automatic Relevance Determination (BLR+ARD) is one of those rare techniques that feels almost too good: it fits your data, quantifies its own uncertainty, and automatically discards the features that don't matter — all from a single, coherent mathematical framework. This post explains how it works and, more importantly, why it has to work that way.*

{{< series >}}

---

## 1. Why Bayesian Statistics for Sensor Calibration?

### The Problem: Calibration Is More Than Curve Fitting

When you calibrate a [Hall effect sensor](https://en.wikipedia.org/wiki/Hall_effect_sensor) — the kind of magnetic sensor that tells a motor controller where the rotor is — you're solving what looks like a simple problem: you know the sensor's output voltage at a set of known magnetic field strengths. You want a function that maps voltage to field strength (or vice versa). Curve fitting, right?

Not quite. The naive approach — fit a polynomial with [least squares](https://en.wikipedia.org/wiki/Least_squares), pick the one with the best cross-validation score, call it done — hides a fundamental question: *how confident are you?* When the sensor operates in the middle of its calibrated range, fine. But what about at the edges? What about when a new sensor unit has slightly different manufacturing tolerances? What about when the next data point comes in: should you trust it, or was the last ten measurements flaky?

Least squares gives you a function. Bayesian Inference gives you a *belief*, expressed as a probability distribution over functions. That belief encodes not just "here's my best fit" but "here's how certain I am, and here's how my certainty changes across the input range."

### Why Not a Neural Network?

The question almost always comes up. Neural networks are powerful, flexible, and increasingly easy to deploy. Why bother with a specialized Bayesian technique? There are three reasons for this.

**First, data.** A Hall sensor calibration session typically produces tens to hundreds of measurements, not millions. Neural networks are data-hungry. A six-feature BLR model with ARD reaches confident, physically meaningful calibration with as few as ten data points.

**Second, interpretability.** When a BLR model tells you that the cubic polynomial feature is irrelevant for your Hall sensor, it is telling you something physically true: the sensor's response really is dominated by a linear term plus a smooth saturation. The $\alpha$ values — the ARD hyperparameters we'll meet in Section 4 — are a direct window into the structure of the data. Neural network weights are not.

**Third, uncertainty.** BLR produces a full posterior predictive distribution at every query point. You know not just the predicted output but how wide the uncertainty band is, and whether that band is wide because of model uncertainty (not enough data to pin down the weights) or measurement noise (the physical process is inherently stochastic). No post-hoc calibration technique, no conformal prediction wrapper — it falls out of the math automatically.

### What Bayesian Linear Regression Promises

To summarize the promise before we build the machinery:

1. **Principled uncertainty**: predictions are distributions, not point estimates.
2. **Automatic feature selection**: ARD hyperparameters "vote" on which basis functions matter. Irrelevant ones are suppressed to zero without manual selection or cross-validation.
3. **Interpretable results**: every parameter has a physical or statistical meaning you can inspect.
4. **Efficiency**: the algorithm converges in a small number of iterations, making it practical for embedded and real-time calibration loops.

The cost is that you may want to understand the math before you trust the results. That's what this post is for.

---

## 2. Bayes' Theorem and the Gaussian Miracle

### 2.1 Bayes' Theorem: From Belief to Data-Driven Belief

The foundation is familiar:

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}$$

In words: the probability of parameters $\theta$ *given data* $\mathcal{D}$ is proportional to the likelihood of the data given those parameters, multiplied by our prior belief in those parameters.

$p(\theta)$ is the **prior** — what we believed before seeing data. $p(\mathcal{D} \mid \theta)$ is the **likelihood** — how probable the data would be if the parameters were $\theta$. $p(\theta \mid \mathcal{D})$ is the **posterior** — what we believe after seeing data. The denominator $p(\mathcal{D})$ is just a normalizing constant; it ensures the posterior integrates to 1.

The hard part of Bayesian inference, in general, is computing this posterior. For arbitrary distributions, the integral in the denominator is intractable. Sampling methods (e.g. MCMC) handle the general case, but they are slow. For the specific choice of **Gaussian distributions**, something remarkable happens: the posterior is also Gaussian, and the update has a closed-form solution in terms of matrix algebra.

This is the Gaussian miracle, and it is the engine that makes BLR+ARD computationally tractable.

### 2.2 The Gaussian Miracle: Closure Under Bayes

A Gaussian distribution for a vector $x \in \mathbb{R}^d$ is written $\mathcal{N}(x; \mu, \Sigma)$, where $\mu$ is the mean vector and $\Sigma$ is the covariance matrix. The key property is:

> **If the prior $p(x)$ is Gaussian and the likelihood $p(y \mid x)$ is Gaussian in $x$, then the posterior $p(x \mid y)$ is also Gaussian.**

This is not a coincidence. The log of a Gaussian is a quadratic function of its argument. Adding two quadratic functions gives another quadratic function. Exponentiating a quadratic gives a Gaussian. Bayesian updating with Gaussians is, at its core, just completing the square — the same algebraic operation you learned in high school.

Prof. Philipp Hennig from University of Tübingen illustrates this beautifully in his lecture ["Probabilistic ML - 03 - Gaussian Inference" | (53:50)](https://www.youtube.com/watch?v=CXCNoAw3YYM&list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm&index=6). His framework for Gaussian inference is:

$$p(x) = \mathcal{N}(x;\, \mu,\, \Sigma)$$

$$p(y \mid x) = \mathcal{N}(y;\, Ax + b,\, \Lambda^{-1})$$

where $A$ is a linear map from the latent $x$ to the observation $y$, $b$ is an offset, and $\Lambda$ is the precision matrix (inverse covariance) of the observation noise.

> 🚧 Warning
>
> In the lecture, the second equation is actually $p(y \mid x) = \mathcal{N}(y;\, Ax + b,\, \Lambda)$.
> Note that there is a difference in the inversion of $\Lambda$.

**Result — the posterior is:**

$$p(x \mid y) = \mathcal{N}(x;\, \mu + K(y - A\mu - b),\, \Sigma - K A \Sigma)$$

where the **Kalman gain** $K$ is:

$$K = \Sigma A^T \left(\Lambda^{-1} + A \Sigma A^T\right)^{-1}$$

Let's unpack the terminology, because each piece has a name for a reason:

- **Prior mean** $\mu$: our best guess before seeing $y$.
- **Residual** $(y - A\mu - b)$: how far the actual observation is from what we predicted. This is the "surprise."
- **Gram matrix** $(\Lambda^{-1} + A \Sigma A^T)$: the total covariance of the observation — noise covariance plus the uncertainty propagated from the prior through the linear map $A$.
- **Gain** $K$: how much we update the mean for each unit of surprise. High gain → observation dominates. Low gain → prior dominates.
- **Posterior covariance** $\Sigma - K A \Sigma$: strictly *less* than the prior covariance. We always become more certain after observing data, never less.

This is exact, not approximate. No samples, no variational bound, no MCMC. Pure linear algebra.

### 2.3 Why This Matters for Regression

The formula above is the general Gaussian inference update. For linear regression, we will specialize it: $x$ will be our weight vector $\mathbf{w}$, $y$ will be our sensor measurements, $A$ will be the design matrix $\Phi$, and $\Lambda^{-1}$ will be our noise model. [Section 3](#3-linear-regression-through-the-bayesian-lens) builds this specialization explicitly.

Before moving on, note what the framework *does not require*: it does not require knowledge of the true weights. It does not require that we iterate until convergence. Given the prior and the likelihood, the posterior is determined in one shot. What *will* require iteration is learning the hyperparameters ($\alpha$ and $\beta$) — but that is a separate problem from computing the posterior.

---

## 3. Linear Regression Through the Bayesian Lens

### 3.1 The Regression Model

We model sensor output as:

$$y_i = \mathbf{w}^T \boldsymbol{\phi}(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \beta^{-1})$$

where:
- $x_i$ is the $i$-th input (e.g., magnetic field strength)
- $\boldsymbol{\phi}(x_i) \in \mathbb{R}^D$ is a vector of **basis functions** evaluated at $x_i$
- $\mathbf{w} \in \mathbb{R}^D$ are the **weights** we want to infer
- $\epsilon_i$ is measurement noise with precision $\beta$ (i.e., variance $\sigma^2 = 1/\beta$)

Stack all $N$ observations into matrix form:

$$\mathbf{y} = \Phi \mathbf{w} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{y} \in \mathbb{R}^N$ is the vector of $N$ observations
- $\Phi \in \mathbb{R}^{N \times D}$ is the **design matrix** — row $i$ is $\boldsymbol{\phi}(x_i)^T$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \beta^{-1} I)$ is the noise vector

The design matrix is the bridge between raw inputs and the linear model. Each column of $\Phi$ corresponds to one basis function applied to every data point. Bayesian inference learns which columns matter.

### 3.2 Feature Engineering: Encoding Physical Hypotheses

The basis functions $\boldsymbol{\phi}(x)$ are *not* learned — they are designed by the engineer based on physical understanding. This is the feature engineering step, and it is where domain knowledge enters the picture.

For a Hall effect position sensor, the physics suggests:

$$\boldsymbol{\phi}(x) = \left[1,\; x,\; x^2,\; x^3,\; \tanh(x/0.8),\; \tanh(x/1.5)\right]^T$$

Each component encodes a physical hypothesis:

| Feature | Physical Hypothesis |
|---------|---------------------|
| $1$ (bias) | The sensor has a constant offset regardless of field (always true) |
| $x$ | The primary Hall response is linear in displacement |
| $x^2$, $x^3$ | Polynomial field non-uniformity at moderate displacements |
| $\tanh(x/0.8)$ | Hard magnetic saturation with characteristic length 0.8 mm |
| $\tanh(x/1.5)$ | Gradual saturation rolloff with characteristic length 1.5 mm |

The $\tanh$ features capture the fact that at large displacements, the permanent magnet's field begins to saturate the ferromagnetic core of the sensor — the response "clips" smoothly rather than increasing indefinitely. Two different width parameters hedge between a tight saturation profile and a gentler rolloff.

The crucial point: **the model does not need to know which features are actually relevant.** The ARD algorithm will determine that from data. If the sensor happens to have a genuinely linear response in the calibration range, then the $x^2$, $x^3$, and one of the $\tanh$ features will be driven to zero. The engineer provides a *vocabulary* of physical hypotheses; the algorithm selects the right ones automatically.

This beats black-box learning in a specific way: every feature that survives ARD selection has a name. You can look at the final $\alpha$ values and read off: "this sensor has a linear response with a gradual saturation rolloff and a 0.5 V offset." That is interpretable calibration.

### 3.3 The Prior on Weights

Before seeing any data, we express our belief about the weights as a Gaussian prior:

$$p(\mathbf{w}) = \mathcal{N}\!\left(\mathbf{0},\; \Lambda^{-1}\right)$$

where $\Lambda = \text{diag}(\alpha_1, \alpha_2, \ldots, \alpha_D)$ is a diagonal **precision matrix** — one regularization parameter $\alpha_j$ per weight.

Why zero mean? Because before seeing data, we have no reason to believe any weight is large. We are agnostic about direction. The prior simply says: "I expect the weights to be small, but I'll learn from data how small."

Why diagonal? Because our prior belief about weight $w_j$ has nothing to do with weight $w_k$. Before seeing data, each feature is independent. Correlations between weights emerge from the data, not from our prior.

Why per-feature $\alpha_j$ rather than a single shared $\alpha$? This is the key ARD design choice. A single shared $\alpha$ would regularize all features equally. But different features genuinely have different signal-to-noise properties. A bias feature is almost always relevant; a high-degree polynomial feature probably is not. With per-feature $\alpha_j$, the algorithm can independently "turn the volume down" on each feature. This is Automatic Relevance Determination (ARD).

Initially, we set all $\alpha_j$ to the same small value (a weak prior). The EM algorithm described in [Section 5](#5-the-mackay-algorithm-finding-the-optimal-hyperparameters) will learn the optimal $\alpha_j$ values from data.

### 3.4 The Posterior: Updating Beliefs with Data

Now we apply the Gaussian inference framework from Section 2 to our regression setting. Mapping the notation:

| Hennig's Framework | BLR Setting |
|-------------------|-------------|
| $x$ (latent variable) | $\mathbf{w}$ (weights) |
| $\mu$ (prior mean) | $\mathbf{0}$ |
| $\Sigma$ (prior covariance) | $\Lambda^{-1} = \text{diag}(\alpha_j^{-1})$ |
| $A$ (linear map) | $\Phi$ (design matrix, transposed; see [note](#ref-transpose)) |
| $y$ (observations) | $\mathbf{y}$ (sensor measurements) |
| $\Lambda^{-1}$ (obs. noise covariance) | $\beta^{-1} I$ (aka *homoscedastic* noise) |

> 🚧 Warning
>
> Be careful with conventions: In Prof. Hennig's framework, the observation model is $y = Ax + b$.
> In regression, $y = \Phi w$, so $A = \Phi$ where each row of $\Phi$ is a feature vector.
> The algebra works out consistently.

Substituting $\mu = \mathbf{0}$, $b = 0$, $A = \Phi$, and the observation noise covariance $\beta^{-1}I$ into the Kalman gain form from [Section 2.2](#22-the-gaussian-miracle-closure-under-bayes) gives an intermediate result that inverts an $N \times N$ **Gram matrix** — one that grows with the number of observations:

$$K = \Lambda^{-1}\Phi^T\underbrace{\left(\beta^{-1}I_N + \Phi\Lambda^{-1}\Phi^T\right)^{-1}}_{\text{Gram matrix}}$$

This is mathematically correct, but it works in the *wrong* space. The question — "what do the 6 feature weights look like after seeing the data?" — lives in a 6-dimensional parameter space, yet the Gram matrix inversion is $N \times N$, growing with every new calibration measurement you add. For $D = 6$ features and $N = 100$ measurements, we would be inverting a $100 \times 100$ matrix to answer a question that lives in $\mathbb{R}^6$.

The **Woodbury matrix identity** (also known as the matrix inversion lemma; see [reference](#ref-woodbury-ref)) provides an exact algebraic route from the $N \times N$ Gram inversion to a $D \times D$ precision matrix inversion — a switch from observation space to parameter space. The full derivation is in [Appendix A](#appendix-a-the-woodbury-lemma---switching-from-observation-space-to-parameter-space). If you want to continue reading without stopping to verify the algebra, the key takeaway is: the transformation is exact, not an approximation. The result is:

$$\boxed{\Sigma_\text{post} = \left(\Lambda + \beta\, \Phi^T \Phi\right)^{-1}}$$

$$\boxed{\boldsymbol{\mu}_\text{post} = \beta\, \Sigma_\text{post}\, \Phi^T \mathbf{y}}$$

This is **Bayesian Linear Regression** in closed form. Let's read these formulas carefully, because they encode everything.

**The posterior covariance** $\Sigma_\text{post}$ is the inverse of the sum of two matrices:

- $\Lambda$: the prior precision — our regularization (depends only on our beliefs, not data)
- $\beta\, \Phi^T \Phi$: the data precision — how strongly the data constrains the weights (depends only on data)

As $N \to \infty$, the data precision dominates and the posterior collapses to a delta function. As $N \to 0$, only the prior survives. The interpolation between prior and data is automatic and continuous.

**The posterior mean** $\boldsymbol{\mu}_\text{post}$ is proportional to $\Phi^T \mathbf{y}$ (the *"signal"* in the data), scaled by both $\beta$ (how much we trust individual measurements) and $\Sigma_\text{post}$ (which redistributes this signal according to the precision structure).

Compare this to **least squares**: you minimize $\mathcal{L}(\mathbf{w}) = \lVert \mathbf{y} - \Phi \mathbf{w} \rVert^2$. In least squares, the [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) (second derivative) is $H_{\text{LS}} = 2\beta\, \Phi^T \Phi$. Notice that $\beta\, \Phi^T \Phi$ appears directly in our Bayesian posterior covariance formula as well:

$$\Sigma_\text{post} = (\Lambda + \underbrace{\beta\, \Phi^T \Phi}_{\text{least squares Hessian term}})^{-1}$$

In the Bayesian form, $\beta\, \Phi^T \Phi$ is the data's **information matrix** (the inverse of curvature in the likelihood), and $\Lambda$ adds **prior information**. The posterior covariance is the inverse of the total information. Where least squares gives only a point estimate, Bayesian regression inverts this information matrix to quantify uncertainty through the covariance. The regularization term $\Lambda$ prevents overfitting — [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression) is exactly the Bayesian posterior under a uniform $\alpha$ prior, and ARD generalizes it by letting each feature have its own regularization strength.

### 3.5 Predictions as Probability Distributions

Given the posterior $p(\mathbf{w} \mid \Phi, \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_\text{post}, \Sigma_\text{post})$, how do we predict at a new input $x_*$?

We want the **posterior predictive distribution**:

$$p(y_* \mid x_*, \Phi, \mathbf{y}) = \int p(y_* \mid \mathbf{w}, x_*)\, p(\mathbf{w} \mid \Phi, \mathbf{y})\, d\mathbf{w}$$

Because everything is Gaussian, this integral is tractable. The result is:

$$p(y_* \mid x_*, \Phi, \mathbf{y}) = \mathcal{N}(\mu_*, \sigma_*^2)$$

where:

$$\mu_* = \boldsymbol{\phi}(x_*)^T \boldsymbol{\mu}_\text{post}$$

$$\sigma_*^2 = \underbrace{\beta^{-1}}_{\text{aleatoric}} + \underbrace{\boldsymbol{\phi}(x_*)^T\, \Sigma_\text{post}\, \boldsymbol{\phi}(x_*)}_{\text{epistemic}}$$

The variance decomposes into two parts:

- **Aleatoric uncertainty** ($\beta^{-1}$): irreducible measurement noise. Even with infinite data, the sensor still has physical noise. This part never goes to zero.
- **Epistemic uncertainty** ($\boldsymbol{\phi}(x_*)^T \Sigma_\text{post} \boldsymbol{\phi}(x_*)$): uncertainty about the weights. This part goes to zero as $N \to \infty$ — more data pins down the weights.

This is where BLR's value becomes concrete. In regions where the training data is dense, epistemic uncertainty is small. In regions where data is sparse (e.g., the edges of the calibration range), it grows. A well-designed calibration system can use this signal to request additional measurements exactly where they are most needed — [**active learning**](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)), which is a natural extension of this framework.

---

## 4. The Unknowns: Hyperparameters and Why They Matter

### 4.1 What We Know, What We Don't

Let's take stock. The BLR posterior formulas are clean and exact. But they depend on two quantities we haven't specified:

| Symbol | Role | Known? |
|--------|------|--------|
| $\mathbf{y}$, $\Phi$ | Observations and design matrix | ✓ Yes — we measured them |
| $\mathbf{w}$ | Regression weights | ✗ Inferred (via posterior) |
| $\alpha_j$ | Per-feature prior precision (ARD) | ✗ Must learn from data |
| $\beta$ | Noise precision | ✗ Must learn from data |

The weights $\mathbf{w}$ are "first-level" unknowns — they are inferred by the posterior distribution, which is exact given $\alpha$ and $\beta$.

The hyperparameters $\alpha_j$ and $\beta$ are "second-level" unknowns — they govern the prior and noise model. Choosing them badly hurts the posterior. We need a principled way to learn them from the same data used to fit the model.

### 4.2 $\alpha_j$: The ARD Knobs

$\alpha_j$ is the precision of the prior on weight $w_j$. Recall that the prior is $w_j \sim \mathcal{N}(0, \alpha_j^{-1})$. So:

- **Large $\alpha_j$**: tight prior, weight strongly pulled toward zero. Feature $j$ is being told "you probably don't matter."
- **Small $\alpha_j$**: loose prior, weight is free to be large. Feature $j$ is being told "do what the data says."

The ARD idea, due to [MacKay (1992)](#ref-mackay-1992) and developed further by [Tipping & Bishop (2001)](#ref-tipping-bishop-2001), is to learn separate $\alpha_j$ values for each feature dimension. During the optimization:

- If the data provides strong evidence for feature $j$ (e.g., the $x$ term in a Hall sensor with linear response), $\alpha_j$ stays small. The posterior for $w_j$ is broad and data-driven.
- If the data provides *no* evidence for feature $j$ (e.g., the $x^2$ term in a genuinely linear sensor), $\alpha_j$ grows large — potentially toward infinity. The posterior for $w_j$ collapses to zero. The feature is automatically pruned.

This is automatic feature selection without any explicit thresholding, cross-validation, or human decision. The physics is in the data; ARD reads it out.

In the Hall sensor calibration validation from our implementation, the result was:

```bash
α[bias]       = 4.06       (relevant: captures the 0.5V offset)
α[B-field]    = 3.77       (relevant: the linear Hall response)
α[B-field²]   = 894,766    (SUPPRESSED: no quadratic signal)
α[B-field³]   = 1,844      (SUPPRESSED: no cubic signal)

Ratio α[B-field²] / α[B-field] = 237,390×
```

That ratio of a quarter-million-to-one is not a numerical glitch. It is the algorithm stating, unambiguously: "the linear term is essential, the quadratic term is noise." That conclusion is physically correct, and it emerged without us telling the algorithm anything about Hall sensor physics.

### 4.3 $\beta$: The Noise Knob

$\beta = 1/\sigma^2$ is the precision of the measurement noise. Higher $\beta$ means less noise — the algorithm trusts individual measurements more. Lower $\beta$ means more noise — the algorithm is more skeptical of individual measurements and regularizes the fit.

In the posterior formulas, $\beta$ multiplies $\Phi^T \Phi$ and $\Phi^T \mathbf{y}$. Increasing $\beta$ is like having more data — it sharpens the posterior. Decreasing $\beta$ broadens it.

Why learn $\beta$ rather than fixing it? Manual noise estimation is error-prone. With $N = 10$ calibration points, it is easy to misestimate noise by a factor of two. The EM algorithm infers $\beta$ from the residuals, automatically accounting for the fact that a portion of the apparent residuals is "explained" by the weights. [Appendix C](#appendix-c-where-does-the--update-come-from) walks through the derivation in detail.

### 4.4 The Central Problem: Evidence Maximization

We now know what we need: optimal values for $\alpha_1, \ldots, \alpha_D$ and $\beta$. How do we find them?

The Bayesian answer: maximize the [**marginal likelihood**](https://en.wikipedia.org/wiki/Marginal_likelihood), also called the **evidence**:

$$p(\mathbf{y} \mid \alpha, \beta) = \int p(\mathbf{y} \mid \mathbf{w}, \beta)\, p(\mathbf{w} \mid \alpha)\, d\mathbf{w}$$

This integral asks: "How probable is the observed data under our model, after having averaged out the weights?" It is the probability of the data according to the prior-weighted ensemble of all possible weight vectors.

Why is this the right objective? Because it automatically trades off **data fit** against **model complexity**. A model with very large $\alpha$ values (strong prior) has low complexity but may not fit the data well. A model with very small $\alpha$ values (weak prior) fits training data well but overfits. The evidence is maximized at the sweet spot — the configuration of hyperparameters that makes the data as probable as possible without overfitting.

This is called **Type-II Maximum Likelihood** or **Empirical Bayes**. It is distinct from Type-I Maximum Likelihood (which would maximize $p(\mathbf{y} \mid \mathbf{w})$ over $\mathbf{w}$ directly, giving least squares). Type-II integrates out the weights and optimizes the hyperparameters. It is a more principled approach precisely because it avoids conditioning on any specific weight vector.

---

## 5. The MacKay Algorithm: Finding the Optimal Hyperparameters

### 5.1 The EM Loop

To maximize the evidence with respect to $\alpha$ and $\beta$, we use the **Expectation-Maximization (EM) algorithm**:

- **E-step** (Expectation): Fix the current hyperparameters $\alpha^{(t)}, \beta^{(t)}$. Compute the posterior distribution $p(\mathbf{w} \mid \Phi, \mathbf{y}, \alpha^{(t)}, \beta^{(t)})$ — exactly, using the BLR formulas from Section 3.4. This gives us $\boldsymbol{\mu}^{(t)}$ and $\Sigma^{(t)}$.

- **M-step** (Maximization): Use the posterior statistics ($\boldsymbol{\mu}^{(t)}$ and $\Sigma^{(t)}$) to update the hyperparameters to maximize the expected log-evidence. MacKay derived closed-form fixed-point update rules that make this step a single matrix operation.

Iterate until convergence. The algorithm is guaranteed to not decrease the evidence at each step, so it converges monotonically.

### 5.2 The $\gamma$ Parameter: Data vs. Prior

Before presenting the update rules, we need to meet $\gamma_j$ — perhaps the most interpretable quantity in the entire framework.

Define:

$$\gamma_j = 1 - \alpha_j \Sigma_{jj}^{\text{post}}$$

where $\Sigma_{jj}^{\text{post}}$ is the $j$-th diagonal element of the posterior covariance.

To understand $\gamma_j$, consider two extremes:

**Case 1: The prior dominates.** If $\alpha_j$ is very large, the prior says "this weight is zero." The data cannot override a strong prior with limited observations. As a result, the posterior variance $\Sigma_{jj}^{\text{post}}$ approaches the prior variance $\alpha_j^{-1}$, so $\alpha_j \Sigma_{jj}^{\text{post}} \approx 1$, and $\gamma_j \approx 0$.

**Case 2: The data dominates.** If $\alpha_j$ is small (loose prior) and the data strongly constrains $w_j$, then the posterior variance $\Sigma_{jj}^{\text{post}}$ is much smaller than the prior variance $\alpha_j^{-1}$, so $\alpha_j \Sigma_{jj}^{\text{post}} \approx 0$, and $\gamma_j \approx 1$.

So $\gamma_j \in [0, 1]$ is the fraction of information about weight $w_j$ that comes from the data (as opposed to the prior). It is sometimes called the **effective number of data points** allocated to feature $j$.

The sum $\gamma = \sum_j \gamma_j$ is the total **effective number of determined parameters** — how many features the data is actually constraining, accounting for the prior regularization.

This is an elegant generalization of the classical degrees-of-freedom concept. In ordinary least squares, the effective degrees of freedom is exactly $D$ (the number of features). In BLR with ARD, features that are being suppressed by large $\alpha_j$ contribute nearly zero to $\gamma$, reducing the effective complexity of the model automatically.

### 5.3 The Update Rules

After deriving the gradient of the log-evidence with respect to $\alpha_j$ and $\beta$ (see [Appendix B](#appendix-b-where-do--and--come-from) and [C](#appendix-c-where-does-the--update-come-from) for the full derivation), MacKay obtained these fixed-point rules:

**Update for $\alpha_j$ (the ARD hyperparameter):**

$$\alpha_j^{\text{new}} = \frac{\gamma_j}{\mu_j^2}$$

where $\mu_j = [\boldsymbol{\mu}_\text{post}]_j$ is the posterior mean of weight $w_j$.

**Update for $\beta$ (the noise precision):**

$$\beta^{\text{new}} = \frac{N - \gamma}{\left\lVert \mathbf{y} - \Phi \boldsymbol{\mu}_\text{post} \right\rVert^2}$$

where $\gamma = \sum_j \gamma_j$ is the total effective number of parameters.

These formulas are deceptively simple. Let's read them carefully.

### 5.4 Reading the $\alpha_j$ Update

$$\alpha_j^{\text{new}} = \frac{\gamma_j}{\mu_j^2}$$

Think of the numerator and denominator as competing forces:

- **Numerator $\gamma_j$**: How much does the *data* say about feature $j$? If the data says nothing (data is irrelevant to this feature), $\gamma_j \to 0$ and the new $\alpha_j$ will be very large — the prior gets tightened, the weight is pushed to zero. ARD prunes the feature.

- **Denominator $\mu_j^2$**: How large is the posterior weight? If the posterior mean is large (the feature has a strong effect), the update keeps $\alpha_j$ small — the feature remains relevant.

The fixed-point property: at convergence, these two forces balance. A feature survives when the data evidence for it is proportional to its squared effect size. This is a natural and elegant condition for feature relevance.

Numerically, we always add a small regularization $\epsilon$ to the denominator:

$$\alpha_j^{\text{new}} = \frac{\gamma_j}{\mu_j^2 + \epsilon}$$

to avoid division by zero when $\mu_j$ is near zero — which happens exactly when a feature is being pruned.

### 5.5 Reading the $\beta$ Update

$$\beta^{\text{new}} = \frac{N - \gamma}{\left\lVert \mathbf{y} - \Phi \boldsymbol{\mu}_\text{post} \right\rVert^2}$$

This is the inverse of a noise variance estimate. The noise variance is estimated as:

$$\hat{\sigma}^2 = \frac{\left\lVert \mathbf{y} - \Phi \boldsymbol{\mu}_\text{post} \right\rVert^2}{N - \gamma}$$

which has the beautiful structure of the classical unbiased estimator of variance:

$$\hat{\sigma}^2_{\text{classical}} = \frac{\text{RSS}}{N - D}$$

where **RSS** (Residual Sum of Squares) is $\sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \left\lVert \mathbf{y} - \Phi \boldsymbol{\mu}_\text{post} \right\rVert^2$ — the sum of squared prediction errors.

But instead of dividing by $N - D$ (where $D$ counts all parameters), we divide by $N - \gamma$ (where $\gamma$ counts only the *effectively used* parameters). Features that have been pruned by ARD don't consume degrees of freedom — their $\gamma_j \approx 0$ contribution is essentially zero. The noise estimate automatically corrects for the fact that the model's complexity is lower than the raw feature count suggests. Then $\beta = 1/\hat{\sigma}^2$ gives the precision formula above.

This also reveals the anti-overfitting mechanism. Suppose the weights try to "soak up" residual noise (overfitting):

1. The fit improves slightly, but $\gamma$ increases (more features are being used).
2. The numerator $N - \gamma$ shrinks.
3. The new $\beta$ is lower (more noise is assumed), which reduces how much we trust each new observation.
4. Lower $\beta$ feeds back to the $\alpha_j$ update, pushing some $\alpha_j$ values up, pruning features.

**The algorithm polices itself.** No explicit regularization parameter to tune. No validation set. No early stopping heuristic. The evidence framework creates an automatic feedback loop between the noise model and the feature relevance — a property that feels almost too elegant to be practical, but which works robustly in implementation.

### 5.6 The Complete Algorithm

Putting it together, the MacKay BLR+ARD algorithm is:

```
Initialize: α₁ = α₂ = ... = αD = α₀,  β = β₀

Repeat until convergence:
  1. (E-step) Compute posterior:
       Σ_post = (diag(α) + β · ΦᵀΦ)⁻¹
       μ_post = β · Σ_post · Φᵀy

  2. Compute gamma:
       γⱼ = 1 − αⱼ · [Σ_post]ⱼⱼ   for each j
       γ = Σⱼ γⱼ

  3. (M-step) Update hyperparameters:
       αⱼ_new = γⱼ / (μⱼ² + ε)     for each j
       β_new = (N − γ) / ‖y − Φμ_post‖²

  4. Check convergence:
       If max_j |αⱼ_new − αⱼ| < tolerance: stop
       Else: α ← α_new,  β ← β_new,  continue
```

**Convergence note:** In practice, the algorithm typically converges in 10–50 iterations for problems of this scale. A practical robustness trick: if $\beta$ oscillates between iterations, damp it with an exponential moving average before using it in the next E-step.

### 5.7 ARD in Action: What Happens During the Iterations

During the iterations, the $\alpha_j$ values for irrelevant features do not merely grow — they grow *unboundedly*. Once $\gamma_j \approx 0$ and $\mu_j \approx 0$, the update rule effectively becomes $\alpha_j \leftarrow 0 / 0^+ = \text{large}$. Large $\alpha_j$ feeds back into the E-step, making the posterior variance for that feature even smaller, pushing $\gamma_j$ closer to zero. The pruning is self-reinforcing and extremely robust.

For relevant features, the opposite happens: small $\alpha_j$ keeps the posterior responsive to data. $\gamma_j$ stays near 1. The weight $\mu_j$ stabilizes at a physically meaningful value.

The net result — as seen in our Hall sensor example — is a sparse solution where the $\alpha$ ratios between relevant and irrelevant features span many orders of magnitude (237,000× for the quadratic term). This is not numerical instability; it is the algorithm communicating very loudly that the irrelevant features should be zero.

---

## 6. Summary: From Bayes' Theorem to Automatic Sensor Calibration

### 6.1 The Full Picture

Let's retrace the journey:

1. **Bayes' Theorem** tells us how to update beliefs with data.
2. **Gaussian priors and likelihoods** make this update a tractable linear algebra operation, with exact closed-form posteriors.
3. **Bayesian Linear Regression** specializes Gaussian inference to the regression setting: the prior is over weights, the likelihood is a linear model with noise.
4. **Feature engineering** encodes physical domain knowledge as basis functions — the algorithm selects which ones matter.
5. **ARD hyperparameters** ($\alpha_j$) give each feature its own regularization strength. Features irrelevant to the data are automatically pruned.
6. **Evidence maximization** determines the optimal $\alpha$ and $\beta$ from data alone — no cross-validation, no external validation set.
7. **The MacKay fixed-point algorithm** makes evidence maximization computationally efficient: one matrix inversion and a handful of scalar updates per iteration.

### 6.2 Takeaways for the Practicing Engineer

**You can now read the math.** The formulas in [MacKay (1992)](#ref-mackay-1992) and [Tipping & Bishop (2001)](#ref-tipping-bishop-2001) will not seem opaque after working through this post. The key equations are:

$$\Sigma_\text{post} = (\Lambda + \beta\, \Phi^T \Phi)^{-1}, \quad \boldsymbol{\mu}_\text{post} = \beta\, \Sigma_\text{post}\, \Phi^T \mathbf{y}$$

$$\gamma_j = 1 - \alpha_j \Sigma_{jj}^{\text{post}}, \quad \alpha_j^{\text{new}} = \frac{\gamma_j}{\mu_j^2 + \epsilon}, \quad \beta^{\text{new}} = \frac{N - \gamma}{\lVert \mathbf{y} - \Phi \boldsymbol{\mu}_\text{post} \rVert^2}$$

**You have a mental model for debugging.** If $\alpha_j$ is not growing for a feature you know should be irrelevant, something is wrong with your basis functions — perhaps a relevant feature has been accidentally correlated with an irrelevant one. If $\beta$ converges to a very low value, your model may be underfitting (your feature vocabulary is incomplete). The hyperparameter values are diagnostic, not just outputs.

**You understand the uncertainty.** When the calibration reports a 95% confidence interval, you know where that interval comes from: the posterior predictive variance decomposes into aleatoric noise ($\beta^{-1}$) and epistemic weight uncertainty ($\boldsymbol{\phi}^T \Sigma_\text{post} \boldsymbol{\phi}$). You can inspect each component separately.

**You can evaluate the method critically.** BLR+ARD makes strong assumptions: Gaussian noise, linear model in the basis functions, i.i.d. observations. For sensor calibration in a controlled measurement session, these assumptions are almost always valid. For more complex settings (time-varying drift, multiplicative noise, heavy-tailed outliers), they may not be. Knowing the assumptions lets you decide when to use the tool and when to look for alternatives.

### 6.3 Coming Up: Part 2

The next post in this series moves from mathematics to code: "From Math to Rust — Implementing BLR+ARD with the `faer` Crate." We will walk through the Rust implementation in the `blr-core` crate, map every formula from this post to the corresponding code, and benchmark the implementation against the Python reference.

If you want to see the algorithm running before then, the `sensor-calibration-component` WASM binary demonstrates end-to-end calibration of a simulated Hall sensor — offline, portable, with no dependencies beyond the component itself.

---

## Appendices

### Appendix A: The Woodbury Lemma - Switching from Observation Space to Parameter Space

*This appendix shows the explicit algebraic step that converts the Kalman gain form from [Section 2.2](#22-the-gaussian-miracle-closure-under-bayes) into the compact BLR posterior formulas in [Section 3.4](#34-the-posterior-updating-beliefs-with-data). If you are comfortable accepting the result on faith, skip this on first reading — you can always come back.*

The Woodbury identity is ubiquitous in machine learning precisely because it lets you choose which space to work in. The canonical rule of thumb: **invert in the smaller space**. When you have fewer features than observations ($D < N$), work in parameter space — this is the BLR case. When you have more features than observations ($D > N$), work in observation space — this is the kernel trick case (e.g., Gaussian Processes). Prof. Hennig's lecture slides show both forms of the posterior side by side (the "Kalman gain" form and the "precision matrix" form) precisely because neither is universally preferable; the right choice depends on the relative sizes of $N$ and $D$.

#### A.1 The Problem: Two Spaces, Two Costs

After substituting the BLR assignments into the general Hennig formula, the posterior covariance takes the Kalman gain form:

$$\Sigma_\text{post} = \Lambda^{-1} - \Lambda^{-1}\Phi^T\underbrace{\left(\beta^{-1}I_N + \Phi\Lambda^{-1}\Phi^T\right)^{-1}}_{N \times N \text{ inversion}}\Phi\Lambda^{-1}$$

and the posterior mean:

$$\boldsymbol{\mu}_\text{post} = \Lambda^{-1}\Phi^T\underbrace{\left(\beta^{-1}I_N + \Phi\Lambda^{-1}\Phi^T\right)^{-1}}_{N \times N \text{ inversion}}\mathbf{y}$$

Both require inverting the $N \times N$ matrix $G = \beta^{-1}I_N + \Phi\Lambda^{-1}\Phi^T$. The cost of inverting an $n \times n$ matrix scales as $O(n^3)$, so:

| Form | Matrix inverted | Size | Scales with |
|------|-----------------|------|-------------|
| Kalman / observation space | $\beta^{-1}I_N + \Phi\Lambda^{-1}\Phi^T$ | $N \times N$ | Observations |
| Precision / parameter space | $\Lambda + \beta\Phi^T\Phi$ | $D \times D$ | Features |

For $D = 6$ features and $N = 100$ observations, the observation-space inversion is roughly $(100/6)^3 \approx 4600\times$ more expensive — and it gets *worse* the more calibration data you collect. The parameter-space form, once the sufficient statistics $\Phi^T\Phi$ and $\Phi^T\mathbf{y}$ are pre-computed ($O(ND^2)$ once), never grows with $N$ again.

#### A.2 The Woodbury Matrix Identity

For matrices of compatible dimensions with $P$ and $R$ invertible:

$$(P^{-1} + B^T R^{-1} B)^{-1} = P - PB^T(BPB^T + R)^{-1}BP \tag{A.1}$$

$$(P^{-1} + B^T R^{-1} B)^{-1} B^T R^{-1} = PB^T(BPB^T + R)^{-1} \tag{A.2}$$

Both identities can be verified by multiplying the left-hand side by $(P^{-1} + B^T R^{-1} B)$ and confirming you recover the identity matrix — straightforward matrix algebra, no approximation involved.

#### A.3 Applying the Identity to BLR

Set $P = \Lambda^{-1}$ (prior covariance, $D \times D$), $B = \Phi$ (design matrix, $N \times D$), $R = \beta^{-1}I_N$ (noise covariance, $N \times N$). Then:

- $P^{-1} = \Lambda$
- $B^T R^{-1} B = \Phi^T (\beta^{-1}I)^{-1} \Phi = \beta\Phi^T\Phi$
- $BPB^T + R = \Phi\Lambda^{-1}\Phi^T + \beta^{-1}I_N$

**Posterior covariance** — applying identity (A.1):

$$\Sigma_\text{post} = (\Lambda^{-1} - \Lambda^{-1}\Phi^T G^{-1}\Phi\Lambda^{-1}) \stackrel{\text{A.1}}{=} (\Lambda + \beta\Phi^T\Phi)^{-1}$$

**Posterior mean** (with zero prior mean) — applying identity (A.2):

$$\boldsymbol{\mu}_\text{post} = \Lambda^{-1}\Phi^T G^{-1}\mathbf{y} \stackrel{\text{A.2}}{=} (\Lambda + \beta\Phi^T\Phi)^{-1}\beta\Phi^T\mathbf{y} = \beta\,\Sigma_\text{post}\,\Phi^T\mathbf{y}$$

The two boxed formulas in Section 3.4 are the direct output of these two substitutions. No approximation, no hidden assumption — just the Woodbury identity applied once to each equation.

---

### Appendix B: Where Do $\alpha_j$ and $\gamma_j$ Come From?

*This appendix derives the MacKay fixed-point rule for $\alpha_j$ from first principles. The main text quoted the result; here we show why it has to be true.*

#### B.1 The Log-Posterior Is a Quadratic Function

In Bayesian inference, we often work with the **log-posterior** because it is easier to differentiate:

$$\ln p(\mathbf{w} \mid \mathbf{y}) = \ln p(\mathbf{y} \mid \mathbf{w}) + \ln p(\mathbf{w}) + \text{const}$$

For our Gaussian setting:

$$\ln p(\mathbf{y} \mid \mathbf{w}) = -\frac{\beta}{2} \lVert\mathbf{y} - \Phi \mathbf{w}\rVert^2 + \text{const}$$

$$\ln p(\mathbf{w}) = -\frac{1}{2} \mathbf{w}^T \Lambda \mathbf{w} + \text{const} = -\frac{1}{2} \sum_j \alpha_j w_j^2 + \text{const}$$

Both terms are **quadratic** in $\mathbf{w}$. Their sum is quadratic. A quadratic with negative leading coefficient is the log of a Gaussian. Therefore the posterior is Gaussian — this is the algebraic proof of the Gaussian miracle from Section 2.2.

#### B.2 The Hessian Is the Precision Matrix

The Hessian of the log-posterior with respect to $\mathbf{w}$ is:

$$H = -\frac{\partial^2 \ln p(\mathbf{w} \mid \mathbf{y})}{\partial \mathbf{w}^2} = \beta\, \Phi^T \Phi + \Lambda$$

Notice: $H$ is exactly the inverse of the posterior covariance:

$$\Sigma_\text{post} = H^{-1} = (\Lambda + \beta\, \Phi^T \Phi)^{-1}$$

This is not a coincidence. In a Gaussian, the **precision matrix** (inverse covariance) is the Hessian of the negative log-density. The Hessian encodes the curvature of the log-posterior surface:

- Large Hessian eigenvalue → sharp curvature → narrow posterior → confident about that direction in weight space.
- Small Hessian eigenvalue → flat surface → broad posterior → uncertain.

The posterior covariance $\Sigma_\text{post}$ is literally the inverse of this curvature.

#### B.3 The Eigenvalue Balance and the Origin of $\gamma$

To understand $\gamma_j$, consider first a simplified case: all features share a single precision $\alpha$ (no ARD yet). The Hessian is:

$$H = \alpha I + \beta\, \Phi^T \Phi$$

Let $\lambda_i$ be the eigenvalues of the data matrix $\beta\, \Phi^T \Phi$. Then the eigenvalues of $H$ are $(\lambda_i + \alpha)$. MacKay defined the effective number of determined parameters as:

$$\gamma = \sum_{i=1}^{D} \frac{\lambda_i}{\lambda_i + \alpha}$$

Each term is a number between 0 and 1:

- If $\lambda_i \gg \alpha$: the data dominates in direction $i$. That parameter is "data-determined." Contribution to $\gamma$: nearly 1.
- If $\lambda_i \ll \alpha$: the prior dominates. That parameter is "prior-determined." Contribution to $\gamma$: nearly 0.

So $\gamma$ counts how many parameters the data is actually constraining, on a soft scale from 0 to $D$.

For the general ARD case with per-feature $\alpha_j$, the per-feature version is:

$$\gamma_j = 1 - \alpha_j \Sigma_{jj}^{\text{post}}$$

This can be derived from the same eigenvalue logic applied to the $j$-th diagonal: $\Sigma_{jj}^{\text{post}}$ is the posterior variance for feature $j$, which is small when data constrains $w_j$ and large (close to $\alpha_j^{-1}$) when the prior dominates. Substituting the prior-dominated limit $\Sigma_{jj} \approx \alpha_j^{-1}$ gives $\gamma_j \approx 0$; substituting a fully data-determined case gives $\gamma_j \approx 1$.

#### B.4 The MacKay Fixed-Point Rule for $\alpha_j$

Now we derive $\alpha_j^{\text{new}} = \gamma_j / \mu_j^2$. The starting point is the log marginal likelihood (evidence):

$$\ln p(\mathbf{y} \mid \alpha, \beta) = \ln \int p(\mathbf{y} \mid \mathbf{w}, \beta)\, p(\mathbf{w} \mid \alpha)\, d\mathbf{w}$$

For Gaussians, this integral evaluates to a closed-form expression involving the posterior quantities. Taking the derivative with respect to $\alpha_j$ and setting it to zero (the optimality condition):

$$\frac{\partial \ln p(\mathbf{y} \mid \alpha, \beta)}{\partial \alpha_j} = 0$$

After algebraic manipulation — using the matrix identity $\frac{\partial}{\partial \alpha_j} \ln \det H = \frac{\partial}{\partial \alpha_j} \text{tr}(\ln H)$ and differentiating through the posterior covariance — MacKay obtained:

$$\frac{1}{\alpha_j} - \Sigma_{jj}^{\text{post}} - \mu_j^2 = 0$$

Rearranging:

$$\frac{1}{\alpha_j} = \mu_j^2 + \Sigma_{jj}^{\text{post}}$$

Multiplying both sides by $\alpha_j$:

$$1 = \alpha_j \mu_j^2 + \alpha_j \Sigma_{jj}^{\text{post}} = \alpha_j \mu_j^2 + (1 - \gamma_j)$$

Therefore:

$$\alpha_j \mu_j^2 = \gamma_j \implies \alpha_j = \frac{\gamma_j}{\mu_j^2}$$

This is a **fixed-point equation**: the optimal $\alpha_j$ is expressed in terms of the posterior statistics ($\mu_j$, $\Sigma_{jj}^{\text{post}}$), which themselves depend on $\alpha_j$. Iterating the update alternately with the posterior computation converges to the evidence-maximizing solution.

The chain of reasoning is:

$$\text{Hessian} = \text{Precision} \xrightarrow{\text{Eigenvalues}} \gamma_j = \text{data fraction} \xrightarrow{\text{Evidence gradient}} \alpha_j = \frac{\gamma_j}{\mu_j^2}$$

**Full reference:** See MacKay[^mackay-1992] in Primary References — Sections 4 and Appendix D contain the complete derivation.

---

### Appendix C: Where Does the $\beta$ Update Come From?

*This appendix derives the noise precision update from the marginal likelihood, explaining the "degrees of freedom" interpretation.*

#### C.1 The Log Evidence and Its Two Competing Terms

The log marginal likelihood decomposes naturally into an **accuracy term** and a **complexity term**:

$$\ln p(\mathbf{y} \mid \alpha, \beta) = \underbrace{-\frac{1}{2} \lVert\mathbf{y} - \Phi \boldsymbol{\mu}_\text{post}\rVert^2 \beta + \ldots}_{\text{accuracy}} \underbrace{- \frac{1}{2} \ln |H| + \ldots}_{\text{complexity}}$$

The accuracy term rewards small residuals (good fit). The complexity term penalizes high curvature in the log-posterior (a model that is too flexible, fitting noise as well as signal). The evidence is maximized at the sweet spot where neither term dominates.

This is the Bayesian Occam's razor: among all models that fit the data roughly equally well, the simpler one has higher evidence.

#### C.2 The Derivative Condition for $\beta$

Taking the derivative of the log evidence with respect to $\beta$ and setting to zero:

$$\frac{\partial \ln p(\mathbf{y} \mid \alpha, \beta)}{\partial \beta} = 0$$

After applying the chain rule through the determinant and the posterior mean, MacKay derived the fixed-point condition:

$$\frac{N}{\beta} - \lVert\mathbf{y} - \Phi \boldsymbol{\mu}_\text{post}\rVert^2 - \text{tr}(\Phi \Sigma_\text{post} \Phi^T) = 0$$

The last term, $\text{tr}(\Phi \Sigma_\text{post} \Phi^T)$, can be shown to equal $(N - \gamma) / \beta$ at the fixed point, leading to:

$$\beta^{\text{new}} = \frac{N - \gamma}{\lVert\mathbf{y} - \Phi \boldsymbol{\mu}_\text{post}\rVert^2}$$

#### C.3 The "Data Points Used Up" Interpretation

This formula has a beautiful interpretation. Think of your $N$ data points as a budget:

- The model "spends" $\gamma$ data points to determine the weights. A feature with $\gamma_j \approx 1$ "consumes" one full data point to determine its weight.
- Only $N - \gamma$ data points remain to estimate the noise.

The noise variance estimate is therefore residuals divided by the remaining budget:

$$\hat{\sigma}^2 = \frac{1}{\beta^{\text{new}}} = \frac{\lVert\mathbf{y} - \Phi \boldsymbol{\mu}_\text{post}\rVert^2}{N - \gamma}$$

Compare to classical statistics, where the unbiased estimator uses $N - D$ (the number of observations minus the number of fitted parameters). The BLR version replaces the hard count $D$ with the soft count $\gamma = \sum_j \gamma_j$. Features pruned by ARD (with $\gamma_j \approx 0$) do not consume degrees of freedom. This is important: it means the noise estimate is not biased by including irrelevant features in the count.

#### C.4 The Self-Policing Anti-Overfitting Mechanism

The interaction between the $\alpha_j$ and $\beta$ updates creates an elegant overfitting prevention mechanism. Suppose the weights try to overfit — absorbing noise into the fit:

1. Overfitting reduces the residuals $\lVert\mathbf{y} - \Phi \boldsymbol{\mu}\rVert^2$.
2. But it also increases $\gamma$ (more features are being actively used).
3. The numerator $N - \gamma$ shrinks; the denominator shrinks too.
4. The net effect on $\beta$ is ambiguous or even decreasing (less confidence in observations).
5. Lower $\beta$ flows into the $\alpha_j$ updates, increasing some $\alpha_j$, pruning features.
6. The model is forced back toward sparsity.

MacKay's summary of this property:

> "The formula ensures that the noise estimate isn't biased by the fact that the weights are already trying to 'soak up' some of the patterns in the data. It's a very elegant way of saying: the noise variance is the average squared error, but only averaged over the dimensions that weren't already captured by the model weights."

The BLR+ARD objective function builds in Occam's razor, complexity penalization, and overfitting prevention — not as separate components that require tuning, but as natural consequences of the evidence maximization framework.

---

## Footnotes & References

### References

This section collects literature and course references cited throughout the post.
For definitions and foundational concepts (e.g., Hessian, Hall sensor), see the
linked Wikipedia articles embedded in the text.

### Implementation Notes

<a id="ref-transpose"></a>
**Notation Convention:** In Prof. Hennig's framework, the linear map $A$ represents the forward transformation from latent variables to observations. In the BLR context, $A = \Phi$ where each *row* of the design matrix $\Phi$ is a feature vector evaluated at a single observation point. This gives the standard regression form: $\mathbf{y} = \Phi \mathbf{w}$.

### Literature & Courses

<a id="ref-mackay-1992"></a>
**MacKay, D.J.C. (1992).** "A Practical Bayesian Framework for Backpropagation Networks." *Neural Computation* 4(3):448–472. — Original derivation of the evidence framework and ARD fixed-point rules.

<a id="ref-tipping-bishop-2001"></a>
**Tipping, M.E. & Bishop, C.M. (2001).** "Sparse Bayesian Learning and the Relevance Vector Machine." *Journal of Machine Learning Research* 1:211–244. — Extension of ARD to kernel methods; clear exposition of fixed-point rules.

<a id="ref-hennig-2025"></a>
**Hennig, P. (2025).** "Probabilistic Machine Learning" course, University of Tübingen. Lecture series: https://www.youtube.com/playlist?list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm. See lecture 3 ("Gaussian Inference") for the framework used in Section 2.

<a id="ref-rasmussen-williams-2006"></a>
**Rasmussen, C.E. & Williams, C.K.I. (2006).** *Gaussian Processes for Machine Learning*. MIT Press. Free PDF: http://gaussianprocess.org/gpml/. Chapters 2 (Regression) and 5 (Model Selection).

<a id="ref-murphy-2022"></a>
**Murphy, K.P. (2022).** *Probabilistic Machine Learning: An Introduction*. MIT Press. Free PDF: https://probml.github.io/pml-book/. Chapter 11 (Linear Regression).

<a id="ref-ramsden-2006"></a>
**Ramsden, E. (2006).** *Hall-Effect Sensors: Theory and Application*. Elsevier. Sensor physics background.

<a id="ref-woodbury-ref"></a>
**Woodbury Matrix Identity.** Wikipedia: https://en.wikipedia.org/wiki/Woodbury_matrix_identity. The matrix inversion lemma used in [Appendix A](#appendix-a-the-woodbury-lemma---switching-from-observation-space-to-parameter-space).

<a id="ref-marginal-likelihood"></a>
**Marginal Likelihood & Evidence.** Wikipedia: https://en.wikipedia.org/wiki/Marginal_likelihood. Overview of evidence and model selection.

---

## Continuing the Series

This article lays the mathematical foundation. The next part, [**From Math to Silicon: Implementing BLR+ARD with Rust and faer**](/blog/blr-implementation-rust-faer/), walks through the production code that translates these formulas into efficient, numerically robust implementations — focusing on the key insight that you should never invert a matrix if a linear solve will do.
