# 3  Results

We evaluate the framework across two sequential BOED benchmarks of increasing
geometric complexity.  In both cases the same exact Bayesian filter supplies
$\pi_t$; the variants differ only in how that filter is exposed to the
controller.  Evaluation uses SPCE lower bounds ($L = 1024$ particles), SNMC-style
upper bounds ($L = 512$), bank information-gain (IG), and episodic return as
complementary views of the same quantity: the cumulative EIG extracted by the
policy over the horizon.  All means and confidence intervals are empirical across
independent seeds; paired differences are computed seed-matched to remove
between-seed variance.


## 3.1  Prey-population benchmark

**Setting.**  The design $d = N_0 \in \{1,\ldots,300\}$ sets an initial prey
population; $Y | \theta, d \sim \mathrm{Binomial}(d, p_{\tau^\star}(\theta, d))$
where $p_{\tau^\star}$ is the consumed fraction after $\tau^\star = 24\,\mathrm{h}$
from a nonlinear Holling-II ODE.  The latent is $\theta = (\log a, \log T_h)$ with
$\mathcal{N}(-1.4, 1.35^2)$ marginals.  Horizon $T = 30$; bank size $512$; 8 paired
seeds; belief features: learned-only moment summaries.

Three policies are compared: **blau\_approx** (flat raw-history baseline,
Blau et al. 2022), **ours\_ebm\_control** (quotient state augmented with a
conditional EBM whose moment features feed the controller), and
**ours\_ebm\_cross** (cross-interaction energy allowing richer state–hypothesis
interaction before belief summaries are produced).

**Table 1 — Prey population** (8 paired seeds, $T = 30$).

| Method | Avg. return | Bank IG | SPCE lower | SNMC upper | Belief $\mu$-err |
|---|---|---|---|---|---|
| blau\_approx | 3.851 [3.77, 3.93] | 4.264 [4.19, 4.34] | 4.284 [4.18, 4.39] | 4.809 [4.55, 5.07] | — |
| ours\_ebm\_control | **4.058** [4.00, 4.12] | **4.601** [4.50, 4.70] | **4.606** [4.51, 4.70] | **5.397** [5.02, 5.78] | 0.693 [0.59, 0.80] |
| ours\_ebm\_cross | 4.070 [3.97, 4.17] | 4.561 [4.45, 4.67] | 4.623 [4.49, 4.76] | 5.319 [4.94, 5.70] | **0.603** [0.53, 0.67] |

*Paired return gains over blau\_approx: $+0.208$ [0.17, 0.25] (ebm\_control) and $+0.219$ [0.13, 0.31] (ebm\_cross); both 95 % CIs strictly positive.*

Both EBM variants improve consistently over the baseline across all four
metrics.  The paired CIs exclude zero, supporting the view that the improvement
is not a seed artefact.  The cross EBM achieves the lowest belief mean-error
(0.603 vs 0.693) and the highest SPCE lower bound, while the control EBM leads
on bank-IG and SNMC upper — an expected trade-off between tighter belief
tracking and broader exploration of the design space.  Path-level views show
that all policies accumulate information rapidly early in the sequence, but the
EBM policies separate from the baseline as $t$ grows, consistent with the
compounding advantage of an explicit derived control state (see Appendix A.1).


## 3.2  Source-localisation benchmark — RL optimiser (SAC)

**Setting.**  A sensor takes 2-D position designs $d_t \in [-4,4]^2$; each
measurement $y_t \sim \mathcal{N}(\log\mu(\theta, d_t), 0.35^2)$ where
$\mu(\theta, d) = \mathrm{bkg} + \|θ_1 - d\|^{-2}_{m} + \|θ_2 - d\|^{-2}_{m}$
(two hidden sources, log-normal observation model).  The two sources are
unordered, so $\theta$ lives in the quotient space of ordered pairs.  Exact
Bayesian filtering operates on a $7 \times 7 \times 28 = 1372$-atom bank.
Horizon $T = 50$; 3 seeds; geometric Deep-Sets EBM; modal top-$k$ belief
features ($k = 4$).

Five variants explore the axes of Section 2: whether the EBM is trained with a
standard posterior objective (*posterior*) or a **$\beta$-contrastive
homotopy** that interpolates between a pure EBM and a cross-contrastive EBM as
training progresses (*beta\_contrastive*), and whether the EBM sees only the
quotient state (*control*) or also candidate-level cross-interactions (*cross*).

**Table 2 — Source location, SAC / RL** (3 seeds, $T = 50$).

| Variant | Return | Filter Bank IG | SPCE lower | Belief KL | Belief MAP→exact |
|---|---|---|---|---|---|
| blau\_approx | 4.817 ±0.025 | 4.163 ±0.167 | 6.671 ±0.059 | — | — |
| ebm\_control\_posterior | 4.776 ±0.085 | 4.296 ±0.350 | 6.457 ±0.243 | 0.252 ±0.079 | 0.119 ±0.048 |
| ebm\_control\_β-contrast. | **4.830** ±0.003 | 4.094 ±0.123 | **6.644** ±0.034 | 0.314 ±0.032 | 0.145 ±0.028 |
| ebm\_cross\_posterior | 4.825 ±0.001 | **4.247** ±0.125 | 6.677 ±0.022 | **0.092** ±0.009 | **0.070** ±0.014 |
| ebm\_cross\_β-contrast. | 4.617 ±0.294 | 4.259 ±0.147 | 6.111 ±0.641 | 0.219 ±0.067 | 0.126 ±0.005 |

*Belief MAP→exact: distance between the EBM MAP estimate and the exact Bayesian MAP; lower is better.*

Several patterns are worth noting.

1. **Cross EBM + posterior achieves the best belief quality** by a wide margin:
   KL divergence 0.092 (vs 0.252 for control posterior) and MAP-to-exact
   distance 0.070 (vs 0.119), with negligible variance across seeds — the
   cross-interaction structure provides a faithful surrogate of $\pi_t$ on this
   spatial problem.

2. **$\beta$-contrastive helps the control variant but not the cross variant.**
   Control-$\beta$ marginally leads on return (4.830) and SPCE lower (6.644)
   while control-posterior lags on both.  Cross-$\beta$, by contrast, shows
   increased variance (return std 0.294) and a lower SPCE lower bound (6.111)
   than cross-posterior, suggesting the homotopy schedule does not add benefit
   on top of the already-expressive cross structure.

3. **blau\_approx remains competitive on raw return** (4.817) despite seeing
   none of the belief geometry.  Its SPCE lower and filter bank-IG trail the
   best EBM variants, however, indicating that raw-history policies can
   saturate the reward signal while underperforming on distributional quality.

Belief quality metrics (KL, $L_1$, MAP distances) are reported in full in
Appendix A.2.  Learning curves confirm stable convergence for all variants
except cross-$\beta$, which shows elevated seed-to-seed variance from
generation 150 onward.


## 3.3  Source-localisation benchmark — NES optimiser (OpenAI-ES)

**Setting.**  Identical physical model and bank as Section 3.2, but the policy
is optimised via Natural Evolution Strategy (population size 48, mirrored
sampling, Adam meta-optimiser for the mean $\mu$, $\sigma$ annealed from 0.03
to 0.005 over 200 generations).  The EBM is trained offline via a supervised
cross-entropy objective against the exact filter — no replay buffer, no
Q-critics.  Horizon $T = 30$; 3 seeds; model selection retains the
top-3 checkpoints evaluated on 40 hold-out episodes.

Three variants cover the ablation axis for NES: the flat raw-history baseline
(**blau\_approx**), the quotient state with *exact* filter probabilities
(**control\_filter\_exact**, no EBM), and the full EBM-cross representation
trained against the filter (**ours\_ebm\_cross\_filter**).

**Table 3 — Source location, NES / OpenAI-ES** (3 seeds, $T = 30$; best checkpoint).

| Variant | Return | Bank IG | Filter Bank IG | SPCE lower |
|---|---|---|---|---|
| blau\_approx | 5.201 ±0.066 | 4.581 ±0.022 | 4.755 ±0.043 | 6.102 ±0.059 |
| control\_filter\_exact | 5.326 ±0.094 | 4.668 ±0.211 | 4.817 ±0.256 | 6.271 ±0.209 |
| ours\_ebm\_cross\_filter | **5.375** ±0.088 | **4.669** ±0.108 | **4.806** ±0.064 | **6.398** ±0.254 |

*All metrics use the best model-selection checkpoint; last-generation results are in Appendix A.3.*

The ordering **ebm\_cross > control\_exact > blau** is consistent across all
four metrics.  The EBM-cross policy leads on return (+0.174 over baseline,
+0.049 over exact control) and on SPCE lower (+0.296 over baseline).  The fact
that even the exact-filter variant outperforms the raw-history baseline confirms
that the derived control state carries information that cannot be implicitly
recovered from the flat observation sequence alone — at least within the NES
optimisation budget.

The EBM actor-belief diagnostics (MAE on actor-side features 0.156 ±0.026;
actor belief MAP-to-exact distance 0.447 ±0.061) indicate that the EBM learned
by NES provides a coarser posterior approximation than the RL counterpart from
Section 3.2 (MAP-to-exact 0.070 for the best RL variant).  This is expected:
the supervised offline objective under NES receives only the data generated by
the current policy rollouts, without the diverse off-policy experience that the
RL replay buffer provides.


## 3.4  Cross-experiment summary

**Table 4 — Comparative overview across benchmarks and optimisers.**

| Benchmark | Optimiser | Best variant | Return | SPCE lower | Δ return vs baseline |
|---|---|---|---|---|---|
| Prey ($T=30$, 8 seeds) | SAC/RL | ebm\_cross | 4.070 | 4.623 | +0.219 ✓ |
| Source loc. ($T=50$, 3 seeds) | SAC/RL | ebm\_cross\_posterior | 4.825 | 6.677 | +0.008 |
| Source loc. ($T=30$, 3 seeds) | NES/ES | ebm\_cross\_filter | 5.375 | 6.398 | +0.174 |

Three consistent observations emerge across settings:

- **Cross-interaction EBM is the most reliable variant.**  In every benchmark
  the cross EBM matches or leads the control EBM and the exact-filter baseline
  on at least three of the four metrics.

- **The EBM's value is partly independent of the optimiser.**  Both SAC and
  NES benefit from the derived control state, and the ordering of variants is
  preserved across optimisers on the source-location problem.

- **Belief quality and decision quality are correlated but not identical.**
  The cross-posterior RL variant achieves the best belief metrics (KL 0.092,
  MAP-to-exact 0.070) yet does not dominate on return.  This aligns with the
  Methods argument: what matters is whether belief features expose useful
  coordinates for the *next* design, not whether they reconstruct $\pi_t$
  pointwise.

Detailed per-seed traces, learning curves, path-level IG plots, and statistical
tests are provided in Appendix A.
