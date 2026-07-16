Typicality Dampening
Overview
Computational-pathology pretraining corpora are strongly long-tailed: a training batch is dominated by morphologically common tiles (stroma, adipose, background, ordinary epithelium), while diagnostically rare morphology is infrequent. Under a standard self-distillation objective each tile contributes comparable per-sample gradient, so the redundant majority dominates the update and the rare signal is under-weighted — even though the redundant tiles are, by construction, the ones whose gradient is most repetitive.
Typicality dampening estimates online how common each tile's morphology is relative to the running data distribution, and down-weights common tiles in the DINO CLS objective. The estimate is a nonparametric redundancy score against a running reservoir of morphology signatures. Only the DINO CLS loss is modulated; the iBOT patch losses are left unchanged.
Components
Representative prototypes (R). `K'` unit-norm anchors on the sphere of the 256-d DINO-head bottleneck. A dedicated optimizer trains them each step to align each anchor with a distinct DINO prototype (a nearest-neighbour term) while keeping the anchors mutually spread (a covariance term); the anchors are re-projected to the sphere after each step. R compresses the DINO prototype space into `K'` representative directions.
Morphology signature. For a tile, let `z` be its L2-normalized DINO-head bottleneck representation of the first global crop. The signature `s = z R̂ᵀ ∈ R^{K'}` is the vector of cosine similarities of `z` to the representative prototypes — a compact descriptor of the tile's morphology in the `K'` learned directions. One signature per tile.
Signature bank (FIFO keep-last-M). A fixed-capacity reservoir of the `M` most recent signatures, maintained as a first-in-first-out queue: each step the current batch of signatures is appended and the oldest are evicted. The bank is a running, representative sample of the current signature distribution. It is global — signatures are all-gathered across data-parallel ranks so every rank maintains the identical bank, and the mechanism is multi-node. Filling begins only after a representation-warmup period, so the reservoir samples a converged representation.
Redundancy scorer (k-NN density). The bank is used as a k-nearest-neighbour density estimator. For a query signature, `d` is the mean distance to its `k` nearest bank entries; the bank's own within-bank k-NN distances provide a calibration `(μ, σ)`. The typicality score is

```
t = 1 − Φ((d − μ) / σ) ∈ [0, 1],

```

with `Φ` the standard normal CDF: `t ≈ 1` for a signature in a densely populated (common) region, `t ≈ 0` for one far from the bank (rare).
Loss modulation. Two variants modulate the per-sample DINO CLS loss by `t`:

* adaptive temperature: `τ_i = τ_base · (1 + α · t_i)` — common tiles receive a higher student temperature, flattening their target distributions and shrinking their gradient contribution;
* weighted loss: `w_i = 1 − β · t_i` — common tiles are down-weighted directly.
Only the DINO CLS loss is affected; iBOT patch losses are unmodulated.
Churn policy: FIFO keep-last-M
The bank is a density estimator, so the redundancy score is meaningful only if the bank is a faithful sample of the data distribution. This determines the replacement policy.
Eviction is content-independent. The stationary distribution of bank contents equals the data distribution only when entries are evicted independently of their content (FIFO or uniform-random). A content-dependent rule — for example evicting the entry nearest each insertion — is repulsive: it drives the bank toward a space-filling configuration of approximately uniform density, so the k-NN distance measures coverage geometry rather than frequency, which is the wrong quantity for a redundancy score. FIFO is content-independent and preserves the data density.
The replacement rate is not a free parameter. Under a stream that is IID at the batch scale, the k-NN sampling variance of the density estimate depends on the bank size `M` and the neighbour count `k` but not on the replacement rate; the only rate-dependent error is a staleness bias from entries produced by earlier model states, and this bias decreases monotonically as the replacement rate increases. The optimal policy is therefore to replace as fast as possible — to keep the most recent `M` signatures. FIFO keep-last-M is this policy, and there is no replacement-fraction hyperparameter. This holds provided the stream carries no persistent slide- or shard-level correlation between consecutive batches; that condition is verified below.
Neighbour count. The estimator uses `k > 1`. The single-nearest-neighbour (`k = 1`) distance has ~100% relative sampling fluctuation; the k-NN distance reduces this by ~`1/√k` at the cost of mild spatial smoothing.
Verification of the batch-IID assumption
The optimality of FIFO keep-last-M rests on the training stream being IID at the batch scale. We verified this on a baseline pretraining run (`FMC_ViT-B_stab_baseline_rev3`, typicality dampening disabled), measuring the stream's correlation structure in the encoder's own representation, independently of the typicality machinery.
Method. Features are the raw pre-norm CLS representation (the pooled per-tile feature). A fixed sequence of training batches from the production dataloader is held constant across checkpoints, so that only the encoder varies. Per checkpoint we compute the variance ratio `VR = var_between / (var_within / batch_size)` (the spread of batch-mean features relative to the IID prediction; `VR = 1` under batch-IID), the lag autocorrelation `ρ(ℓ)` of the batch means, and the model drift `v` of a fixed probe set between checkpoints. A positive control re-orders the same tiles by slide; a projection-invariance check recomputes the verdict in a signature space.
Results (raw pre-norm CLS space):

```
iter   | VR   | var_between | var_within | rho(1)  | rho(2)  | rho(5)  | T_corr | v(20k-interval) | v(per-step) | h
-------+------+-------------+------------+---------+---------+---------+--------+-----------------+-------------+------
 20000 | 2.05 |   0.03031   |    3.786   | -0.0809 | -0.0548 | -0.0181 |   0    |      n/a        |    n/a      |
 40000 | 1.89 |   0.11127   |   15.075   | -0.0840 | -0.0399 | -0.0088 |   0    |     3.6110      |  0.000181   |
 60000 | 1.80 |   0.09642   |   13.698   | -0.0860 | -0.0338 | -0.0104 |   0    |     3.4318      |  0.000172   | 0.065
 80000 | 1.79 |   0.05308   |    7.612   | -0.0766 | -0.0358 | -0.0081 |   0    |     2.5901      |  0.000130   |
100000 | 1.77 |   0.02752   |    3.985   | -0.0720 | -0.0365 | -0.0079 |   0    |     1.5472      |  0.000077   |
120000 | 1.73 |   0.01956   |    2.888   | -0.0738 | -0.0385 | -0.0007 |   0    |     0.6379      |  0.000032   |
124000 | 1.73 |   0.01927   |    2.857   | -0.0733 | -0.0389 | -0.0005 |   0    |     0.0892      |  0.000022   |

```

Positive control (same tiles at 60k, two orderings):

```
ordering        |   VR   | rho(1)  | sigma_batch^2
----------------+--------+---------+--------------
IID (training)  |  1.80  | -0.086  |   0.09642
slide-grouped   | 21.22  | +0.482  |   1.05577

```

Projection-invariance (raw CLS vs reconstructed-signature space):

```
iter   | CLS-space VR | CLS rho(1) | sig-space VR | sig rho(1)
-------+--------------+------------+--------------+-----------
 40000 |     1.89     |  -0.0840   |     5.58     |  -0.1720
 60000 |     1.80     |  -0.0860   |     5.15     |  -0.1264
100000 |     1.77     |  -0.0720   |     5.15     |  -0.1264

```

Across every checkpoint the stream shows `VR ≈ 1.7–2.0` and `|ρ(1)| < 0.1` (negative) with `T_corr = 0` — no persistent batch-to-batch correlation. The positive control confirms the statistic is sensitive: slide-grouped ordering of the same tiles drives `VR` to ~21 and `ρ(1)` to +0.48. The verdict is invariant to the measurement space (`ρ(1)` remains small and negative in both raw and signature space; the larger `VR` magnitude in signature space is an artifact of the intervening L2-normalization, not positive correlation). Model drift `v` decays monotonically across the run, so the staleness bias also diminishes with training. This confirms the batch-IID condition under which FIFO keep-last-M is optimal, and the residual `VR > 1` reflects mild static heterogeneity across the corpus sources with no temporal persistence — which FIFO does not require to be absent.
Filling and warmup
Bank filling begins at `typicality_warmup_iters`, so the reservoir is populated from a converged representation. The representative prototypes R train from the start of the run (they are not gated by the warmup), so both `z` and R are settled by the time filling begins. Loss modulation begins once the bank is full.
Instrumentation
Bank health is logged during training: the fraction of low-peak (diffuse) bank signatures, the within-bank k-NN distance statistics `(μ, σ)`, and the fraction of the current batch scored below a low typicality threshold. These provide a direct, ongoing signal that the bank remains a faithful, discriminating sample of the data.
Hyperparameters

* `typicality_K_prime` — number of representative prototypes `K'` (default 256).
* `typicality_bank_size` — reservoir capacity `M` (default 8192).
* `typicality_k` — k-NN neighbour count for the density estimate (default 20).
* `typicality_modulation` — `adaptive_temp` or `weighted_loss`.
* `typicality_alpha` / `typicality_beta` — dampening strength for the two variants.
* `typicality_warmup_iters` — iterations before filling and modulation begin.
* `typicality_repr_lr` — learning rate for the representative-prototype optimizer.
The bank is a fixed-capacity FIFO reservoir; there is no replacement-fraction parameter.
