# Literature Review: Discovering Position-Invariant, Action-Carrying Latents

**Scope.** Approaches relevant to our open problem in the Stage-0 toy: a VQ-bottleneck
latent-action model whose discovered codes track the agent's *position* and under-use
the *action*. Organized by approach family, each grounded to our failure, with verified
citations and a ranked set of candidate next experiments. Compiled from five focused
surveys (latent-action models, controllability, motion discovery, object-centric world
models, invariance/equivariance). arXiv IDs were verified during the survey; a handful
flagged below were taken from search listings and should be re-checked before formal
citation.

---

## Where we are (the two obstacles)

Our pipeline (inverse model → VQ code → dynamics → predict EMA-teacher next-latent, with
VICReg) reproduces a **named failure mode of latent-action models** (LAPO → distractor/
position hijack). Our experiments (`experiments/0_synthetic_toy/subexperiments/`) isolated
*two distinct* obstacles:

1. **Position entanglement (largely addressed).** The code tracked absolute position
   (`NMI(code,position) 0.064` vs `action 0.013`). A translation-invariant *head* (Exp 7,
   global-avg-pool over a feature-map difference) failed because a strided 4×4 CNN isn't
   shift-equivariant and the 6-px move is *sub-feature-cell*. Reading the action from a
   **16×16** map (Exp 8) lifted `NMI(code,action)` to 0.36 and dropped `NMI(code,position)`
   to 0.044 — resolution was real.

2. **The action is under-used (the current fundamental obstacle).** Even at NMI 0.36, we
   measured that swapping the code changes the predicted next-latent by only **~1%** of the
   across-sample variation: the dynamics predicts the next frame almost entirely from the
   *current* state, because the next state is highly predictable from the current one
   (static scene + agent's current position). The action is a small, correctly-aimed nudge
   (it still cuts error ~18%) but is dwarfed by the predictable static content.

The literature splits along exactly these axes. Obstacle 1 is "make the representation/
readout position-invariant"; obstacle 2 is "make the action *necessary* for prediction."
Obstacle 2 is now the priority, so it is foregrounded.

---

## A. Latent-action models & the bottleneck (our exact setup)

- **LAPO — Learning to Act without Actions** (Schmidt & Jiang, ICLR 2024, [2312.10812]).
  Our architecture: IDM infers a latent action from (past, *future*) frames; FDM predicts
  the next frame from (past, latent) only. The restricted decoder + small bottleneck is
  what *should* force the latent to carry the transition. Its documented failure: when the
  next state is cheaply predictable, the latent encodes nuisance (our position).
- **Genie** (Bruce et al., ICML 2024, [2402.15391]). Same restricted-decoder idea **plus a
  tiny VQ codebook, |A|=8**, and discovered codes are *consistent across starting frames*
  (left/right/jump) — exactly the position-invariance we lack. The load-bearing lever: a
  codebook barely larger than the true action count cannot store position. **We use K=16
  for 4 actions — over-provisioned.**
- **LAPA** ([2410.11758], ICLR 2025), **ILPO** ([1805.07914], ICML 2019, forward-model
  selection over discrete latents), **VPT** ([2206.11795], NeurIPS 2022, the labeled-IDM
  contrast) round out the family.

**Confound papers — directly on our bug:**
- **What Do Latent Action Models Actually Learn?** ([2506.15691], 2025). A linear/PCA model:
  the latent absorbs the **highest-variance** frame-to-frame direction, which need not be
  the action. For us, translating a red blob over a plain background *is* the highest-
  variance change → the bottleneck grabs position. Prescribes augmentation, data cleaning,
  and an auxiliary action-prediction loss.
- **Latent Action Learning Requires Supervision in the Presence of Distractors / LAOM**
  ([2502.00379], ICML 2025). LAPO latents encode control-unrelated content under distractors;
  fixes: multi-step inverse, latent temporal-consistency (not pixel) loss, augmentation, and
  — most reliably — **~2.5% action labels → ~4.2×**. Our position is an action-correlated
  distractor; **labels are free in our synthetic toy.**
- **MaskLAM — Segment to Focus** ([2602.02259], 2026). Reweight the reconstruction loss by an
  agent mask so gradients flow only from the controllable region. We can build the mask
  trivially (we know the red pixels / frame difference).
- **Object-Centric Latent Action Learning** ([2502.09680], AAAI 2026). Slots → linear probe
  to pick the **agent slot** → infer the action from that slot only. Nearest published match
  to our failure; reports large gains under distractors.

---

## B. Making the action *necessary* (the fundamental fix)

These target obstacle 2 directly — stop the model leaning on the predictable current state.

- **C-SWM — Contrastive Structured World Models** (Kipf et al., ICLR 2020, [1911.12247]).
  Two transferable ideas: (i) model the transition as an **additive translation**
  `z + T(z,a) ≈ z_next` so the action is a relative delta decoupled from absolute state;
  (ii) a **contrastive (energy) loss instead of pixel/latent MSE** — copying `z_t` no longer
  minimizes the loss if negatives are nearby, so the model must use the action to
  discriminate. The single most relevant recipe for obstacle 2.
- **Contingency-Aware Exploration** ([1811.01483], ICLR 2019) and **AC-State / multi-step
  inverse** ([2207.08229], 2022): an inverse model that predicts the *action* localizes the
  controllable element and (AC-State) provably filters exogenous noise. Caveat for us:
  position *is* control-endogenous, so a 1-step inverse keeps it — combine with a delta /
  selectivity term.
- **Independently Controllable Factors** ([1703.07718], [1708.01289], [1802.09484]). The
  **selectivity** objective: each code should change a *single, consistent* direction of the
  latent. Operationalized for us: penalize the **variance, across positions, of the
  code-induced latent delta** `dynamics(z_t,k) − z_t` — directly attacks "the effect of a
  code depends on where the agent is." Cheap, label-free.
- **Denoised MDPs** ([2206.15477], ICML 2022). Factor the latent into controllable vs.
  uncontrollable; keep only the controllable signal. Suggests splitting `z` into a static-
  context sub-vector (conditions the prediction) and a controllable sub-vector (feeds the
  code), decorrelated.
- **Empowerment / conditional MI**: maximize `I(code; z_{t+1} | z_t)` while minimizing
  `I(code; z_t)`. The conditional-MI form is *exactly* "the code should explain the part of
  the next state the current state does not." (Empowerment: Klyubin et al. 2005; MI
  estimation e.g. [1810.05533].)
- **Action-Bisimulation** ([2403.16369], RLC 2024), **Disentangling Controllable/
  Uncontrollable factors** ([1804.06955]): bisimulation/interaction objectives that collapse
  uncontrollable variation; principled but heavier.
- **VICReg / Barlow Twins** ([2105.04906], [2103.03230]). We already run VICReg. Its
  covariance term can be repurposed: **decorrelate the action-code embedding from a detached
  `z_t`** (Barlow-style off-diagonal penalty) so the code carries info *not* in position.
  Near-free given our existing machinery.

---

## C. Localize-the-mover / self-supervised motion discovery

Our agent is the *only* mover, so motion isolates exactly the controllable object.

- **Self-supervised VOS by Motion Grouping** ([2104.07658], ICCV 2021): slot-attention over
  *flow* reconstructs the flow as layers; a 2-slot version separates mover from background
  for free. **Blind to no-op (flow=0).**
- **The Emergence of Objectness** ([2111.06394], NeurIPS 2021): "segment-flow" = pool flow
  offsets per region. The clearest blueprint: *segment the mover → pool its displacement →
  that displacement is the (position-invariant) action.*
- **Guess What Moves** ([2205.07844], BMVC 2022): learn a **single-frame** mover-localizer
  with motion as the *teacher* — solves the no-op/stationary case at inference.
- **Learning Features by Watching Objects Move** ([1612.06370], CVPR 2017): motion-segmented
  masks as free pseudo-labels for a single-frame detector. **Betrayed by Motion**
  ([2011.11630]) adds background registration + memory for object permanence.
- **SAVi** ([2111.12594], ICLR 2022): **optical-flow prediction target** makes slots latch
  onto movers; weak center-of-mass conditioning (free in our toy). **Learning to See by
  Moving** ([1505.01596]) — predict the inter-frame transformation, the ancestor of
  inverse-dynamics action learning.

**Lightest version for us:** a (non-learned) frame-difference localizes the agent; pool its
**center-of-mass displacement (Δx, Δy)** → feed that as the action. Position-invariant by
construction; trivial on the toy.

---

## D. Object-centric / structured world models (obstacle 1, principled)

- **C-SWM** ([1911.12247], also §B) — lightest credible object-centric build: small CNN →
  K sigmoid masks → per-slot MLP → additive-delta transition + contrastive loss, *no slot
  attention, no reconstruction*. Quantize the **Δz / action**, keep our VQ.
- **Slot Attention** ([2006.15055], NeurIPS 2020), **SAVi** ([2111.12594]), **OP3**
  ([1910.12827]), **SLATE** ([2110.11405]), **STEVE** ([2205.14065]) — slot encoders; STEVE's
  transformer decoder is documented to handle **static objects** (relevant to our static
  distractors). All are **finicky, hyperparameter-sensitive, color-cue-reliant** on sprite
  worlds — honest risk.
- **SPACE** ([2001.02407], ICLR 2020): explicit `z_where`/`z_what` split → define the action
  as **Δz_where** of the agent. Architectural position/appearance disentanglement.
- **SIMONe** ([2106.03849]): object (time-invariant) vs. frame (time-varying) latents — the
  conceptual cure: put position in a static latent, leave the action latent to carry motion.
- **DINOSAUR** ([2209.14860]): reconstruct DINO *features* not pixels to stop slots wasting
  capacity on background — a stabilizer if distractors get busy.

---

## E. Invariance / equivariance & explicit displacement (obstacle 1)

- **Making Convolutional Networks Shift-Invariant Again / BlurPool** ([1904.11486], ICML 2019)
  and **APS** ([2011.14214], CVPR 2021): strided CNNs alias under small shifts; anti-aliased
  downsampling restores (approx./exact-for-integer-shifts) shift-equivariance. Explains why
  Exp 7's 4×4 difference was garbage; a cheap drop-in to make the difference a clean delta.
  **Tiny-object anti-aliasing** ([2310.14221], *author list unverified*) is the closest
  analogue — small-object motion lives in high frequencies that downsampling destroys.
- **Phase correlation / soft-argmax displacement**: **PCDNet** ([2110.03473]), **DPCN++**
  ([2206.05707], *search-listing*). Estimate the **translation** between two feature maps via
  the cross-correlation peak, made differentiable + **sub-pixel** by soft-argmax. Translation-
  equivariant *by construction*; recovers the displacement directly instead of pooling it
  away. **The literature's predicted ceiling-raiser over global-avg-pool.**
- **RAFT** ([2003.12039], ECCV 2020) / **SelFlow** ([1904.09117]): dense flow via all-pairs
  correlation at input resolution; a pooled agent-flow vector is a clean action. Heavier;
  phase correlation is the minimal version.
- **STN** ([1506.02025]) / **Learned Canonicalization** ([2211.06489]) / **G-CNNs**
  ([1602.07576]): register-to-canonical-pose or build-in the symmetry; theory anchors.

---

## F. Prediction targets & anti-collapse (what we predict into)

- **V-JEPA 2** ([2506.09985], 2025), **I-JEPA** ([2301.08243]): EMA-teacher latent prediction
  (our recipe). V-JEPA 2-AC's **two-stage** split — learn the action-free representation
  first, then a *separate* action-conditioned head — is a near-free wiring lesson: learn the
  code against a frozen latent so it can't co-adapt with the position-carrying encoder.
- **DINO** ([2104.14294]) / **Temporal-DINO** ([2308.04589]): EMA self-distillation; DINO's
  emergent foreground localization is the property our generic CNN lacks.
- **VICReg** ([2105.04906]) / **Barlow Twins** ([2103.03230]): see §B — VICReg fixed our
  Exp-3 collapse but *keeps position alive* (it's the dominant variance), so it must be paired
  with a decorrelate-from-position term.

---

## Synthesis — candidate next experiments (ranked)

Two axes: **(I) make the action necessary** (the current obstacle) and **(II) position-
invariance** (mostly handled by Exp 8's higher-res; finish with anti-aliasing/displacement).
All are measurable against our existing `NMI(code,action)` / `NMI(code,position)` / no-action-
gap diagnostics. Cheapest first.

| # | Experiment | Axis | Effort | Why / evidence |
|---|---|---|---|---|
| 1 | **Shrink the codebook** K=16 → ~6 | I/II | trivial | Genie's |A|=8; less capacity to store position; forces transition-indexing |
| 2 | **Action supervision on ~2.5% of transitions** (CE tying VQ code → known L/R/U/D) | I | trivial (labels free) | LAOM: +4.2× under exactly our distractor failure |
| 3 | **Selectivity / decorrelate-code-from-position loss** (penalize variance of `dynamics(z_t,k)−z_t` across positions; Barlow off-diagonal vs detached `z_t`) | I | cheap (reuse VICReg) | ICF selectivity; VICReg/Barlow; targets the root cause label-free |
| 4 | **Contrastive next-latent prediction** (C-SWM energy; the true next vs negatives, ideally same-`z_t`-different-action — generatable in our toy) | I | medium | C-SWM; copying `z_t` stops winning → action becomes necessary |
| 5 | **Anti-aliased downsampling** (BlurPool/APS) + keep 16×16 action features | II | cheap | makes the feature-difference a clean delta; finishes Exp 8 |
| 6 | **Explicit displacement readout** (soft-argmax cross-correlation between the two feature maps → (Δx,Δy) → VQ) | I+II | medium | PCDNet/phase-corr; translation-equivariant, sub-pixel; raises the pooling ceiling |
| 7 | **Localize-the-mover front end** (frame-diff mask → pool agent center-of-mass displacement) | I+II | cheap (toy) | Emergence-of-Objectness "segment-flow"; near-trivial here |
| 8 | **Object-centric / agent-slot encoder** (mini-C-SWM additive `z+Δz`, or slots→pick-agent) | I+II | heavy | the principled build; do once cheap levers plateau |

**Recommendation.** The fundamental problem (the action is optional because the next state
is predictable from the current state) is best attacked first by **#1+#2+#3 together** (all
near-free, directly make the code matter and decouple it from position) and, if they plateau,
**#4 contrastive prediction** (the most principled "stop copying `z_t`" fix). #6 (explicit
displacement) is the strongest single architectural change and doubles as the obstacle-1 fix.

---

## Bibliography (verified arXiv IDs)

Latent action: LAPO 2312.10812 · Genie 2402.15391 · LAPA 2410.11758 · ILPO 1805.07914 ·
VPT 2206.11795 · What-Do-LAMs-Learn 2506.15691 · LAOM 2502.00379 · MaskLAM 2602.02259 ·
Object-Centric Latent Action 2502.09680.
Controllability: C-SWM 1911.12247 · Contingency-Aware 1811.01483 · AC-State 2207.08229 ·
ICF 1703.07718 / 1708.01289 / 1802.09484 · Denoised MDPs 2206.15477 · Action-Bisimulation
2403.16369 · Disentangling Controllable/Uncontrollable 1804.06955 · Empowerment-MI 1810.05533
(concept: Klyubin et al. 2005) · Learning to Poke 1606.07419.
Motion discovery: Motion Grouping 2104.07658 · Emergence of Objectness 2111.06394 · Guess
What Moves 2205.07844 · Betrayed by Motion 2011.11630 · Relaxed Common Fate 2304.08025 ·
Common Fate (Tangemann) 2110.06562 · Learning Features by Watching Objects Move 1612.06370 ·
SIMONe 2106.03849 · Learning to See by Moving 1505.01596 · TCN 1704.06888.
Object-centric: Slot Attention 2006.15055 · SAVi 2111.12594 · SAVi++ 2206.07764 · OP3
1910.12827 · SLATE 2110.11405 · STEVE 2205.14065 · SPACE 2001.02407 · SOLD 2410.08822 ·
DINOSAUR 2209.14860.
Invariance/displacement: BlurPool 1904.11486 · APS 2011.14214 / 2105.04040 · tiny-object
anti-aliasing 2310.14221 · G-CNN 1602.07576 · STN 1506.02025 · Learned Canonicalization
2211.06489 · PCDNet 2110.03473 · DPCN++ 2206.05707 · RAFT 2003.12039 · SelFlow 1904.09117.
SSL targets: V-JEPA 2 2506.09985 · I-JEPA 2301.08243 · DINO 2104.14294 · Temporal-DINO
2308.04589 · VICReg 2105.04906 · Barlow Twins 2103.03230.

**Verification caveats.** IDs were confirmed by the surveying agents via arXiv fetch or
corroborated snippets. Re-check before formal citation: tiny-object anti-aliasing
(2310.14221, author list), DPCN++ (2206.05707), VideoPCDNet (2506.19621), the motion-seg set
(2304.01430, 2308.11239), and the exact metrics in Object-Centric Latent Action (2502.09680).
Genie 2/3 are blog posts (no methods paper). Slow Feature Analysis (Wiskott & Sejnowski 2002)
is pre-arXiv. Empowerment originates with Klyubin et al. 2005.
