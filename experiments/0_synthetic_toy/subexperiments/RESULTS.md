# Results — Stage-0 Synthetic Toy

Procedural sprite toy (one red agent moving L/R/U/D + static distractors), K=16, 5000 steps/run on `bench`. Metrics/figures are regenerated from each fetched checkpoint on the held-out val set (seed+1).

| Run | loss / head | wandb |
|---|---|---|
| Baseline | prediction + VQ (pixel) | [jzj52ot1](https://wandb.ai/rva_nsr/ssa/runs/jzj52ot1) |
| + margin + usage | + no-action margin (m=0.002) + usage (w=0.1), pixel | [1saxa1eg](https://wandb.ai/rva_nsr/ssa/runs/1saxa1eg) |
| + delta | margin + usage, **delta head** (predict `I_{t+1}−I_t`) | offline (regenerated from checkpoint) |

## Headline

| Metric (val) | Baseline | + margin + usage | + delta | Target |
|---|---|---|---|---|
| codes used / perplexity | 1 / 1.00 | 7 / 5.60 | **10 / 9.11** | non-collapsed |
| no-action gap | 3.2e-6 | 2.1e-4 | **7.6e-7** | clearly > 0 |
| NMI(code, action) | 0.00 | 0.003 | **0.008** | > 0.8 |
| pixel / delta val MSE | 0.0083 | 0.0096 | 0.0117 | — |

**The usage loss reliably breaks codebook collapse (1 → 7 → 10 codes), but nothing yet makes the action necessary or semantic.** The no-action gap stays ~0 (delta even drove it back to ~0), and NMI stays ~0 across all three. Predicting the *change* instead of the frame did **not** raise the action's leverage — my hypothesis was wrong, which is the useful finding here.

## Baseline (prediction + VQ): total failure — as predicted

One code carries every assignment (perplexity 1.0); applying any of the 16 codes to a fixed frame yields identical predictions — the action is ignored.

![baseline codebook usage](codebook_usage.png)
![baseline counterfactual](counterfactual.png)

The toy's future is largely predictable without the action (`I_{t+1} ≈ I_t`; a blurry near-static prediction already gets low pixel MSE), so the dynamics has no pressure to use `a_t`, and nothing stops VQ collapse.

## + margin + usage: collapse broken, but action only weakly necessary, codes not semantic

7 of 16 codes used (perplexity 5.6 — usage works). The no-action gap goes positive (2.1e-4 vs 3e-6) but is tiny, and the used codes spread roughly uniformly across all four actions — no code is action-selective (NMI ~0):

![full code-action confusion](full_losses/code_action_confusion.png)

## + delta prediction: collapse broken further, but the action is *still* ignored

Predicting the residual `I_{t+1}−I_t` (so the action ought to explain the *change*) does **not** help. Usage spreads further (10 codes, perplexity 9.1), but the no-action gap falls back to ~0 (7.6e-7) — the dynamics predicts the change just as well with a zero action as with the inferred one. The counterfactual makes it visual: reconstructing `I_t + residual` for every code leaves the agent in the **same position** in all 16 panels — no code produces a move:

![delta counterfactual](delta/counterfactual.png)
![delta code-action confusion](delta/code_action_confusion.png)

## Interpretation (revised)

The earlier guess — that frame-vs-delta was the bottleneck — was wrong. Across all three runs the real obstacle is the same: **the model predicts the (easy) future without using the action, and MSE rewards a near-static / blurry prediction**, so there is almost no gradient making `a_t` necessary. The weak no-action margin (m=0.002) cannot overcome this, and delta prediction does not change it (a near-zero residual is still MSE-cheap). Usage reliably prevents collapse, but diverse-but-ignored codes are not action codes.

Note the margin loss is also **gameable**: it can be satisfied by *degrading* the zero-action prediction rather than by *using* the action — so simply increasing `m` is not obviously the fix.

## Next levers (in priority order)

1. **Predict teacher latents, not pixels (V-JEPA style).** This is the design doc's first recommendation and directly attacks the root cause — pixel/delta MSE rewards the blur that lets the action be ignored; predicting an EMA-teacher's latent of `I_{t+1}` penalizes the hedge. The `LatentHead` is already built and tested for exactly this pivot (watch for representational collapse).
2. **Make the action genuinely necessary** — a low-bandwidth context and/or a harder future where a static prediction is bad, so the action carries real information; and a stronger but non-gameable necessity signal.
3. **Contrastive action loss** — push similar transitions to share codes, structuring the codebook by transition type directly rather than hoping prediction induces it.

Verdict vs. the Stage-0 criterion (NMI > 0.8 + clear no-action gap): **not met.** Confirmed wins: usage breaks collapse. Confirmed negatives: the no-action margin (frame or delta) does not make the action necessary on this toy. The evidence now points at latent-space prediction as the next experiment.
