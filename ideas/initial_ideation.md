**You:**

I want to build a self supervised action discovery model.

Take images I_{t-k} .. I{t}. Encode them through encoder e_s (student). Then feed them through latent action model f to get latent action a. Then feed them through dynamics model g(e_t(I_{t-k} .. I_{t}), a) -> I_{t+1} . 

Somehow we have to make a matter the most to get it to discover actions

* * *

**ChatGPT:**

I’ll treat this as a model-design problem: define what “action” should mean, identify why the latent may be ignored, then propose losses/architectures that force the latent action to capture controllable change rather than appearance.

The key issue: in your current setup, **the dynamics model can ignore `a`** if the encoded history already contains enough information to predict `I_{t+1}`. To make `a` discover actions, you need to make it the **minimal variable that explains controllable transition information**.

A cleaner formulation:

$$z_i^s = e_s(I_i), \quad z_i^t = e_t(I_i)$$ $$a_t = f_\phi(z_{t-k:t}^s, z_{t+1}^t)$$ $$\hat z_{t+1} = g_\theta(z_{t-k:t}^s, a_t)$$

Train against the teacher future latent:

$$\mathcal L_{\text{pred}} = \| \hat z_{t+1} - \text{sg}(z_{t+1}^t) \|$$

I would predict **future latent**, not raw image, at least initially. Pixel prediction makes the model spend capacity on texture, lighting, background, and stochastic details. JEPA-style video models use latent-space prediction for this reason; V-JEPA predicts masked spatiotemporal regions in representation space, and V-JEPA 2 extends this toward action-conditioned planning/world models. [![](https://www.google.com/s2/favicons?domain=https://openreview.net&sz=128)OpenReview+1](https://openreview.net/forum?id=WFYbBOEOtv&utm_source=chatgpt.com)

## The core objective

You want:

$$a_t \approx \text{what changed because of an action}$$

not:

$$a_t \approx \text{the whole future frame}$$

So impose three pressures:

1. **`a_t` must be necessary**
2. **`a_t` must be minimal**
3. **`a_t` must explain change, not static scene content**

A practical loss:

$$\mathcal L =  
\mathcal L_{\text{pred}}  
+ \lambda_{\text{bottleneck}}\mathcal L_{\text{bottleneck}}  
+ \lambda_{\text{contrast}}\mathcal L_{\text{action-contrast}}  
+ \lambda_{\text{cycle}}\mathcal L_{\text{cycle}}  
+ \lambda_{\text{diverse}}\mathcal L_{\text{usage}}$$

## 1. Make `a_t` necessary: action dropout / context bottleneck

Do **not** give `g` an overpowered history encoder. If `g(z_{t-k:t})` can predict the future alone, `a_t` becomes optional.

Use an ablation loss:

$$\hat z_{t+1}^{\text{no-action}} = g(z_{t-k:t}, 0)$$ $$\mathcal L_{\text{necessity}} =  
\max(0, m + \mathcal L_{\text{pred}}(g(z_{t-k:t}, a_t)) -  
\mathcal L_{\text{pred}}(g(z_{t-k:t}, 0)))$$

This says: prediction with the inferred action must be better than prediction without it by margin `m`.

Also constrain context:

$$c_t = h(z_{t-k:t})$$

but make `c_t` low bandwidth. For example:

$$\hat z_{t+1} = g(c_t, a_t)$$

where `c_t` is scene/state and `a_t` is transition. If `c_t` is too rich, `a_t` collapses.

## 2. Make `a_t` minimal: information bottleneck

Use a discrete or low-dimensional latent action.

Good options:

### Option A: VQ action code

$$a_t = \text{VQ}(f(z_{t-k:t}, z_{t+1}))$$

with codebook size like 32, 64, 128, or 256.

This encourages the model to discover a finite action vocabulary: push, pull, open, close, move-left, move-right, grasp-like transitions, etc.

Loss:

$$\mathcal L_{\text{VQ}} =  
\|\text{sg}(a_e) - a_q\|^2  
+  
\beta \|a_e - \text{sg}(a_q)\|^2$$

### Option B: Gaussian bottleneck

$$a_t \sim q_\phi(a \mid z_{t-k:t}, z_{t+1})$$

with:

$$\mathcal L_{\text{KL}} =  
D_{\text{KL}}(q_\phi(a \mid z_{t-k:t}, z_{t+1}) \| p(a))$$

This is weaker than VQ if you want crisp action discovery, but useful if actions are continuous.

## 3. Make `a_t` explain change, not appearance

Use a delta target:

$$\Delta z_t = z_{t+1} - z_t$$

Then structure the model as:

$$\widehat{\Delta z}_t = g(c_t, a_t)$$ $$\hat z_{t+1} = z_t + \widehat{\Delta z}_t$$

This encourages `a_t` to encode transition information.

Even better, make `a_t` predict **object-centric deltas**:

$$z_t = \{o_t^1, o_t^2, ..., o_t^N\}$$ $$a_t = \{a_t^1, ..., a_t^N\}$$ $$\hat o_{t+1}^i = o_t^i + g(o_t^i, a_t^i, c_t)$$

Object-centric latent action models are promising because actions usually affect a small subset of objects, not the whole frame. Recent object-centric latent action/world-model work also argues that object-centric structure helps discover controllable dynamics and avoid distractors. [![](https://www.google.com/s2/favicons?domain=https://openreview.net&sz=128)OpenReview+1](https://openreview.net/forum?id=lTaPtGiUUc&utm_source=chatgpt.com)

## 4. Add contrastive pressure on actions

You want same transition type → similar `a`, different transition type → different `a`.

Construct positives and negatives without labels.

### Positive pairs

Two clips are positives if they have similar state change:

$$\Delta z_i \approx \Delta z_j$$

For example, nearest neighbors in delta-latent space.

Then:

$$\mathcal L_{\text{action-contrast}}  
=  
-\log  
\frac{  
\exp(\text{sim}(a_i, a_j^+)/\tau)  
}{  
\sum_j \exp(\text{sim}(a_i, a_j)/\tau)  
}$$

This prevents arbitrary per-video action codes.

### Negative pairs

Use clips with similar current state but different future:

$$z_t^i \approx z_t^j, \quad z_{t+1}^i \not\approx z_{t+1}^j$$

These should have different actions.

This is important: actions are most identifiable when the same state can lead to different futures.

## 5. Add an inverse-forward cycle

Use inverse model:

$$a_t = f(z_t, z_{t+1})$$

Forward model:

$$\hat z_{t+1} = g(z_t, a_t)$$

Then re-infer:

$$\hat a_t = f(z_t, \hat z_{t+1})$$

Cycle loss:

$$\mathcal L_{\text{cycle}} = \| \hat a_t - a_t \|$$

This encourages `a_t` to be a stable latent transition descriptor.

For discrete actions, use cross-entropy between original and reconstructed code.

## 6. Prevent code collapse

If `a` is discrete, the model may use only one or two codes. Add a usage regularizer:

$$\mathcal L_{\text{usage}} =  
- H(\bar p(a))$$

where:

$$\bar p(a) = \frac{1}{B}\sum_{i=1}^B p(a_i)$$

Minimizing `-H` maximizes batch-level entropy, encouraging all codes to be used.

But also avoid making every sample random. Use:

$$\mathcal L_{\text{commit}} = H(p(a_i))$$

So each individual sample has low entropy, while the batch distribution has high entropy:

$$\mathcal L_{\text{discrete}} =  
\lambda_1 H(p(a_i)) - \lambda_2 H(\bar p(a))$$

Meaning:

* each transition picks a clear action;
* the dataset uses many actions.

## 7. Use stop-gradients carefully

A common failure mode:

$$e_s, f, g$$

collude to put arbitrary future information into `a`.

Use a teacher encoder:

$$e_t \leftarrow \text{EMA}(e_s)$$

and stop gradient on targets:

$$\mathcal L_{\text{pred}} =  
\|g(e_s(I_{t-k:t}), a_t) - \text{sg}(e_t(I_{t+1}))\|$$

This follows the general student/teacher pattern used in modern self-supervised video representation learning. Temporal-DINO, for example, uses a student that sees past frames and a teacher with broader temporal context to provide a future-aware target. [![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=128)CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023W/ROAD%2B%2B/papers/Teeti_Temporal_DINO_A_Self-Supervised_Video_Strategy_to_Enhance_Action_Prediction_ICCVW_2023_paper.pdf?utm_source=chatgpt.com)

## Recommended architecture

I would build it like this:

```
Images:
  I_{t-k}, ..., I_t, I_{t+1}

Student encoder:
  z_{t-k:t} = e_s(I_{t-k:t})

Teacher encoder:
  z_{t+1}^T = stopgrad(e_t(I_{t+1}))

State encoder:
  c_t = h(z_{t-k:t})

Inverse latent action model:
  a_t = VQ(f(c_t, z_{t+1}^T))

Dynamics:
  z_hat_{t+1} = g(c_t, a_t)

Losses:
  L_pred       = ||z_hat_{t+1} - z_{t+1}^T||
  L_no_action  = margin loss forcing a_t to help
  L_vq         = VQ commitment/codebook loss
  L_usage      = high batch entropy, low sample entropy
  L_contrast   = similar deltas -> similar actions
  L_cycle      = f(c_t, z_hat_{t+1}) ≈ a_t
```

## More precise training objective

$$a_t = \text{VQ}(f_\phi(c_t, z_{t+1}^T))$$ $$\hat z_{t+1} = g_\theta(c_t, a_t)$$ $$\mathcal L_{\text{pred}} =  
\|\hat z_{t+1} - z_{t+1}^T\|_2^2$$ $$\mathcal L_{\text{margin}} =  
\max(0, m +  
\|\hat z_{t+1} - z_{t+1}^T\|_2^2  
-  
\|g(c_t, 0) - z_{t+1}^T\|_2^2)$$ $$\mathcal L_{\text{usage}} =  
\frac{1}{B}\sum_i H(p(a_i)) - H(\frac{1}{B}\sum_i p(a_i))$$ $$\mathcal L =  
\mathcal L_{\text{pred}}  
+ \lambda_m \mathcal L_{\text{margin}}  
+ \lambda_v \mathcal L_{\text{VQ}}  
+ \lambda_u \mathcal L_{\text{usage}}  
+ \lambda_c \mathcal L_{\text{contrast}}$$

## Important correction to your original setup

You wrote:

$$g(e_t(I_{t-k} .. I_t), a) \rightarrow I_{t+1}$$

I would not use the teacher encoder as the input to dynamics. Use:

$$g(e_s(I_{t-k:t}), a) \rightarrow e_t(I_{t+1})$$

The teacher should provide the prediction target, not the input path. Otherwise the student does not learn the correct predictive representation.

## What makes `a` “matter most”

The most important tricks are:

1. **Do not predict pixels at first. Predict teacher latents.**
2. **Do not let the history encoder fully predict the future without `a`.**
3. **Use a small discrete bottleneck for `a`.**
4. **Train with a no-action counterfactual loss.**
5. **Make `a` represent deltas/change, not the whole future.**
6. **Use batch-level entropy to avoid code collapse.**
7. **Use contrastive pairs so similar transitions share action codes.**

A minimal strong baseline would be:

$$a_t = \text{VQ}(f(e_s(I_t), e_t(I_{t+1})))$$ $$\hat z_{t+1} = g(e_s(I_t), a_t)$$

with:

$$\mathcal L =  
\|\hat z_{t+1} - \text{sg}(e_t(I_{t+1}))\|^2  
+  
\lambda \mathcal L_{\text{VQ}}  
+  
\alpha \mathcal L_{\text{margin}}  
+  
\beta \mathcal L_{\text{usage}}$$

That is the simplest version I would implement first.

* * *

**You:**

what datasets should I use? what are other things I need to consider?

I want to follow research using this blog post:

https://www.vincentsitzmann.com/blog/on_research/

* * *

**ChatGPT:**

I’ll ground this in the research process from the blog post and map it to concrete dataset choices, evaluation design, and failure modes for self-supervised latent action discovery.

You should structure this as a **two-mode project**, matching Sitzmann’s blog: first make it work on the smallest faithful toy setting, then switch to proving it works with ablations, baselines, and harder datasets. The blog’s main advice is to find a simple toy problem, iterate quickly, time-box the proof-of-concept, then become skeptical: refactor, ablate, compare side-by-side, add baselines, and scale the setting only after the toy version works. [![](https://www.google.com/s2/favicons?domain=https://www.vincentsitzmann.com&sz=128)Vincent Sitzmann](https://www.vincentsitzmann.com/blog/on_research/)

## Dataset progression I would use

### Stage 0: synthetic toy dataset you generate yourself

Start here. Do **not** start with Ego4D, Something-Something, or Open X-Embodiment.

Create a simple 2D environment:

```
Objects:
  colored shapes on a plain background

Latent actions:
  move-left, move-right, move-up, move-down
  rotate
  open/close
  push object A into object B
  pick-and-place-like teleport / drag

Observation:
  image sequence I_{t-k:t+1}

Training:
  hide the true action labels from the model

Evaluation:
  compare discovered latent action clusters to ground-truth hidden actions
```

This is the “make it work” dataset. It lets you test whether `a_t` actually becomes a transition/action variable rather than an image-compression variable.

You should require the model to solve:

$$z_{t+1} = g(z_t, a_t)$$

where `a_t` clusters according to the hidden synthetic action. If it cannot recover actions here, it will not recover them in real video.

### Stage 1: BAIR robot pushing

Use **BAIR robot pushing** as the first real dataset. It is small enough to iterate on and directly matches your setup: image sequences with robot-object interactions. TensorFlow Datasets lists `bair_robot_pushing_small` as roughly **44,000 examples** with train, seen-object test, and novel-object test splits. [![](https://www.google.com/s2/favicons?domain=https://www.tensorflow.org&sz=128)TensorFlow](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small?utm_source=chatgpt.com)

Why it is good:

* Simple visual domain.
* Physical interactions.
* Mostly single-arm pushing.
* Has enough repeated structure for latent actions to cluster.
* Has object generalization splits.

What to test:

* Do discovered actions correspond to push direction?
* Do they separate contact vs no-contact?
* Do they separate large object motion vs small object motion?
* Does `a_t` transfer to novel objects?

This is probably the best first “real” dataset.

### Stage 2: RoboNet

After BAIR, move to **RoboNet**. RoboNet has over **15 million video frames** from robot-object interaction and multiple robot platforms/camera viewpoints; TFDS describes action vectors as end-effector position/rotation deltas plus a gripper dimension. [![](https://www.google.com/s2/favicons?domain=https://www.tensorflow.org&sz=128)TensorFlow](https://www.tensorflow.org/datasets/catalog/robonet?utm_source=chatgpt.com)

Why it is useful:

* Larger scale than BAIR.
* Multiple viewpoints.
* Multiple robots.
* Real interaction data.
* Ground-truth robot actions exist, so you can evaluate whether your discovered `a_t` aligns with actual end-effector deltas.

Use it for “prove it generalizes” experiments.

Good evaluation:

$$\text{linear probe}(a_t) \rightarrow \Delta x, \Delta y, \Delta z, \Delta \text{gripper}$$

If a linear probe on `a_t` predicts true robot action better than baselines, that is strong evidence that `a_t` captures action-relevant structure.

### Stage 3: BridgeData V2

Use **BridgeData V2** once the method is stable. It contains **60,096 trajectories** across **24 environments** and is designed for scalable robot manipulation with task/environment variability. [![](https://www.google.com/s2/favicons?domain=https://rail-berkeley.github.io&sz=128)rail-berkeley.github.io+1](https://rail-berkeley.github.io/bridgedata/?utm_source=chatgpt.com)

Why it matters:

* More diverse manipulation than BAIR.
* Goal-conditioned and language-conditioned use cases.
* Better test of whether latent actions become reusable skills rather than dataset-specific motion codes.

Good questions:

* Do latent actions cluster into semantic manipulation primitives?
* Are the same codes reused across objects/environments?
* Can you compose latent actions for planning?
* Does the latent action vocabulary help downstream imitation learning?

### Stage 4: DROID

Use **DROID** for large-scale real-world robustness. DROID has about **76k demonstration trajectories**, **350 hours** of interaction data, **564 scenes**, and **86 tasks**, collected by many data collectors across locations. [![](https://www.google.com/s2/favicons?domain=https://droid-dataset.github.io&sz=128)droid-dataset.github.io](https://droid-dataset.github.io/?utm_source=chatgpt.com)

Why it is useful:

* Real-world diversity.
* Multiple scenes and tasks.
* Better stress test for spurious visual correlations.
* Good benchmark for whether latent actions remain meaningful outside a controlled lab setting.

This is not where you should start. It is where you go after your losses, ablations, and metrics are settled.

### Stage 5: Open X-Embodiment

Use **Open X-Embodiment** only after you have a strong method. It contains **1M+ real robot trajectories** across **22 robot embodiments**, pooled from many robot datasets. [![](https://www.google.com/s2/favicons?domain=https://robotics-transformer-x.github.io&sz=128)Robotics Transformer X](https://robotics-transformer-x.github.io/?utm_source=chatgpt.com)

This is the scale-up dataset. It is valuable if your claim becomes:

> A single self-supervised latent action model can discover reusable action abstractions across embodiments.

But it introduces hard problems:

* Different action spaces.
* Different cameras.
* Different control frequencies.
* Different robots.
* Different task distributions.
* Different data quality.

Do not use it for your first proof-of-concept.

### Stage 6: Something-Something V2 for human-object actions

Use **Something-Something V2** if you want to show that your method is not robot-specific. It has about **220,847 video clips** over **174 action categories** involving human-object interactions. [![](https://www.google.com/s2/favicons?domain=https://www.qualcomm.com&sz=128)Qualcomm+1](https://www.qualcomm.com/developer/software/something-something-v-2-dataset?utm_source=chatgpt.com)

This dataset is useful because the labels are fine-grained action verbs:

* pushing left vs right;
* opening vs closing;
* putting into vs taking out;
* pretending vs actually causing change.

But it is harder because there are no robot actions and there is large human/object/background variation. Use the labels only for evaluation, not training.

### Stage 7: Ego4D / Ego-Exo4D for long-horizon human activity

Use these only for later-stage validation. Ego4D is very large, with over **3,700 hours** of first-person video, while Ego-Exo4D adds synchronized egocentric and exocentric views and action descriptions for skilled activities. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/facebookresearch/Ego4d?utm_source=chatgpt.com)

These are useful if your goal becomes general video action discovery, but they are too unconstrained for early research.

## Recommended dataset order

I would use this exact order:

```
1. Synthetic shapes / toy physical environment
2. BAIR robot pushing
3. RoboNet
4. BridgeData V2
5. DROID
6. Something-Something V2
7. Open X-Embodiment
8. Ego4D / Ego-Exo4D
```

For a first paper-style project, I would probably stop at:

```
Synthetic → BAIR → RoboNet or BridgeData → Something-Something V2
```

That gives you controlled proof, robot evaluation, and human-video evaluation.

## What you need to consider

### 1. What exactly is an “action”?

You need to define this before designing metrics.

Possible definitions:

**Transition action**

$$a_t = \text{minimal variable needed to map } z_t \rightarrow z_{t+1}$$

**Controllable action**

$$a_t = \text{latent variable corresponding to agent-induced change}$$

**Semantic action**

$$a_t = \text{human-interpretable category like push, pull, open, close}$$

**Motor action**

$$a_t = \text{low-level end-effector delta or joint command}$$

These are not the same. Your model may discover transition codes that are predictive but not semantic. That can still be useful, but your claims must match your evidence.

### 2. Avoid the “future information leak”

If `f` gets `I_{t+1}` and `a_t` has high capacity, it can encode the entire future.

Then:

$$a_t \approx z_{t+1}$$

That is not action discovery.

Mitigations:

* use a small discrete codebook;
* add an information bottleneck;
* predict deltas instead of full future latents;
* add no-action counterfactual loss;
* evaluate whether `a_t` generalizes across different states;
* prevent `a_t` from carrying static object/background identity.

### 3. Avoid the “history explains everything” failure

If the history encoder sees enough frames, it may infer velocity and predict the future without `a_t`.

Then:

$$g(z_{t-k:t}, a_t) \approx g(z_{t-k:t}, 0)$$

Your loss should explicitly penalize this.

Use:

$$\mathcal L_{\text{necessity}}  
=  
\max(0, m +  
\mathcal L(g(c_t,a_t), z_{t+1})  
-  
\mathcal L(g(c_t,0), z_{t+1}))$$

This forces the action to matter.

### 4. Use datasets where multiple futures are possible from similar states

Action discovery is easiest when:

$$z_t^i \approx z_t^j$$

but:

$$z_{t+1}^i \neq z_{t+1}^j$$

That means the same state can lead to different outcomes because of different actions.

Bad datasets:

* mostly passive camera motion;
* deterministic natural dynamics;
* videos where the future is obvious from momentum;
* datasets where each object/context implies only one likely action.

Good datasets:

* robot manipulation;
* human-object interaction;
* repeated scenes with different interventions.

### 5. You need evaluation labels, even if training is self-supervised

Train without action labels. Evaluate with labels where available.

Useful evaluations:

#### Cluster purity

Cluster `a_t`, compare to ground-truth action labels.

$$\text{NMI}(a, y), \quad \text{ARI}(a, y)$$

Use for synthetic and Something-Something.

#### Linear probe to robot action

Train:

$$\hat u_t = W a_t$$

where `u_t` is true robot action.

Use for BAIR, RoboNet, BridgeData, DROID, Open X-Embodiment when action metadata is available.

#### Counterfactual prediction

Swap actions:

$$\hat z_{t+1}^{i \leftarrow j} = g(z_t^i, a_t^j)$$

Then visually inspect whether the model applies the other transition to the current state.

This is one of the most important qualitative tests.

#### Planning utility

Use latent actions in model-predictive control:

$$a_{1:H}^* = \arg\min_{a_{1:H}} d(g^H(z_t, a_{1:H}), z_{\text{goal}})$$

If the discovered actions help planning, that is strong evidence they are useful.

#### Retrieval

Given a transition, retrieve other transitions with the same `a_t`.

You want nearest neighbors in action space to share **motion/interaction type**, not object identity or background.

### 6. Baselines you need

Following the “prove it works” phase from the blog, you need boring baselines. [![](https://www.google.com/s2/favicons?domain=https://www.vincentsitzmann.com&sz=128)Vincent Sitzmann](https://www.vincentsitzmann.com/blog/on_research/)

Minimum baselines:

```
1. No-action model:
   z_hat_{t+1} = g(z_{t-k:t})

2. Random action code:
   a_t ~ categorical uniform

3. Delta latent clustering:
   k-means on z_{t+1} - z_t

4. Optical-flow clustering:
   k-means on flow features

5. Inverse model without bottleneck:
   a_t = f(z_t, z_{t+1}), continuous high-dimensional

6. VQ-VAE transition tokenizer:
   discretize transition latents directly

7. Supervised oracle upper bound:
   train using true robot actions or action labels
```

The most dangerous baseline is probably:

$$a_t = \text{k-means}(z_{t+1} - z_t)$$

If your method does not beat this, your novelty may be weak.

### 7. Strong ablations

Ablate:

```
- no VQ bottleneck
- no usage entropy
- no no-action margin loss
- no contrastive action loss
- no cycle consistency
- predict pixels instead of latents
- predict z_{t+1} instead of delta z
- small vs large action codebook
- short vs long history
- frozen encoder vs trained encoder
- teacher EMA vs no teacher
```

You should expect some of your proposed components not to matter. That is normal. The blog explicitly warns that the thing you think matters may not be the thing that actually matters. [![](https://www.google.com/s2/favicons?domain=https://www.vincentsitzmann.com&sz=128)Vincent Sitzmann](https://www.vincentsitzmann.com/blog/on_research/)

### 8. Codebook size matters

Try:

```
K = 8, 16, 32, 64, 128, 256
```

Expected behavior:

* too small: merges distinct actions;
* too large: memorizes visual details;
* just right: captures reusable transition primitives.

For BAIR, start with `K = 32` or `64`.

For Something-Something, start with `K = 128` or `256`.

For Open X-Embodiment, you may need hierarchical codes.

### 9. Temporal abstraction matters

Not all actions are single-frame transitions.

You may need:

$$a_{t:t+\tau} = f(z_t, z_{t+\tau})$$

Try horizons:

```
τ = 1, 2, 4, 8, 16 frames
```

Short horizons discover motor primitives: nudge left, move down, close gripper.

Long horizons discover semantic actions: pick up, open drawer, pour, place.

A good model may need two levels:

$$a_t^{\text{low}} = \text{motor primitive}$$ $$A_t^{\text{high}} = \text{skill / option}$$

### 10. Separate agent motion from world motion

In robot datasets, the robot arm may dominate the image. The model may discover “arm moved left” rather than “object was pushed left.”

This is not necessarily wrong, but you should know which one you want.

Mitigations:

* mask robot arm;
* use object-centric features;
* evaluate object displacement separately;
* use contact/no-contact labels if available;
* train with random camera/background augmentations.

### 11. Beware camera motion

In human video datasets, camera motion can dominate `a_t`.

For egocentric video, your model may discover:

```
look left
walk forward
camera tilts down
hand enters frame
```

instead of object actions.

Mitigations:

* stabilize video;
* use flow decomposition;
* use hand/object masks;
* use exocentric datasets first;
* compare with optical-flow baselines.

### 12. Decide whether actions are discrete, continuous, or hierarchical

For discovery, I would start discrete:

$$a_t \in \{1, ..., K\}$$

Discrete codes are easier to inspect and evaluate.

Later, use hybrid actions:

$$a_t = (c_t, v_t)$$

where:

* `c_t` is a discrete action type;
* `v_t` is a continuous parameter.

Example:

```
c_t = push
v_t = direction and magnitude
```

This is more realistic than pure discrete action discovery.

## My concrete recommendation

For your first 4–6 weeks, do this:

### Week 1: synthetic

Build synthetic moving-object dataset.

Success criterion:

```
NMI(discovered_action, true_action) > 0.8
```

Also verify that the no-action model fails.

### Week 2: BAIR

Train on BAIR robot pushing.

Success criteria:

```
1. a_t improves future prediction over no-action baseline
2. action clusters correspond to push direction/contact
3. nearest neighbors in action space share transition type
4. code usage is non-collapsed
```

### Week 3: ablations

Run:

```
- no bottleneck
- no no-action loss
- no usage loss
- delta-clustering baseline
- optical-flow baseline
```

Do not move on until your method beats dumb baselines.

### Week 4–5: RoboNet or BridgeData V2

Add real diversity.

Success criteria:

```
1. linear probe from a_t predicts robot action
2. same latent actions appear across objects/scenes
3. counterfactual action swap produces plausible futures
```

### Week 6: Something-Something V2

Evaluate whether the method discovers human-object action categories.

Success criteria:

```
1. clusters align with labels better than flow/delta baselines
2. action retrieval returns same verb, not same object/background
3. open/close, push/pull, put/take separate cleanly
```

## The core research claim to aim for

A strong claim would be:

> A bottlenecked latent inverse-forward model can discover reusable action variables from video alone, and these variables improve future prediction, align with true robot controls when available, and transfer across objects, scenes, and embodiments.

To support that claim, you need evidence for:

```
Prediction:
  a_t improves future prediction.

Necessity:
  removing a_t hurts.

Minimality:
  a_t does not encode the whole future.

Semantic structure:
  action clusters align with human/robot action labels.

Transfer:
  same actions work across objects/scenes.

Utility:
  latent actions improve planning or downstream control.
```

## Most important practical advice

Do not start with a giant dataset. Start with a toy dataset where you know the answer. Then BAIR. Then one larger robot dataset. Then one human-video dataset.

Your first milestone should not be “train on DROID.” It should be:

> On a toy dataset, the model discovers the hidden action variable; on BAIR, the discovered codes correspond to meaningful robot-object transition types; and ablations prove the bottleneck and no-action loss are actually responsible.
