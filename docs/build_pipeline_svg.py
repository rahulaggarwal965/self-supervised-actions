"""Build docs/pipeline.svg — the pipeline+losses diagram with LaTeX (mathtext) labels.

Renders math to vector paths (no system TeX needed), so the SVG is portable.
Run:  uv run python docs/build_pipeline_svg.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["svg.fonttype"] = "path"  # embed text as paths -> renders anywhere
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern: real-LaTeX look

W, H = 1340, 860
COMP = "#e0ecff"
COMP_E = "#2563eb"
GEN = "#dcfce7"
GEN_E = "#16a34a"
IN = "#f1f5f9"
IN_E = "#64748b"
OUT = "#fef9c3"
OUT_E = "#ca8a04"
CF, ALL, VQ, DEL, DEC, USE, MAR = (
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#475569",
    "#7c3aed",
    "#0891b2",
    "#d97706",
)

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # y increases downward (SVG-like)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, fc, ec, dashed=False, lw=1.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=9",
            fc=fc,
            ec=ec,
            lw=lw,
            ls="--" if dashed else "-",
            mutation_aspect=1,
            zorder=2,
        )
    )


def txt(x, y, s, size=8, color="#0f172a", weight="normal", style="normal", ha="center"):
    ax.text(
        x, y, s, fontsize=size, color=color, ha=ha, va="center", fontweight=weight, fontstyle=style,
        zorder=3,
    )


def arrow(p1, p2, color="#334155", lw=1.7, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1,
        )
    )


def dot(x, y, color):
    ax.add_patch(plt.Circle((x, y), 5.5, color=color, zorder=4))


# ---- title ----
txt(28, 30, "Self-Supervised Action Discovery — pipeline & losses", 15, "#0f172a", "bold", ha="left")
txt(28, 52, r"frame pair $\rightarrow$ discrete latent action (VQ) $\rightarrow$ additive latent "
    r"transition $\rightarrow$ decoded next frame.  label-free.", 9, "#475569", ha="left")

# ---- inputs ----
box(24, 120, 92, 60, IN, IN_E, lw=1.4)
txt(70, 143, r"$I_t$", 12, "#334155", "bold")
txt(70, 165, r"$3\times 64\times 64$", 8, "#64748b")
box(24, 300, 92, 60, IN, IN_E, lw=1.4)
txt(70, 322, r"$I_{t+1}$", 12, "#334155", "bold")
txt(70, 344, "observed next", 8, "#64748b")
box(18, 418, 104, 58, "#f8fafc", "#94a3b8", dashed=True, lw=1.3)
txt(70, 440, r"$\{\,I_{t+1}^{\,a}\,\}$", 11, "#475569", "bold")
txt(70, 458, "same-state futures", 8, "#64748b")
txt(70, 470, "(all actions)", 8, "#64748b")

# ---- encoder ----
box(150, 150, 140, 210, COMP, COMP_E, lw=2)
txt(220, 240, r"Encoder $E$", 12, "#1e3a8a", "bold")
txt(220, 262, r"CNN, 4$\times$ stride-2", 8.5, "#1e40af")
txt(220, 278, "(shared weights)", 8.5, "#1e40af")

# ---- inverse ----
box(420, 300, 196, 86, COMP, COMP_E, lw=2)
txt(518, 322, r"Inverse $g$", 11.5, "#1e3a8a", "bold")
txt(518, 341, r"$\Delta$feature $\rightarrow$ circular-conv", 8, "#1e40af")
txt(518, 356, r"$\rightarrow$ global-avg-pool", 8, "#1e40af")
txt(518, 373, "shift-invariant: ignores position", 8, "#0891b2", style="italic")

# ---- quantizer ----
box(680, 300, 162, 86, COMP, COMP_E, lw=2)
txt(761, 320, "Vector Quantizer", 11, "#1e3a8a", "bold")
txt(761, 340, r"$a_q=\mathrm{argmin}_k\,\|a_{\mathrm{pre}}-c_k\|$", 8.5, "#1e40af")
txt(761, 357, r"codebook $C\in\mathbb{R}^{K\times 64},\ K{=}6$", 8.5, "#1e40af")
txt(761, 374, "straight-through gradient", 8, "#b45309", style="italic")

# ---- dynamics ----
box(680, 140, 170, 72, GEN, GEN_E, lw=2)
txt(765, 163, r"Dynamics $f$", 11.5, "#166534", "bold")
txt(765, 184, r"$z_{\mathrm{ctx}} + T(a_q)$", 10.5, "#15803d")
txt(765, 201, "additive: per-action displacement", 8, "#15803d", style="italic")

# ---- head ----
box(920, 140, 180, 72, GEN, GEN_E, lw=2)
txt(1010, 162, "Head (decoder)", 11.5, "#166534", "bold")
txt(1010, 182, r"PixelDecoder: $\Delta = I_{t+1}-I_t$", 8.5, "#15803d")
txt(1010, 198, r"(Composite: $\alpha F + (1-\alpha)I_t$)", 8, "#15803d")

# ---- outputs ----
box(1160, 120, 96, 52, OUT, OUT_E, lw=1.8)
txt(1208, 140, r"$\hat I_{t+1}$", 12, "#854d0e", "bold")
txt(1208, 159, r"$= I_t + \Delta$", 8.5, "#a16207")
box(1160, 250, 96, 48, IN, IN_E, lw=1.4)
txt(1208, 274, r"target $I_{t+1}$", 10, "#334155", "bold")

# ---- flow arrows ----
arrow((116, 150), (150, 188))
arrow((116, 330), (150, 322))
arrow((290, 185), (680, 176))
txt(470, 170, r"$z_{\mathrm{ctx}}\in\mathbb{R}^{256}$  (context: scene + position)", 9.5, "#0f172a", "bold")
arrow((290, 330), (420, 340))
txt(356, 302, r"$\phi_2(I_t),\ \phi_2(I_{t+1})$", 8.5, "#334155")
txt(356, 316, r"$16\times 16\times 64$", 7.5, "#64748b")
arrow((616, 343), (680, 343))
txt(648, 332, r"$a_{\mathrm{pre}}$", 9, "#0f172a", "bold")
arrow((761, 300), (765, 212))
txt(835, 258, r"$a_q$  (discrete action)", 9, "#0f172a", "bold")
arrow((850, 176), (920, 176))
txt(885, 168, "feat", 8, "#334155")
arrow((1100, 160), (1160, 150))

# I_t skip (composite)
arrow((70, 120), (1010, 140), color="#94a3b8", lw=1.3, rad=-0.28, ls="--")
txt(560, 78, r"$I_t$ skip  (decoder composites over the current frame)", 8.5, "#64748b")

# ---- loss connectors on the pipeline ----
arrow((640, 333), (470, 190), color=DEC, lw=1.4, rad=0.3, ls="--")
dot(640, 333, DEC)
txt(556, 240, r"$L_{\mathrm{decorr}}$  ($a_{\mathrm{pre}}\perp z_{\mathrm{ctx}}$)", 9, DEC, "bold")
arrow((1208, 172), (122, 458), color=CF, lw=1.5, rad=0.32, ls="--")
txt(905, 424, r"$L_{\mathrm{cf}} + L_{\mathrm{allact}}$", 10.5, CF, "bold")
txt(905, 442, "pred(code) vs real per-action futures", 8.5, "#b91c1c")
dot(668, 343, VQ)
dot(761, 300, USE)
dot(765, 212, MAR)
dot(1130, 152, DEL)

# ---- legend ----
ax.plot([24, 1256], [512, 512], color="#e2e8f0", lw=1.5, zorder=1)
txt(24, 538, r"objective   $L_{\mathrm{total}} = \sum_i w_i\, L_i$      (weight $w_i$ in brackets)",
    14, "#0f172a", "bold", ha="left")


def entry(x, y, color, title, lines):
    ax.add_patch(plt.Rectangle((x, y), 14, 14, color=color, zorder=3))
    txt(x + 22, y + 7, title, 11, "#0f172a", "bold", ha="left")
    for i, ln in enumerate(lines):
        txt(x + 22, y + 26 + i * 15, ln, 9.5, "#334155", ha="left")


entry(24, 556, CF, "cf_contrastive [4] — discrimination driver",
      [r"InfoNCE: $\mathrm{sim}(pred,cand)=-\|pred-cand\|^2/\tau,\ \ \tau=0.03$.",
       r"positive = observed next; negatives = same-state futures under",
       r"other actions.  forces the code to encode WHICH action.  (breaks mean-seeking)"])
entry(24, 636, ALL, "all_action_prediction [2] — per-code supervised move",
      [r"$L=\frac{1}{A}\sum_a \| \mathrm{head}\, f(z_{\mathrm{ctx}},\,"
       r"\mathrm{code}(I_t,I_{t+1}^a)) - (I_{t+1}^a - I_t) \|^2$.",
       r"one code can't fit two futures $\Rightarrow$ distinct actions get distinct codes;",
       r"each code renders its action's real frame-change."])
entry(24, 716, VQ, "vq [1] — codebook + commitment",
      [r"$L=\|\,sg[a_{\mathrm{pre}}]-a_q\|^2 + \beta\|a_{\mathrm{pre}}-sg[a_q]\|^2,\ \ \beta=0.25$.",
       r"the term that spikes at the phase transition (codebook reassigns)."])
entry(24, 776, DEL, "delta_sparsity [1] — clean background",
      [r"$L=\mathrm{mean}\,|\Delta|$.  true change is sparse (only the agent moves).",
       r"fragile: too strong erases the move itself."])

entry(660, 556, DEC, "decorrelation [2] — code $\\perp$ position",
      [r"$C=\hat a^{\top}\hat z/(B{-}1)$ between $a_{\mathrm{pre}}$ and detached $z_{\mathrm{ctx}}$;"
       r"  $L=\mathrm{mean}(C^2)$.",
       r"drives the code to carry info NOT in $z_{\mathrm{ctx}}$ (which holds position)."])
entry(660, 620, USE, "usage [0.1] — anti-collapse",
      [r"$L=H(p\,|\,\mathrm{sample}) - H(\mathbb{E}_{\mathrm{batch}}[p]),\ \ p=\mathrm{softmax}(-\mathrm{dist})$.",
       r"decisive per-sample + all codes used across the batch."])
entry(660, 684, MAR, "margin [1] — the action must matter",
      [r"$L=\mathrm{ReLU}(m + \mathrm{err} - \mathrm{err}_0),\ \ \mathrm{err}_0=$ error with a ZERO action.",
       r"using the code must beat no-action by margin $m=0.002$."])
txt(660, 742, "dashed grey = $I_t$ skip (Composite head).  coloured dots mark where each",
    9.5, "#64748b", style="italic", ha="left")
txt(660, 757, "loss attaches.  prediction on the observed transition is subsumed by all_action.",
    9.5, "#64748b", style="italic", ha="left")
txt(660, 780, "green = trained to render pixels    ·    blue = trained to discover the action",
    10, "#166534", "bold", ha="left")

txt(24, 834, r"shapes: $I$ $3{\times}64{\times}64$ · $\phi_2$ $16{\times}16{\times}64$ · "
    r"$z_{\mathrm{ctx}}$ 256 · $a_{\mathrm{pre}},a_q$ 64 · codebook $6{\times}64$.   "
    r"$sg[\cdot]$ = stop-gradient.   $A$ = #actions.", 9, "#94a3b8", ha="left")

fig.savefig("docs/pipeline.svg", format="svg")
fig.savefig("/tmp/pipeline_preview.png", format="png", dpi=110)
print("wrote docs/pipeline.svg")
