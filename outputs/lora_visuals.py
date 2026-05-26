import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

FROZEN  = "#90A4AE"
LORA_A  = "#42A5F5"
LORA_B  = "#66BB6A"
ADD_CLR = "#FFA726"
BG      = "#FAFAFA"
TXT     = "#212121"

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(ax, cx, cy, w, h, color, title, sub=None):
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                       facecolor=color, edgecolor="white", lw=2, zorder=4)
    ax.add_patch(p)
    dy = 0.13 if sub else 0
    ax.text(cx, cy + dy, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5)
    if sub:
        ax.text(cx, cy - 0.18, sub, ha="center", va="center",
                fontsize=8.5, color="white", alpha=0.9, zorder=5)

def circle(ax, cx, cy, r, color, label):
    c = plt.Circle((cx, cy), r, facecolor=color, edgecolor="white",
                   lw=2, zorder=4)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=18, fontweight="bold", color="white", zorder=5)

def harrow(ax, x1, x2, y, color, lw=2):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=6)

def varrow(ax, x, y1, y2, color, lw=2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=6)

def line(ax, xs, ys, color, lw=2):
    ax.plot(xs, ys, color=color, lw=lw, zorder=3, solid_capstyle="round")

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – LoRA architecture
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis("off")

# ── Layout constants ─────────────────────────────────────────────────────────
Y_TOP  = 3.6   # top path  (W)
Y_BOT  = 1.5   # bottom path (A → B)
Y_MID  = 2.55  # vertical midpoint
X_IN   = 1.0   # input x
X_W1   = 3.8   # left edge of W
X_W2   = 7.2   # right edge of W
X_A    = 3.0   # centre of A
X_B    = 5.5   # centre of B
X_ADD  = 8.8   # addition circle
X_OUT  = 10.5  # output

# ── Boxes ────────────────────────────────────────────────────────────────────
rbox(ax, (X_W1+X_W2)/2, Y_TOP, X_W2-X_W1, 1.0, FROZEN,
     "W  (pretrained)", "d × d  |  frozen")
rbox(ax, X_A, Y_BOT, 1.4, 0.9, LORA_A, "A", "d × r")
rbox(ax, X_B, Y_BOT, 1.4, 0.9, LORA_B, "B", "r × d")
circle(ax, X_ADD, Y_MID, 0.38, ADD_CLR, "+")

# ── Input label — above the incoming arrow, clear of the line ────────────────
ax.text(X_IN - 0.85, Y_MID + 0.32, "x", ha="center", va="center",
        fontsize=14, fontstyle="italic", color=TXT)

# ── Output label ─────────────────────────────────────────────────────────────
ax.text(X_OUT + 0.3, Y_MID, "h", ha="center", va="center",
        fontsize=14, fontstyle="italic", color=TXT)

# ── Top path: x → (up) → W → (right+down) → + ──────────────────────────────
line(ax, [X_IN, X_IN], [Y_MID, Y_TOP], FROZEN)          # up
harrow(ax, X_IN, X_W1 - 0.05, Y_TOP, FROZEN)            # into W
line(ax, [X_W2, X_ADD], [Y_TOP, Y_TOP], FROZEN)         # right to above +
ax.annotate("", xy=(X_ADD, Y_MID + 0.38),               # down into + (one clean arrow)
            xytext=(X_ADD, Y_TOP),
            arrowprops=dict(arrowstyle="-|>", color=FROZEN,
                            lw=2, mutation_scale=14), zorder=6)

# ── Bottom path: x → (down) → A → B → (right+up) → + ───────────────────────
line(ax, [X_IN, X_IN], [Y_MID, Y_BOT], LORA_A)          # down
harrow(ax, X_IN, X_A - 0.7 - 0.05, Y_BOT, LORA_A)      # into A
harrow(ax, X_A + 0.7, X_B - 0.7 - 0.05, Y_BOT, LORA_B) # A → B
ax.text((X_A + X_B)/2, Y_BOT + 0.62,
        "rank bottleneck  (r = 8 ≪ d)", ha="center", va="center",
        fontsize=8, color="#37474F", fontstyle="italic")
line(ax, [X_B + 0.7, X_ADD], [Y_BOT, Y_BOT], LORA_B)   # right to below +
ax.annotate("", xy=(X_ADD, Y_MID - 0.38),               # up into + (one clean arrow)
            xytext=(X_ADD, Y_BOT),
            arrowprops=dict(arrowstyle="-|>", color=LORA_B,
                            lw=2, mutation_scale=14), zorder=6)

# ── Input stub ───────────────────────────────────────────────────────────────
harrow(ax, X_IN - 0.7, X_IN - 0.01, Y_MID, "#546E7A")

# ── + → output ───────────────────────────────────────────────────────────────
harrow(ax, X_ADD + 0.38, X_OUT + 0.1, Y_MID, ADD_CLR)

# ── Equation box ─────────────────────────────────────────────────────────────
ax.text(5.5, 0.55, r"$h = Wx\ +\ BAx$",
        ha="center", va="center", fontsize=13, color=TXT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#CFD8DC", lw=1.5))

# ── Parameter note ────────────────────────────────────────────────────────────
ax.text(5.5, 0.12,
        "W  frozen — 600M params  (100%)      "
        "BA  trainable — 1.18M params  (0.19%)",
        ha="center", va="center", fontsize=9, color="#546E7A")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=FROZEN,  label="Frozen weights (W)"),
    mpatches.Patch(facecolor=LORA_A,  label="LoRA matrix A  (d × r)"),
    mpatches.Patch(facecolor=LORA_B,  label="LoRA matrix B  (r × d)"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9,
          framealpha=0.85, edgecolor="#CFD8DC")

ax.set_title("LoRA: Low-Rank Adaptation", fontsize=15,
             fontweight="bold", color=TXT, pad=10)

fig.tight_layout()
fig.savefig("outputs/lora_architecture.png", dpi=180,
            bbox_inches="tight", facecolor=BG)
print("Saved: outputs/lora_architecture.png")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Parameter comparison bar chart
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

methods = ["Full fine-tuning", "LoRA\n(r=8, q+v)", "IA³", "Prefix tuning"]
params  = [600_000_000, 1_180_000, 560_000, 491_520]
pct     = ["100%", "0.19%", "0.09%", "0.08%"]
colors  = [FROZEN, LORA_B, "#AB47BC", "#FF7043"]

bars = ax2.barh(methods, params, color=colors,
                edgecolor="white", linewidth=1.5, height=0.5)

ax2.set_xscale("log")
ax2.set_xlim(1e5, 3e9)
ax2.set_xlabel("Trainable parameters (log scale)", fontsize=11, color=TXT)
ax2.set_title("Trainable Parameters by Method",
              fontsize=13, fontweight="bold", color=TXT, pad=10)
ax2.tick_params(colors=TXT, labelsize=10)
for spine in ax2.spines.values():
    spine.set_edgecolor("#CFD8DC")

for bar, p, label in zip(bars, params, pct):
    ax2.text(bar.get_width() * 1.5,
             bar.get_y() + bar.get_height() / 2,
             f"{p/1e6:.2f}M  ({label})",
             va="center", fontsize=10, color=TXT, fontweight="bold")

ax2.invert_yaxis()
fig2.tight_layout()
fig2.savefig("outputs/lora_param_comparison.png", dpi=180,
             bbox_inches="tight", facecolor=BG)
print("Saved: outputs/lora_param_comparison.png")
