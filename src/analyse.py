"""Tables and figures for the RAG authorisation paper."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
facts = {}

# E1: authorised recall with post-filtering vs pre-filtering.
e1 = pd.read_csv(RES / "e1_postfilter.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, r in zip(axes, ("BM25", "Dense")):
    d = e1[e1.retriever == r]
    for f, c in zip((1, 2, 4, 8, 16), plt.cm.Blues(np.linspace(0.35, 0.95, 5))):
        g = d[d.overfetch == f].groupby("tenants").hit_post.mean()
        ax.plot(g.index, g.values, marker="o", ms=3, color=c, label=f"post-filter, fetch {f}$\\times$k")
    g = d.groupby("tenants").hit_pre.mean()
    ax.plot(g.index, g.values, color="#c0504d", lw=1.6, ls="--", label="pre-filter (exact)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of tenants")
    ax.set_title(r)
axes[0].set_ylabel("Target document in top 5")
axes[0].legend(fontsize=6, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.38))
fig.tight_layout()
fig.savefig(FIG / "postfilter.pdf", bbox_inches="tight")
plt.close(fig)
rows = []
for r in ("BM25", "Dense"):
    d = e1[e1.retriever == r]
    pre = d.hit_pre.mean()
    row = {"Retriever": r, "Pre-filter": f"{pre:.3f}"}
    for f in (1, 4, 16):
        x = d[d.overfetch == f]
        row[f"Post, {f}$\\times$"] = f"{x.hit_post.mean():.3f}"
    row["Short answers at 4$\\times$"] = f"{(d[d.overfetch == 4].returned < 5).mean() * 100:.0f}\\%"
    rows.append(row)
(TAB / "postfilter.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Known-item recall in the top 5 with exact pre-filtering and with post-filtering "
    "of a global top-$fk$ list, averaged over 2--32 tenants, 3 seeds and 600 user queries per setting. The last column is "
    "the share of queries for which post-filtering returned fewer than 5 documents.}\n\\label{tab:postfilter}\n\\small\n"
    "\\setlength{\\tabcolsep}{4pt}\n" + pd.DataFrame(rows).to_latex(index=False, escape=False, column_format="lrrrrr")
    + "\\end{table}\n")
facts["e1"] = {r: {"pre": float(e1[e1.retriever == r].hit_pre.mean()),
                   **{f"post{f}": float(e1[(e1.retriever == r) & (e1.overfetch == f)].hit_post.mean()) for f in (1, 2, 4, 8, 16)},
                   "pre_by_tenants": e1[e1.retriever == r].groupby("tenants").hit_pre.mean().round(3).to_dict(),
                   "post1_by_tenants": e1[(e1.retriever == r) & (e1.overfetch == 1)].groupby("tenants").hit_post.mean().round(3).to_dict()}
               for r in ("BM25", "Dense")}
facts["e1"]["n_queries_per_cell"] = int(len(e1) / (2 * 5 * 5))

# E2: existence side channel.
e2 = pd.read_csv(RES / "e2_sidechannel.csv")
facts["e2"] = {"auc_mean": float(e2.auc.mean()), "auc_max": float(e2.auc.max()), "auc_min": float(e2.auc.min()),
               "by": {f"{a}-{b}": v for (a, b), v in e2.groupby(["retriever", "overfetch"]).auc.mean().round(3).items()}}

# E3: shared-statistics attack.
e3 = pd.read_csv(RES / "e3_idf.csv")
facts["e3"] = {d: {"exact": float((g.est_hidden == g.true_hidden).mean()),
                   "mae": float((g.est_hidden - g.true_hidden).abs().mean()), "queries": float(g.queries.mean()),
                   "trials": int(len(g))} for d, g in e3.groupby("design")}
fig, ax = plt.subplots(figsize=(3.2, 2.4))
for d, m, c in (("shared statistics", "o", "#c0504d"), ("per-tenant statistics", "x", "#1f4e79")):
    g = e3[e3.design == d]
    ax.scatter(g.true_hidden, g.est_hidden, s=8, marker=m, color=c, label=d, alpha=0.8)
ax.plot([0, 40], [0, 40], color="grey", lw=0.5, ls=":")
ax.set_xlabel("Hidden documents containing the secret term")
ax.set_ylabel("Attacker's estimate")
ax.legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "idf_attack.pdf")
plt.close(fig)

# E4: semantic cache.
if (RES / "e4_cache.csv").exists():
    e4 = pd.read_csv(RES / "e4_cache.csv")
    t = e4.assign(**{"Cache key": e4.design.map({"shared cache": "query only", "tenant-keyed cache": "query + tenant", "user-keyed cache": "query + permission set"}), "Threshold": e4.tau.map(lambda v: f"{v:.2f}"),
                     "Hit rate": e4.hit_rate.map(lambda v: f"{v * 100:.1f}\\%"),
                     "Leaking answers": e4.leak_rate.map(lambda v: f"{v * 100:.1f}\\%"),
                     "Share of hits that leak": e4.leaks_per_hit.map(lambda v: f"{v * 100:.1f}\\%")})
    t = t[t.tau.isin([0.85, 0.95])][["Cache key", "Threshold", "Hit rate", "Leaking answers",
                                     "Share of hits that leak"]]
    (TAB / "cache.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{A semantic answer cache over 10{,}000 paraphrased queries from 400 users in "
        "8 tenants. An answer leaks if it contains a document the asking user may not see.}\n\\label{tab:cache}\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n" + t.to_latex(index=False, escape=False, column_format="llrrr") + "\\end{table}\n")
    facts["e4"] = e4.round(4).to_dict("records")

(RES / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
print(json.dumps(facts, indent=1, default=str)[:3000])
