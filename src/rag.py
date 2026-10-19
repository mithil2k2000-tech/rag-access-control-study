"""Authorisation leaks in RAG retrieval: four experiments on a multi-tenant Reuters corpus.

E1  Post-filtering: retrieve globally, then drop documents the user may not see. How much authorised recall is lost?
E2  Existence side channel: with post-filtering, does the number of results reveal hidden matching documents?
E3  Shared corpus statistics: with correct pre-filtering on a shared BM25 index, can a tenant learn how many
    hidden documents contain a secret term, from rankings alone?
E4  Semantic answer cache: a cache keyed on query similarity but not on tenant serves other tenants' answers.

Documents are the 10,788 Reuters-21578 (ApteMod) articles; headlines act as known-item queries. Tenants and roles are
assigned at random, so any leak comes from the retrieval design, not from the data.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data" / "reuters", ROOT / "results"
RES.mkdir(exist_ok=True)
K = 5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ corpus

def load_corpus() -> pd.DataFrame:
    cats = {}
    for line in (DATA / "cats.txt").read_text().splitlines():
        parts = line.split()
        cats[parts[0]] = parts[1:]
    rows = []
    for fid, c in cats.items():
        text = (DATA / fid).read_text(encoding="latin-1")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        head, body = lines[0], " ".join(lines[1:])
        if len(head.split()) < 3 or len(body.split()) < 20:
            continue
        rows.append({"fid": fid, "headline": head.lower(), "body": body.lower(), "cat": c[0]})
    df = pd.DataFrame(rows).drop_duplicates("headline").reset_index(drop=True)
    return df


def assign_acl(df: pd.DataFrame, n_tenants: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["tenant"] = rng.integers(0, n_tenants, len(out))
    # Within a tenant: 70% visible to all staff, 20% to a department (one of 4), 10% to executives only.
    r = rng.random(len(out))
    out["scope"] = np.where(r < 0.7, "all", np.where(r < 0.9, "dept", "exec"))
    out["dept"] = rng.integers(0, 4, len(out))
    return out


def allowed(df: pd.DataFrame, tenant: int, dept: int, exec_: bool) -> np.ndarray:
    m = (df.tenant.values == tenant) & ((df.scope.values == "all") | ((df.scope.values == "dept") & (df.dept.values == dept))
                                        | ((df.scope.values == "exec") & exec_))
    return m


# ------------------------------------------------------------------ retrievers

class BM25:
    """Okapi BM25 (k1=1.2, b=0.75) over a sparse term matrix. IDF uses the whole index unless a mask is given."""

    def __init__(self, texts: list[str], k1=1.2, b=0.75, vocab=None):
        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)
        self.tf = self.cv.fit_transform(texts).tocsc().astype(np.float32)
        self.k1, self.b = k1, b
        self.dl = np.asarray(self.tf.sum(axis=1)).ravel()
        self.N = self.tf.shape[0]

    def scores(self, query: str, mask: np.ndarray | None = None, stats_mask: np.ndarray | None = None) -> np.ndarray:
        """mask: documents that may be returned. stats_mask: documents whose statistics feed IDF and avgdl
        (None = the whole shared index, the usual design; equal to mask = per-tenant statistics)."""
        sm = np.ones(self.N, bool) if stats_mask is None else stats_mask
        N = sm.sum()
        avgdl = self.dl[sm].mean()
        ids = [self.cv.vocabulary_[t] for t in re.findall(r"\w+", query.lower()) if t in self.cv.vocabulary_]
        s = np.zeros(self.N, np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.dl / avgdl)
        for j in ids:
            col = self.tf[:, j]
            rows, f = col.indices, col.data
            df_ = int(sm[rows].sum())
            if df_ == 0:
                continue
            idf = np.log(1 + (N - df_ + 0.5) / (df_ + 0.5))
            s[rows] += idf * f * (self.k1 + 1) / (f + norm[rows])
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return s


class Dense:
    """LSA embeddings (TF-IDF + 256-d SVD), a reproducible stand-in for a neural embedding model."""

    def __init__(self, texts: list[str], dim: int = 256, seed: int = 0):
        self.tv = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
        X = self.tv.fit_transform(texts)
        self.svd = TruncatedSVD(dim, random_state=seed)
        E = self.svd.fit_transform(X)
        self.E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def embed(self, texts: list[str]) -> np.ndarray:
        E = self.svd.transform(self.tv.transform(texts))
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def scores(self, q: str, mask: np.ndarray | None = None) -> np.ndarray:
        s = self.E @ self.embed([q])[0]
        return np.where(mask, s, -np.inf) if mask is not None else s


def topk(s: np.ndarray, k: int) -> np.ndarray:
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return np.array([], int)
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])]


# ------------------------------------------------------------------ E1 and E2

def e1_e2(df: pd.DataFrame, bm: BM25, dn: Dense, seeds=(0, 1, 2)):
    rows, side = [], []
    for seed in seeds:
        for n_t in (2, 4, 8, 16, 32):
            acl = assign_acl(df, n_t, seed)
            rng = np.random.default_rng(100 + seed)
            # Users: 60 per configuration, random tenant/department/exec flag.
            for u in range(60):
                t, d, ex = int(rng.integers(n_t)), int(rng.integers(4)), bool(rng.random() < 0.2)
                m = allowed(acl, t, d, ex)
                mine = np.flatnonzero(m)
                targets = rng.choice(mine, min(10, len(mine)), replace=False)
                for name, ret in (("BM25", bm), ("Dense", dn)):
                    for target in targets:
                        q = acl.headline.iat[target]
                        s = ret.scores(q)
                        exact = topk(np.where(m, s, -np.inf), K)
                        hit_exact = target in exact
                        for f in (1, 2, 4, 8, 16):
                            cand = topk(s, K * f)
                            kept = cand[m[cand]][:K]
                            rows.append({"seed": seed, "tenants": n_t, "retriever": name, "overfetch": f,
                                         "auth_share": m.mean(), "exec": ex, "hit_post": target in kept,
                                         "hit_pre": hit_exact, "returned": len(kept)})
            # E2: an outsider probes with headlines of documents they cannot see (positives) and of documents that
            # are not in the index at all (negatives, a held-out 10%); the observable is how many results come back.
            held = rng.random(len(acl)) < 0.10
            att_t = 0
            m_att = allowed(acl, att_t, 0, False) & ~held
            index_mask = ~held
            hidden = np.flatnonzero(index_mask & ~m_att)
            pos = rng.choice(hidden, 300, replace=False)
            neg = rng.choice(np.flatnonzero(held), 300, replace=False)
            for name, ret in (("BM25", bm), ("Dense", dn)):
                for f in (2, 4):
                    feats, labels = [], []
                    for lab, ids in ((1, pos), (0, neg)):
                        for i in ids:
                            s = np.where(index_mask, ret.scores(acl.headline.iat[i]), -np.inf)
                            cand = topk(s, K * f)
                            kept = cand[m_att[cand]][:K]
                            feats.append(-len(kept))  # fewer results suggests hidden matches took the slots
                            labels.append(lab)
                    side.append({"seed": seed, "tenants": n_t, "retriever": name, "overfetch": f,
                                 "auc": roc_auc_score(labels, feats),
                                 "pos_short": float(np.mean([-x < K for x, y in zip(feats, labels) if y == 1])),
                                 "neg_short": float(np.mean([-x < K for x, y in zip(feats, labels) if y == 0]))})
            log("E1/E2", seed, n_t)
    pd.DataFrame(rows).to_csv(RES / "e1_postfilter.csv", index=False)
    pd.DataFrame(side).to_csv(RES / "e2_sidechannel.csv", index=False)


# ------------------------------------------------------------------ E3

def e3(df: pd.DataFrame, trials: int = 100, seed: int = 0):
    """Rank-only attack on shared BM25 statistics, with correct pre-filtering.

    The victim tenant's hidden documents contain a secret term in m of them (m in 0..40). The attacker can upload
    documents to their own tenant and query; they only see the ORDER of their own results. They plant one probe
    document containing the secret term, and control documents containing nonce terms planted in exactly j copies
    (j = 1..64), all of equal length. Ranking the probe against control j reveals whether df(secret) > df(control).
    """
    rng = np.random.default_rng(seed)
    texts = (df.headline + " " + df.body).tolist()
    base_len = 60
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do".split()
    out = []
    for trial in range(trials):
        m = int(rng.integers(0, 41))
        victims = rng.choice(len(texts), m, replace=False) if m else np.array([], int)
        secret = f"zq{rng.integers(10**8):08d}"
        docs = list(texts)
        for v in victims:
            docs[v] = docs[v] + " " + secret
        # Attacker documents: one probe plus controls c_j with j copies each (j in powers of two and fills).
        ladder = list(range(1, 65))
        att_docs, att_labels = [], []
        body = " ".join(rng.choice(filler, base_len - 1))
        att_docs.append(body + " " + secret)
        att_labels.append(("probe", 0))
        for j in ladder:
            nonce = f"cn{trial}x{j}"
            for c in range(j):
                att_docs.append(" ".join(rng.choice(filler, base_len - 1)) + " " + nonce)
                att_labels.append(("ctrl", j))
        all_docs = docs + att_docs
        n0 = len(docs)
        tenant_mask = np.zeros(len(all_docs), bool)
        tenant_mask[n0:] = True  # the attacker sees only their own uploads
        bm = BM25(all_docs)
        for design, stats in (("shared statistics", None), ("per-tenant statistics", tenant_mask)):
            probe_idx = n0
            # Binary search over j using rank comparisons between the probe and the first copy of control j.
            first = {}
            for k_, (kind, j) in enumerate(att_labels):
                if kind == "ctrl" and j not in first:
                    first[j] = n0 + k_
            lo, hi = 0, 64  # estimate of the probe term's document frequency among *all* documents
            comparisons = 0
            while lo < hi:
                mid = (lo + hi + 1) // 2
                s = bm.scores(f"{secret} cn{trial}x{mid}", mask=tenant_mask, stats_mask=stats)
                comparisons += 1
                # Higher score = rarer term. Probe ranks above control(mid) => df(secret) < mid.
                if s[probe_idx] > s[first[mid]]:
                    hi = mid - 1
                else:
                    lo = mid
            est_df = lo  # largest j with df(secret) >= j, so df(secret) is about lo
            out.append({"trial": trial, "design": design, "true_hidden": m, "est_hidden": max(0, est_df - 1),
                        "queries": comparisons})
        if trial % 20 == 0:
            log("E3", trial)
    pd.DataFrame(out).to_csv(RES / "e3_idf.csv", index=False)


# ------------------------------------------------------------------ E4

def paraphrase(q: str, rng) -> str:
    w = q.split()
    keep = [x for x in w if rng.random() > 0.25] or w[:1]
    if len(keep) > 2 and rng.random() < 0.5:
        i = int(rng.integers(len(keep) - 1))
        keep[i], keep[i + 1] = keep[i + 1], keep[i]
    return " ".join(keep)


def e4(df: pd.DataFrame, dn: Dense, n_tenants: int = 8, n_queries: int = 10000, seed: int = 0):
    """Shared semantic answer cache. Popular questions (Zipf) are asked in paraphrased form by users of all tenants.
    An answer is the set of document ids retrieved for the asking user. A cache hit for a user of another tenant
    returns documents they are not authorised to see."""
    rng = np.random.default_rng(seed)
    acl = assign_acl(df, n_tenants, seed)
    pool = rng.choice(len(df), 2000, replace=False)
    zipf = 1 / np.arange(1, len(pool) + 1) ** 1.1
    zipf /= zipf.sum()
    users = [(int(rng.integers(n_tenants)), int(rng.integers(4)), bool(rng.random() < 0.2)) for _ in range(400)]
    stream = []
    for _ in range(n_queries):
        base = df.headline.iat[pool[rng.choice(len(pool), p=zipf)]]
        stream.append((users[int(rng.integers(len(users)))], paraphrase(base, rng)))
    Q = dn.embed([q for _, q in stream])
    masks = {}
    results = []
    for design in ("shared cache", "tenant-keyed cache", "user-keyed cache"):
        for tau in (0.80, 0.85, 0.90, 0.95, 0.98):
            keys = np.zeros((n_queries, Q.shape[1]), np.float32)
            entries = []  # (scope key, answer doc ids)
            hits = leaks = leaked_docs = 0
            for i, ((t, d, ex), q) in enumerate(stream):
                scope = {"shared cache": 0, "tenant-keyed cache": t, "user-keyed cache": (t, d, ex)}[design]
                if (t, d, ex) not in masks:
                    masks[(t, d, ex)] = allowed(acl, t, d, ex)
                m = masks[(t, d, ex)]
                hit = None
                if len(entries):
                    sims = keys[:len(entries)] @ Q[i]
                    for j in np.flatnonzero(sims >= tau)[::-1]:
                        if entries[j][0] == scope:
                            hit = j
                            break
                if hit is not None:
                    hits += 1
                    ans = entries[hit][1]
                    bad = [a for a in ans if not m[a]]
                    if bad:
                        leaks += 1
                        leaked_docs += len(bad)
                else:
                    ans = topk(dn.scores(q, m), K)
                    keys[len(entries)] = Q[i]
                    entries.append((scope, ans))
            results.append({"design": design, "tau": tau, "hit_rate": hits / n_queries, "leak_rate": leaks / n_queries,
                            "leaks_per_hit": leaks / max(1, hits), "leaked_docs": leaked_docs})
            log("E4", design, tau, results[-1])
    pd.DataFrame(results).to_csv(RES / "e4_cache.csv", index=False)


if __name__ == "__main__":
    df = load_corpus()
    log("corpus", len(df), "documents,", df.cat.nunique(), "categories")
    steps = sys.argv[1:] or ["e1", "e3", "e4"]
    texts = df.body.tolist()  # headlines are the queries, so they are kept out of the index
    bm, dn = None, None
    if {"e1", "e4"} & set(steps):
        bm, dn = BM25(texts), Dense(texts)
    if "e1" in steps:
        e1_e2(df, bm, dn)
    if "e3" in steps:
        e3(df)
    if "e4" in steps:
        e4(df, dn)
    (RES / "corpus.json").write_text(json.dumps({"documents": len(df), "categories": int(df.cat.nunique()),
                                                 "top_categories": Counter(df.cat).most_common(8)}))
