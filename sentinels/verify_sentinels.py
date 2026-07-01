#!/usr/bin/env python3
"""verify_sentinels.py — BlindSpot paper number-reproducibility gate.

Reads canonical_bundle.yaml (each sentinel = canonical value + regex anchor in the
paper .tex) and checks that every paper number still matches its canonical
z1/test_10k_1 (M1a-config) source. Any drift -> FAIL.

Usage:
    python3 verify_sentinels.py                      # check default paper main.tex
    python3 verify_sentinels.py --tex <path>         # check a specific .tex
    python3 verify_sentinels.py --self-test          # known-good + known-bad fixtures

Exit code = number of FAIL sentinels (0 = all reproducible).

Why this exists: 2026-06-21 the EW numbers were wrong because inference_mag1921's
CANONICAL_CONFIG/CKPT defaults were stale (E1 baseline/ep65 vs M1a/ep190), giving
garbage mu_x. A sentinel that pins each paper number to its canonical source catches
exactly this drift class. See logs/rca/2026-06-21-blindspot-ALL-PITFALLS-and-doc-index.md.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "canonical_bundle.yaml"
REPO_LOCAL_TEX = HERE.parent / "paper/main.tex"
MONOREPO_TEX = HERE.parents[2] / "work/papers/BlindspotDenoiser/main.tex"
DEFAULT_TEX = REPO_LOCAL_TEX if REPO_LOCAL_TEX.exists() else MONOREPO_TEX


def load_sentinels(bundle_path):
    try:
        import yaml
        d = yaml.safe_load(open(bundle_path))
        return d["sentinels"]
    except ImportError:
        # minimal fallback parser for our flat list-of-dicts schema
        sents, cur = [], None
        for line in open(bundle_path):
            s = line.rstrip("\n")
            if re.match(r"\s*- id:", s):
                if cur:
                    sents.append(cur)
                cur = {"id": s.split("id:", 1)[1].strip()}
            elif cur is not None and re.match(r"\s+\w+:", s):
                k, v = s.strip().split(":", 1)
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                cur[k] = v
        if cur:
            sents.append(cur)
        return sents


def check(sentinels, tex):
    rows = []
    for s in sentinels:
        rx = s["regex"]
        m = re.search(rx, tex)
        if not m:
            rows.append((s["id"], "FAIL", f"regex no match (number drifted or format changed): {rx[:50]}"))
            continue
        got = m.group(1)
        want = str(s["value"])
        if got == want:
            rows.append((s["id"], "PASS", f"{got} == canonical {want}"))
        else:
            rows.append((s["id"], "FAIL", f"tex={got} != canonical {want}  [{s.get('source','')}]"))
    return rows


def self_test():
    """known-good must all PASS, known-bad must FAIL the mutated sentinel."""
    sents = [
        {"id": "snr_noisy", "value": "7.6",
         "regex": r"reconstruction S/N rises from \$([0-9.]+)\$ on the noisy input"},
        {"id": "ew_8498_den", "value": "0.121",
         "regex": r"8498\$~\\AA\{\} & \$0.737\$ & \\mathbf\{([0-9.]+)\}"},
    ]
    good = r"reconstruction S/N rises from $7.6$ on the noisy input to $122$ \\ $\quad 8498$~\AA{} & $0.737$ & \mathbf{0.121} \\"
    bad = good.replace("0.121", "0.260")  # the E1-config-bug wrong value
    g = check(sents, good)
    b = check(sents, bad)
    ok = all(r[1] == "PASS" for r in g) and any(r[1] == "FAIL" and r[0] == "ew_8498_den" for r in b)
    print("[self-test] known-good:", [r[1] for r in g])
    print("[self-test] known-bad :", [(r[0], r[1]) for r in b])
    print("[self-test]", "PASS" if ok else "FAIL — verifier miscalibrated")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=str(DEFAULT_TEX))
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    sents = load_sentinels(a.bundle)
    tex = Path(a.tex).read_text()
    rows = check(sents, tex)
    nfail = sum(1 for _, st, _ in rows if st == "FAIL")
    print(f"=== verify_sentinels: {a.tex} ({len(sents)} sentinels) ===")
    for sid, st, msg in rows:
        print(f"  [{st}] {sid}: {msg}")
    print(f"=== {len(rows)-nfail}/{len(rows)} PASS, {nfail} FAIL ===")
    sys.exit(nfail)


if __name__ == "__main__":
    main()
