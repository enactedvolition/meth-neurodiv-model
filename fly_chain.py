"""Fast-and-dirty end-to-end chain: drug -> DA terminal -> dFSB -> arousal index.

Links and what each one actually is (honest labels):
  1 terminal   repaired model, fly Vmax/Km/k_diff (Vickrey 2013); peak SIZE unconstrained.
  2 cell type  NOT used: one shared terminal for every DA neuron (my clusters do not
               map to FlyWire types).
  3 spread     FlyWire v783 synapse counts from DA neurons onto FB6/FB7 tangential
               neurons (proxy for dFSB). Wiring-weighted, no diffusion model.
  4 receptor   sign only: DA via Dop1R1 inhibits dFSB sleep neurons -> arousal (Ueno 2012).
  5 readout    ASSUMED: arousal index = sum(syn_count * De). Monotone, no data.
Checked only for direction against fly behavior.
With one shared terminal and a linear readout the synapse weight cancels from every
ratio, so P1-P3 are inherited from link 1 and are NOT independent evidence.
"""
import os, re, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terminal_repaired as tr

# FlyWire v783: annot/annotations.tsv and conn/proofread_connections_783.feather (not in repo)
D = os.environ.get("FLYWIRE_DIR", "/home/claude/fly-meth/data/").rstrip("/") + "/"
a = pd.read_csv(D + "annot/annotations.tsv", sep="\t", low_memory=False)
ct, knt = a.cell_type.fillna(""), a.known_nt.fillna("")
# DA = annotated DAN class, or a known_nt token exactly "dopamine" (not "dopamine-negative").
# Predicted top_nt is not used: it labels columnar CX cells (PFR, PFGs) as dopaminergic.
is_da = (a.cell_class == "DAN") | knt.apply(lambda s: "dopamine" in [x.strip() for x in re.split(r"[,;]", s)])
da_ids = set(a.root_id[is_da]); dfsb_ids = set(a.root_id[ct.str.match(r"^FB[67]")])
c = pd.read_feather(D + "conn/proofread_connections_783.feather",
                    columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
e = c[c.pre_pt_root_id.isin(da_ids) & c.post_pt_root_id.isin(dfsb_ids)]
tot_in = c[c.post_pt_root_id.isin(dfsb_ids)].syn_count.sum()
t = a.set_index("root_id").cell_type.fillna("untyped")
by = e.groupby(e.pre_pt_root_id.map(t)).syn_count.sum().sort_values(ascending=False)
print(f"DA neurons: {len(da_ids)}; FB6/FB7 targets: {len(dfsb_ids)}")
print(f"DA->FB6/7 synapses: {e.syn_count.sum()} from {e.pre_pt_root_id.nunique()} DA neurons "
      f"({e.syn_count.sum()/tot_in:.1%} of all input to FB6/7)")
print("top DA types:", by.head(6).to_dict())
W = e.syn_count.sum()

arch, p, _ = tr.setup(); p.Vmax, p.Km, p.k_diff = 0.11, 1.3, 0.020; tr.ETA = 0.1
Dd = tr.derive(p)
res = {}
for name, v in (("control", 1.0), ("fumin-like 5%", 0.05), ("DAT-null", 0.0)):
    x, _ = tr.fixed_point(p, Dd, v)
    s, _ = tr.simulate(p, Dd, "AMPH", 1.0, vscale=v, escale=v, temp_rise=0.0)
    res[name] = (W * x[0], W * s.y[0].max())
b0 = res["control"][0]
print("\narousal index, relative to control baseline (assumed readout):")
for k, (base, peak) in res.items():
    print(f"  {k:14s} baseline {base/b0:6.2f}   AMPH peak {peak/b0:8.1f}   fold over own baseline {peak/base:7.1f}")
ok1 = res["DAT-null"][0] > res["control"][0]
ok2 = res["control"][1] / res["control"][0] > 3
ok3 = res["DAT-null"][1] / res["DAT-null"][0] <= 1.5
print(f"\nP1 DAT-null more aroused at rest (fumin short sleep, Kume 2005): {ok1}")
print(f"P2 AMPH raises arousal in controls: {ok2}")
print(f"P3 no AMPH arousal rise in DAT-null (Neurochem Res 2021, 5 mM): {ok3}")
