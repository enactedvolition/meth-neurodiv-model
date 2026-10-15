"""Robustness of ordinal fly results to unmeasured terminal parameters.

Not a calibration: nothing is fitted.  Fly-measured values are sampled at their
reported mean and SD; every other parameter (inherited from the mammalian
model) is sampled over a +/-5x log-uniform range (n_hill 1-3, ETA 0.02-0.5)
and the constants are re-derived from the baseline balance for each draw.
Efflux is tied to DAT capacity (escale = vscale); the 37 C temperature module
is switched off (temp_rise=0) because it is not meaningful for flies.

Fly-measured (larval CNS, FSCV; Vickrey et al. 2013, doi:10.1021/cn400019q):
  Vmax = 0.11 +/- 0.02 uM/s, Km = 1.3 +/- 0.6 uM, k_diff = 0.020 +/- 0.002 /s
  (k_diff from DAT-null fumin larvae).  Larval data, not adult.

Ordinal outcomes checked per draw (AMPH 1 uM, 1 h), from the adult-fly
DAT-null amphetamine behavior paper (doi:10.1007/s11064-021-03275-4):
  O1 DAT-null resting DA > control      O2 control surge > 3x baseline
  O3 DAT-null no surge (<= 1.5x own)    O4 fumin-like (5% Vmax) peak < control
Not sampled: VMAT-block factors (0.9 METH / 0.5 AMPH) hard-coded in rhs.
Run:  python3 fly_robustness.py   (about a minute)
"""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terminal_repaired as tr
from scipy.stats import spearmanr
rng=np.random.default_rng(1)
N=250
LOG5=lambda c: c*np.exp(rng.uniform(-np.log(5),np.log(5)))
names=["k_rev","K_AMPH","n_hill","k_leak","k_MAO","k_pack","De0","Dc0","ETA"]
rows=[];infeas=0;unstable=0
for i in range(N):
    arch,p,_=tr.setup()
    p.Vmax=max(0.03,rng.normal(0.11,0.02)); p.Km=max(0.3,rng.normal(1.3,0.6)); p.k_diff=max(0.008,rng.normal(0.020,0.002))
    p.k_rev=LOG5(8.0); p.K_AMPH=LOG5(0.75); p.n_hill=rng.uniform(1,3)
    p.k_leak=LOG5(0.01); p.k_MAO=LOG5(0.1); p.k_pack=LOG5(0.5)
    p.De_baseline=LOG5(0.025); p.Dc_baseline=LOG5(1.0); tr.ETA=rng.uniform(0.02,0.5)
    D=tr.derive(p)
    if D["Dv0"]<=0: infeas+=1; continue
    try:
        xc,rc=tr.fixed_point(p,D,1.0); xk,rk=tr.fixed_point(p,D,0.0); xf,rf=tr.fixed_point(p,D,0.05)
        if max(abs(rc).max(),abs(rk).max(),abs(rf).max())>1e-8 or min(xc)<=0 or min(xk)<=0: unstable+=1; continue
        if np.linalg.eigvals(tr.jacobian(p,D,xc)).real.max()>=0: unstable+=1; continue
        pk={}
        for n,v in (("c",1.0),("f",0.05),("k",0.0)):
            s,_=tr.simulate(p,D,"AMPH",1.0,vscale=v,escale=v,temp_rise=0.0)
            pk[n]=s.y[0].max()
    except Exception as e:
        unstable+=1; continue
    rows.append(dict(par=[p.k_rev,p.K_AMPH,p.n_hill,p.k_leak,p.k_MAO,p.k_pack,p.De_baseline,p.Dc_baseline,tr.ETA],
        O1=xk[0]>xc[0], O2=pk["c"]/xc[0]>3, O3=pk["k"]/xk[0]<=1.5, O4=pk["f"]<pk["c"],
        mag=pk["c"]/xc[0], peak_uM=pk["c"], ko_base=xk[0]/xc[0]))
tr.ETA=0.1
n=len(rows)
print(f"draws={N} infeasible(Dv0<=0)={infeas} unstable/failed={unstable} usable={n}")
for k,lab in (("O1","DAT-null baseline DA > control"),("O2","control AMPH surge >3x"),("O3","DAT-null no surge (<=1.5x own baseline)"),("O4","fumin-like peak < control peak")):
    print(f"{lab:46s} {np.mean([r[k] for r in rows]):6.1%}")
allp=np.mean([r["O1"] and r["O2"] and r["O3"] and r["O4"] for r in rows]); print(f"{'ALL FOUR':46s} {allp:6.1%}")
mag=np.array([r["mag"] for r in rows]); pk=np.array([r["peak_uM"] for r in rows]); kb=np.array([r["ko_base"] for r in rows])
q=lambda a:np.percentile(a,[5,50,95]).round(2)
print("control peak x baseline  5/50/95%:",q(mag)); print("control peak (uM)        5/50/95%:",q(pk)); print("DAT-null baseline x ctrl 5/50/95%:",q(kb))
P=np.array([r["par"] for r in rows])
print("\nSpearman vs log control peak (uM):")
for j,nm in enumerate(names): print(f"  {nm:7s} {spearmanr(P[:,j],np.log(pk))[0]:+.2f}")
print("Spearman vs DAT-null baseline fold:")
for j,nm in enumerate(names): print(f"  {nm:7s} {spearmanr(P[:,j],kb)[0]:+.2f}")
