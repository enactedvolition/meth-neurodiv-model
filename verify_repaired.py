import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terminal_repaired as tr
from scipy.integrate import solve_ivp

arch, p, D = tr.setup()
print("derived:", {k: round(v, 6) for k, v in D.items()})
ok = {}

# C1 conservation in a closed system, random states, with and without drug
rng = np.random.default_rng(0)
worst = 0.0
for drug, c in [("none", 0.0), ("METH", 1.0), ("AMPH", 1.0), ("MPH", 0.3)]:
    for _ in range(50):
        y = [rng.uniform(0, 2), rng.uniform(0, 5), rng.uniform(0, 100), 0, 1]
        d = tr.rhs(y, p, D, c, drug, closed=True)
        worst = max(worst, abs(d[:3].sum()))
ok["C1 closed-system |d(De+Dc+Dv)/dt| max"] = (worst, worst < 1e-12)

# C2 fixed point at stated baselines
x, res = tr.fixed_point(p, D)
ok["C2 fixed point residual"] = (np.abs(res).max(), np.abs(res).max() < 1e-9)
print("fixed point (De,Dc,Dv):", x, " stated baselines:", p.De_baseline, p.Dc_baseline)

# C3 eigenvalues
J = tr.jacobian(p, D, x)
ev = np.linalg.eigvals(J)
print("eigenvalues:", ev)
ok["C3 max Re(eig)"] = (ev.real.max(), ev.real.max() < 0)

# C4 perturbation recovery
tmax = 8.0 / abs(ev.real.max())
worstrel = 0.0
for sgn in (+0.5, -0.5):
    for i in range(3):
        z = x.copy(); z[i] *= 1 + sgn
        s = solve_ivp(lambda t, y: tr.rhs([*y, 0.0, 1.0], p, D)[:3], [0, tmax], z,
                      method="Radau", rtol=1e-10, atol=1e-13)
        worstrel = max(worstrel, np.max(np.abs(s.y[:, -1] / x - 1)))
ok["C4 worst residual |y_end/x-1| after perturbation"] = (worstrel, worstrel < 1e-2)

# C5 METH dose ordering (peak De, fold of fixed point)
print("\nMETH peak/mean extracellular DA, fold of baseline (1 h):")
peaks = []
for c in [0.1, 0.5, 1.0, 2.0, 5.0]:
    s, x0 = tr.simulate(p, D, "METH", c)
    pk = s.y[0].max() / x0[0]
    tpk = s.t[s.y[0].argmax()]
    peaks.append(pk)
    print(f"  {c:4.1f} uM  peak x{pk:8.2f} at t={tpk:6.0f}s   mean x{np.trapezoid(s.y[0], s.t)/s.t[-1]/x0[0]:7.2f}   end N={s.y[4,-1]:.3f}  max Dc={s.y[1].max():.2f}")
ok["C5 METH peak De monotone in dose"] = (peaks, all(np.diff(peaks) > 0))

# C6 low-DAT baseline ordering (fumin direction), fixed release machinery
print("\nlow-DAT, no drug: fixed point and stability")
de = []
for vs in [1.0, 0.5, 0.25, 0.1, 0.05]:
    xx, rr = tr.fixed_point(p, D, vs)
    e = np.linalg.eigvals(tr.jacobian(p, D, xx, vs)).real.max()
    de.append(xx[0])
    print(f"  Vmax x{vs:<5} De*={xx[0]:.4f} uM (x{xx[0]/x[0]:.2f})  Dc*={xx[1]:.3f}  Dv*={xx[2]:.1f}  resid={np.abs(rr).max():.1e}  max Re(eig)={e:.3f}")
ok["C6 baseline De monotone as Vmax falls"] = (de, all(np.diff(de) > 0))

print("\nRESULTS")
for k, (v, b) in ok.items():
    print(("PASS " if b else "FAIL ") + k, v if not isinstance(v, list) else [round(float(a), 4) for a in v])
sys.exit(0 if all(b for _, b in ok.values()) else 1)
