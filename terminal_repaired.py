"""Mass-conserving variant of the DA terminal ODE in meth_neurodiv_model.py.

The original module is imported for its parameter dataclass and temperature
gate only; it is NOT edited.  Changes vs. the original da_terminal_odes (each
is a missing term, not a new mechanism):
  1. dDc: add the packing loss  -k_pack_eff*Dc   (original adds it to Dv only).
  2. dDv: subtract release R     (original never drains the vesicle pool).
  3. dDv: add repackaged uptake (1-eta_eff)*U   (paper text: eta is the fraction
          going to cytoplasm "rather than vesicular"; the rest was dropped).
  4. dDc: add synthesis S = S0*N (a source is required for any steady state;
          scales with terminal integrity N so damage shifts the setpoint).
  5. Unit conversion for release is DERIVED from the baseline balance, not fit.
  6. Dv is in uM-equivalents; its baseline is derived (the paper says Dv is
     'normalized', so its absolute scale is free).
Integrator: stiff (Radau), not explicit Euler dt=0.5.
"""
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meth_neurodiv_model as m  # noqa: E402

ETA = 0.1  # fraction of uptake entering cytoplasm (original hard-codes 0.1)


def derive(p):
    """Constants fixed by requiring the stated baselines to be a steady state."""
    U0 = p.Vmax * p.De_baseline / (p.Km + p.De_baseline)
    S0 = p.k_MAO * p.Dc_baseline + p.k_diff * p.De_baseline
    R0 = U0 + p.k_diff * p.De_baseline
    L0 = (p.k_MAO + p.k_pack) * p.Dc_baseline - ETA * U0 - S0
    return dict(S0=S0, R0=R0, Dv0=L0 / p.k_leak,
                conv=R0 / (p.q * p.p * p.F_tonic))  # uM per released molecule/s


def rhs(y, p, D, conc=0.0, drug="none", T=37.0, vscale=1.0, closed=False, escale=1.0):
    De, Dc, Dv, Q, N = y
    De, Dc, Dv = max(De, 0.0), max(Dc, 0.0), max(Dv, 0.0)
    N = float(np.clip(N, 0.0, 1.0))
    Vm = p.Vmax * vscale
    if drug in ("AMPH", "METH"):
        P = conc**p.n_hill / (p.K_AMPH**p.n_hill + conc**p.n_hill)
        E = escale * p.k_rev * P * Dc * N
        U = Vm * De / (p.Km + De) * (1.0 - P)
        kp = p.k_pack * (1.0 - (0.9 if drug == "METH" else 0.5) * P)
    elif drug == "MPH":
        E, kp = 0.0, p.k_pack
        U = Vm * De / (p.Km * (1.0 + conc / 0.1) + De)
    else:
        E, kp = 0.0, p.k_pack
        U = Vm * De / (p.Km + De)
    R = p.q * p.p * p.F_tonic * D["conv"] * (Dv / D["Dv0"]) * N
    eta = 1.0 - (1.0 - ETA) * kp / p.k_pack          # repackaged share falls with VMAT2 block
    L = p.k_leak * Dv
    S = 0.0 if closed else D["S0"] * N
    mao = 0.0 if closed else p.k_MAO
    dif = 0.0 if closed else p.k_diff
    dDe = R + E - U - dif * De
    dDc = L + S + eta * U - mao * Dc - kp * Dc - E
    dDv = kp * Dc + (1.0 - eta) * U - L - R
    k_ox = p.k0_ox * p.Q10 ** ((T - 37.0) / 10.0)
    theta = 1.0 / (1.0 + np.exp(-(T - 40.0) / 0.5))
    return np.array([dDe, dDc, dDv, k_ox * Dc * (T / 37.0), -p.k_deg * Q * theta])


def setup(vscale=1.0):
    arch = m.make_neurotypical()
    p = arch.terminal
    tox = arch.toxicity
    p.k0_ox, p.Q10, p.k_deg = tox.k0_ox, tox.Q10, tox.k_deg
    return arch, p, derive(p)


def fixed_point(p, D, vscale=1.0):
    g = lambda x: rhs([*x, 0.0, 1.0], p, D, vscale=vscale)[:3]
    x0 = [p.De_baseline, p.Dc_baseline, D["Dv0"]]
    x = fsolve(g, x0, xtol=1e-13)
    return x, g(x)


def jacobian(p, D, x, vscale=1.0, h=1e-7):
    g = lambda z: rhs([*z, 0.0, 1.0], p, D, vscale=vscale)[:3]
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = h * max(1.0, abs(x[i]))
        J[:, i] = (g(x + d) - g(x - d)) / (2 * d[i])
    return J


def simulate(p, D, drug, conc, duration=3600.0, vscale=1.0, y0=None, temp_rise=3.0, tau_T=1800.0, escale=1.0):
    """Same drug/temperature profiles as the original simulate_acute_da_response."""
    x, _ = fixed_point(p, D, vscale)
    y0 = np.array([*x, 0.0, 1.0]) if y0 is None else y0

    def c(t):
        if t < 60: return conc * (1 - np.exp(-t / 15.0))
        if t < duration * 0.7: return conc * np.exp(-(t - 60.0) / (duration * 0.5))
        return conc * 0.01

    def T(t):
        r = temp_rise * (1 - np.exp(-t / tau_T))
        return 37.0 + (r if drug == "METH" else r * 0.3)

    f = lambda t, y: rhs(y, p, D, c(t), drug, T(t), vscale, escale=escale)
    return solve_ivp(f, [0, duration], y0, method="Radau", rtol=1e-8, atol=1e-12,
                     max_step=5.0, dense_output=False), x
