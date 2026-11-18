# SPDX-FileCopyrightText: 2026 Thorsten Bartel, Akshay Balachandran Jeeja, Bjoern Kiefer
# SPDX-License-Identifier: BSD-3-Clause
"""Homogeneous MSMA response with piecewise-monotone loading and sensitivities.

Adapted from ch10_msma_extd.py. This remains a stress-controlled homogeneous
material-point model with a constant demagnetization tensor and a fixed twin
normal, NOT the spatial FEM material routine discussed elsewhere.

Quick start (Python >=3.9; numpy, scipy, matplotlib):
    python msma_loadpaths.py
    python msma_loadpaths.py --backend cyipopt
    python msma_loadpaths.py --quick --no-show --out results_quick
    python msma_loadpaths.py --no-tangents --no-show

In Spyder: edit USER_STAGES / USER_SETTINGS below and run the file.
Stage targets: stresses in MPa (tension positive), external mu0*H in tesla.
Omitted components keep their preceding value. All specified components change
linearly and simultaneously during a stage. Use successive stages for reversals,
sequential loading, or a different time discretization. Equal endpoints mean a
hold. Loading is rate independent: time parametrizes the path, not viscosity.

POSTPROCESSING
Kelvin order: [11,22,33,sqrt(2)*12,sqrt(2)*23,sqrt(2)*13] for BOTH stress/strain.
The input field beta=mu0*H_ext [T] is NOT the magnetic induction B.
At each sampled path point, fixed-history finite differences determine
    S = d epsilon_K / d sigma_K |_beta        [1/Pa]
    P = d epsilon_K / d beta |_sigma         [1/T]
    Q = d m / d sigma_K |_beta               [A/m/Pa]
    X = d m / d beta |_sigma                 [A/m/T].
C = inv(S) is the algorithmic stiffness at fixed EXTERNAL magnetic field,
when S is invertible and sufficiently conditioned. This includes relaxation
of the internal variables. It is distinct from frozen-state elastic stiffness,
from a path slope, and from a tangent at fixed local total magnetic field.

The perturbed solves all start from the SAME previous converged state, not the
current state. Plus/minus sensitivities are also saved. At switching surfaces
there may be no unique derivative: the central result is still exported but
flagged, and C is withheld when the mechanical derivative is unreliable.
Finite differences are step-size dependent; adjust fd_sigma_pa / fd_beta_t and
compare results for scientific use. Local optimization + multiple starts gives
no guarantee of a global minimum. Sampled starts favor lower incremental energy.

OUTPUTS
  history.csv: time, full stress/strain/magnetization, states, solver diagnostics.
  tangents.csv: selected effective moduli and differential magnetic response.
  results.npz: full matrices, one-sided values, validity flags and states.
  settings.json: parameters, path, solver settings, units and array definitions.
  response.png, sensitivities.png: overview plots.
  cross_P.png, cross_Q.png: all magnetomechanical cross-sensitivity components.

Model retained: identical isotropic elastic tensors, two variants, four domains,
theta2=-theta1 and theta4=-theta3, analytical elimination of a at fixed phi.
Changes from the original: exact volume/angle bounds; separate smooth positive
and negative xi-increment subproblems instead of a shifted sign switch; explicit
solver checks; no global mutable material state; analytic objective gradients.
The original endpoint-evaluated hardening dissipation Y(xi)*abs(delta_xi) is
retained (not replaced by an integral dissipation distance). Dissipation can be
disabled stage by stage, as in the original prestress stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize, nnls, lsq_linear

MU0 = 4.0 * np.pi * 1e-7
SQRT2 = np.sqrt(2.0)
STRESS_KEYS = ("11", "22", "33", "12", "23", "13")
FIELD_KEYS = ("1", "2", "3")
KELVIN_FACTORS = np.array([1., 1., 1., SQRT2, SQRT2, SQRT2])
CONSTRAINT_JAC = np.array([[1., -1., 0., 0., 0.],
                           [1., 1., 0., 0., 0.],
                           [-1., 0., -1., 0., 0.],
                           [-1., 0., 1., 0., 0.]])
CONSTRAINT_OFFSET = np.array([0., 0., 1., 1.])


@dataclass
class Stage:
    """Targets at END of a monotone stage; omitted components are held."""
    duration: float
    steps: int
    stress_mpa: Dict[str, float] = field(default_factory=dict)
    mu0_h_t: Dict[str, float] = field(default_factory=dict)
    dissipative: bool = True
    name: str = ""


@dataclass
class Parameters:
    E: float = 5e9
    nu: float = 0.3
    msat: float = 514e3
    K: float = 0.1365e6
    bain_long: float = 0.0188
    bain_short: float = -0.0394
    phi: float = np.pi / 4
    demag_diagonal: tuple = (0.2154, (1-0.2154)/2, (1-0.2154)/2)
    Y0: float = 0.03e6
    DY: float = 0.029e6


@dataclass
class Settings:
    backend: str = "scipy"  # "scipy" (SLSQP), or "cyipopt" (optional dependency)
    maxiter: int = 600
    ftol: float = 1e-12  # dimensionless objective, scaled by energy_scale
    energy_scale: float = 1e5  # J/m^3
    feasibility_tol: float = 2e-8
    kkt_tol: float = 2e-5  # normalized gradient residual, checked independently
    multistart: bool = True
    tangent_every: int = 1  # 1 = every converged point; 0 = disabled
    fd_sigma_pa: float = 100.0  # perturbation of one Kelvin stress component
    fd_beta_t: float = 1e-4
    derivative_rtol: float = 0.05  # disagreement of forward/backward columns
    max_compliance_condition: float = 1e10
    initial_state: tuple = (0.99, 0.001, 0.001, 0., -np.pi/4)


# ======================== EDIT YOUR LOAD PATH HERE ========================
# This example reproduces the original sequence (small numerical changes are
# expected because unphysical constraint offsets and sign-switch shifts are gone).
USER_STAGES = [
    Stage(1.0, 5, stress_mpa={"11": -1.4},
          name="Apply compressive prestress"), # Dissipation kann auch durch 'dissipative=False' ausgeschaltet werden; Sinn aber fraglich
    Stage(0.5, 60, mu0_h_t={"2": 1.0}, name="Increase magnetic field"),
    Stage(0.5, 60, mu0_h_t={"2": 0.0}, name="Decrease magnetic field"),
]
# Example of simultaneous changes (append/replace stages as desired):
# Stage(1.0, 40, stress_mpa={"11": -2.0, "22": -0.5, "12": 0.1},
#       mu0_h_t={"1": 0.2, "2": 0.7}, name="Combined loading")
# Stage(0.5, 20, stress_mpa={"11": 0.0}, name="Unload 11; hold all others")
USER_INITIAL_STRESS_MPA = {"11": -0.001}
USER_INITIAL_MU0_H_T = {}
USER_SETTINGS = Settings()
USER_PARAMETERS = Parameters()
# ========================================================================


def component_vector(values, keys, scale=1.0):
    unknown = set(values) - set(keys)
    if unknown:
        raise ValueError("Unknown component(s): " + str(sorted(unknown)))
    out = np.array([values.get(k, 0.) for k in keys], dtype=float) * scale
    if not np.all(np.isfinite(out)):
        raise ValueError("Nonfinite load value")
    return out


def make_path(stages, initial_stress=None, initial_field=None):
    """Build time grid with every stage endpoint included exactly once."""
    stress = component_vector(initial_stress or {}, STRESS_KEYS, 1e6)
    beta = component_vector(initial_field or {}, FIELD_KEYS)
    times, loads, dissipative, labels = [0.], [np.r_[stress*KELVIN_FACTORS, beta]], [False], ["Initial equilibrium"]
    t = 0.
    for stage in stages:
        if (not np.isfinite(stage.duration) or stage.duration <= 0 or
                isinstance(stage.steps, bool) or int(stage.steps) != stage.steps or stage.steps < 1):
            raise ValueError("Each stage needs finite duration > 0 and integer steps >= 1")
        target_stress, target_beta = stress.copy(), beta.copy()
        component_vector(stage.stress_mpa, STRESS_KEYS)
        component_vector(stage.mu0_h_t, FIELD_KEYS)
        for key, value in stage.stress_mpa.items():
            target_stress[STRESS_KEYS.index(key)] = float(value)*1e6
        for key, value in stage.mu0_h_t.items():
            target_beta[FIELD_KEYS.index(key)] = float(value)
        for j in range(1, stage.steps+1):
            f = j/stage.steps
            times.append(t+f*stage.duration)
            loads.append(np.r_[(stress+f*(target_stress-stress))*KELVIN_FACTORS,
                               beta+f*(target_beta-beta)])
            dissipative.append(bool(stage.dissipative))
            labels.append(stage.name)
        stress, beta = target_stress, target_beta
        t += stage.duration
    return np.array(times), np.array(loads), np.array(dissipative), labels


def fractions(x):
    xi, eta1, eta2 = x[:3]
    return .5*np.array([xi+eta1, xi-eta1, 1-xi+eta2, 1-xi-eta2])


def magnetization(x, msat):
    """Return m and dm/dx for x=[xi1,eta1,eta2,theta1,theta3]."""
    xi, e1, e2, t1, t3 = x
    s1, c1, s3, c3 = np.sin(t1), np.cos(t1), np.sin(t3), np.cos(t3)
    m = msat*np.array([e1*c1-(1-xi)*s3, xi*s1+e2*c3, 0.])
    dm = msat*np.array([[s3, c1, 0., -e1*s1, -(1-xi)*c3],
                        [s1, 0., c3, xi*c1, -e2*s3],
                        [0., 0., 0., 0., 0.]])
    return m, dm


class Material:
    def __init__(self, parameters=None, settings=None):
        self.p = parameters or Parameters()
        self.s = settings or Settings()
        p, s = self.p, self.s
        if not (p.E > 0 and -1 < p.nu < .5 and p.msat > 0 and p.K >= 0 and p.Y0 >= 0 and p.DY >= 0):
            raise ValueError("Invalid material parameters")
        if s.backend not in ("scipy", "cyipopt"):
            raise ValueError("backend must be scipy or cyipopt")
        if s.energy_scale <= 0 or s.fd_sigma_pa <= 0 or s.fd_beta_t <= 0:
            raise ValueError("Energy scale and perturbations must be positive")
        if s.tangent_every < 0 or int(s.tangent_every) != s.tangent_every:
            raise ValueError("tangent_every must be an integer >= 0")
        lam = p.E*p.nu/((1+p.nu)*(1-2*p.nu))
        mu = p.E/(2*(1+p.nu))
        self.C_el = 2*mu*np.eye(6)
        self.C_el[:3, :3] += lam
        self.S_el = np.linalg.inv(self.C_el)
        self.e1 = np.array([p.bain_short, p.bain_long, p.bain_long, 0., 0., 0.])
        self.e2 = np.array([p.bain_long, p.bain_short, p.bain_long, 0., 0., 0.])
        self.delta = self.e1-self.e2
        self.N = np.diag(np.asarray(p.demag_diagonal, dtype=float))
        if self.N.shape != (3, 3) or np.any(np.diag(self.N) < 0) or not np.all(np.isfinite(self.N)):
            raise ValueError("demag_diagonal needs three finite nonnegative entries")
        n = np.array([np.cos(p.phi), np.sin(p.phi), 0.])
        # Kelvin map a -> sym(a tensor n). Exact elimination with common C.
        B = np.zeros((6, 3))
        for j in range(3):
            v = np.eye(3)[j]
            A = .5*(np.outer(v, n)+np.outer(n, v))
            B[:, j] = np.array([A[0,0], A[1,1], A[2,2], SQRT2*A[0,1], SQRT2*A[1,2], SQRT2*A[0,2]])
        self.a = np.linalg.solve(B.T@self.C_el@B, B.T@self.C_el@self.delta)
        self.A = B@self.a
        residual = self.A-self.delta
        self.laminate_penalty = float(residual@self.C_el@residual)
        if s.backend == "cyipopt":
            try:
                import cyipopt
            except ImportError as exc:
                raise ImportError("Install cyipopt or set backend='scipy'.") from exc
            self.cyipopt = cyipopt

    def energy_gradient(self, x, q, old, dissipative, branch):
        """Smooth branch of the stress-controlled incremental Gibbs density.

        The x-independent term -.5*sigma:S:sigma is omitted to reduce cancellation.
        branch=+1 restricts xi>=old_xi; branch=-1 restricts xi<=old_xi.
        """
        xi, _, _, t1, t3 = x
        sigma, beta = q[:6], q[6:]
        p = self.p
        m, dm = magnetization(x, p.msat)
        bar = self.e2+xi*self.delta
        value = (-sigma@bar + .5*xi*(1-xi)*self.laminate_penalty
                 + p.K*(xi*np.sin(t1)**2+(1-xi)*np.sin(t3)**2)
                 + .5*MU0*m@self.N@m-beta@m)
        g = dm.T@(MU0*self.N@m-beta)
        g[0] += (-sigma@self.delta+.5*(1-2*xi)*self.laminate_penalty
                 + p.K*(np.sin(t1)**2-np.sin(t3)**2))
        g[3] += p.K*xi*np.sin(2*t1)
        g[4] += p.K*(1-xi)*np.sin(2*t3)
        if dissipative and branch != 0:
            dx = xi-old[0]
            if branch == 1:
                Y = p.Y0+p.DY*xi
                value += Y*dx
                g[0] += Y+p.DY*dx
            else:
                Y = p.Y0+p.DY*(1-xi)
                value -= Y*dx
                g[0] += -Y+p.DY*dx
        return value/self.s.energy_scale, g/self.s.energy_scale

    @staticmethod
    def feasible_seed(x, lo, hi):
        x = np.clip(np.asarray(x, float), lo, hi)
        x[1] = np.clip(x[1], -x[0], x[0])
        x[2] = np.clip(x[2], -(1-x[0]), 1-x[0])
        return x

    @staticmethod
    def kkt_residual(x, g, lo, hi, active_tol=2e-6):
        """Gradient residual with nonnegative active-constraint multipliers."""
        rows = [row for row, c in zip(CONSTRAINT_JAC, CONSTRAINT_JAC@x+CONSTRAINT_OFFSET) if c <= active_tol]
        for j in range(5):
            unit = np.eye(5)[j]
            if x[j]-lo[j] <= active_tol:
                rows.append(unit)
            if hi[j]-x[j] <= active_tol:
                rows.append(-unit)
        if not rows:
            return float(np.linalg.norm(g, np.inf))
        A = np.array(rows).T
        try:
            multipliers, _ = nnls(A, g, maxiter=1000)
            return float(np.linalg.norm(g-A@multipliers, np.inf))
        except RuntimeError:
            return float("inf")

    def physical_kkt(self, x, q, old, dissipative):
        """Stationarity of the FULL problem, including the dissipation interval.

        Artificial bounds xi>=xi_n / xi<=xi_n do not certify stationarity of
        the complete problem at sticking. Use the subdifferential there.
        """
        lo = np.array([0.,-1.,-1.,-np.pi/2,-np.pi/2])
        hi = np.array([1.,1.,1.,np.pi/2,np.pi/2])
        dx = x[0]-old[0]
        if not dissipative or abs(dx) > 2e-8:
            _, g = self.energy_gradient(x,q,old,dissipative,1 if dx >= 0 else -1)
            return self.kkt_residual(x,g,lo,hi)
        _, g = self.energy_gradient(x,q,old,False,1)
        rows = [row for row,c in zip(CONSTRAINT_JAC,CONSTRAINT_JAC@x+CONSTRAINT_OFFSET) if c <= 2e-6]
        for j in range(5):
            if x[j]-lo[j] <= 2e-6: rows.append(np.eye(5)[j])
            if hi[j]-x[j] <= 2e-6: rows.append(-np.eye(5)[j])
        A = np.array(rows).T if rows else np.empty((5,0))
        A = np.column_stack([A,-np.eye(5)[0]])
        yl = -(self.p.Y0+(1-x[0])*self.p.DY)/self.s.energy_scale
        yu = (self.p.Y0+x[0]*self.p.DY)/self.s.energy_scale
        if yu-yl < 1e-14:
            return self.kkt_residual(x,g,lo,hi)
        fit = lsq_linear(A,g,bounds=(np.r_[np.zeros(len(rows)),yl],
                                     np.r_[np.full(len(rows),np.inf),yu]),
                         tol=1e-12,max_iter=300)
        return float(np.linalg.norm(A@fit.x-g,np.inf))

    def solve(self, q, old, dissipative=True, seed=None):
        """No state mutation. Raise rather than silently commit failed solves."""
        q, old = np.asarray(q, float), np.asarray(old, float)
        if q.shape != (9,) or old.shape != (5,) or not np.all(np.isfinite(np.r_[q, old])):
            raise ValueError("Expected finite q[9] and old[5]")
        if np.min(fractions(old)) < -self.s.feasibility_tol or np.max(np.abs(old[3:])) > np.pi/2+self.s.feasibility_tol:
            raise ValueError("Previous state violates physical bounds")
        candidates, messages = [], []
        # Explicit sticking candidate: fixed xi avoids ill-scaled constrained
        # steps when yield resistance is large. Slip branches still compete.
        branches = (0, 1, -1) if dissipative else (1,)
        for branch in branches:
            lo = np.array([0., -1., -1., -np.pi/2, -np.pi/2])
            hi = np.array([1., 1., 1., np.pi/2, np.pi/2])
            if dissipative:
                if branch == 0:
                    lo[0] = hi[0] = old[0]
                elif branch == 1:
                    lo[0] = old[0]
                else:
                    hi[0] = old[0]
            starts = [seed if seed is not None else old]
            if self.s.multistart:
                starts += [old, np.array([.05, 0., 0., .13, -.17]),
                           np.array([.95, 0., 0., -.11, .19])]
            unique = []
            for initial in starts:
                x0 = self.feasible_seed(initial, lo, hi)
                if any(np.max(np.abs(x0-z)) < 1e-12 for z in unique):
                    continue
                unique.append(x0)
                fg = lambda x: self.energy_gradient(x, q, old, dissipative, branch)
                if self.s.backend == "scipy":
                    result = minimize(fg, x0, method="SLSQP", jac=True,
                                      bounds=list(zip(lo, hi)),
                                      constraints={"type": "ineq", "fun": lambda x: CONSTRAINT_JAC@x+CONSTRAINT_OFFSET,
                                                   "jac": lambda x: CONSTRAINT_JAC},
                                      options={"ftol": self.s.ftol, "maxiter": self.s.maxiter})
                    x, ok, msg = result.x, result.success, str(result.message)
                else:
                    class Problem:
                        def objective(self, x): return fg(x)[0]
                        def gradient(self, x): return fg(x)[1]
                        def constraints(self, x): return CONSTRAINT_JAC@x+CONSTRAINT_OFFSET
                        def jacobian(self, x): return CONSTRAINT_JAC.ravel()
                    nlp = self.cyipopt.Problem(n=5, m=4, problem_obj=Problem(), lb=lo, ub=hi,
                                               cl=np.zeros(4), cu=np.full(4, 1e19))
                    nlp.add_option("hessian_approximation", "limited-memory")
                    nlp.add_option("max_iter", self.s.maxiter)
                    nlp.add_option("tol", 1e-9)
                    nlp.add_option("constr_viol_tol", 1e-10)
                    nlp.add_option("bound_relax_factor", 0.)
                    nlp.add_option("print_level", 0)
                    nlp.add_option("sb", "yes")
                    x, info = nlp.solve(x0)
                    nlp.close()
                    ok, msg = info["status"] in (0, 1), str(info["status_msg"])
                value, g = fg(x)
                violation = max(0., -np.min(CONSTRAINT_JAC@x+CONSTRAINT_OFFSET),
                                np.max(lo-x), np.max(x-hi))
                residual = self.physical_kkt(x,q,old,dissipative)
                if (ok and np.isfinite(value) and violation <= self.s.feasibility_tol
                        and residual <= self.s.kkt_tol):
                    candidates.append((value, x.copy(), residual, msg))
                else:
                    messages.append(f"branch {branch}: {msg}; violation={violation:.2e}; KKT={residual:.2e}")
        if not candidates:
            raise RuntimeError("No valid local minimizer. " + " | ".join(messages))
        best = min(candidates, key=lambda c: c[0])
        x = best[1]
        m, _ = magnetization(x, self.p.msat)
        eps = self.S_el@q[:6]+self.e2+x[0]*self.delta
        # Include the omitted x-independent elastic term in reported potential.
        potential = best[0]*self.s.energy_scale-.5*q[:6]@self.S_el@q[:6]
        return {"x": x, "eps": eps, "m": m, "potential": potential,
                "kkt": best[2], "candidates": len(candidates), "message": best[3]}

    def sensitivities(self, q, old, base, dissipative=True):
        """Fixed-history partial derivatives, not differences along the path."""
        scale = np.r_[np.full(6, self.s.fd_sigma_pa), np.full(3, self.s.fd_beta_t)]
        y0 = np.r_[base["eps"], base["m"]]
        plus, minus = np.full((9,9), np.nan), np.full((9,9), np.nan)
        valid = np.zeros(9, bool)
        mechanical_valid = np.zeros(6, bool)
        mismatch = np.full(9, np.nan)
        failures = []
        for j, h in enumerate(scale):
            # With the retained common elasticity, stresses orthogonal to the
            # transformation-strain difference do not affect internal evolution.
            # Likewise beta3 cannot act on the strictly in-plane magnetization.
            # Use these exact derivatives instead of reoptimization noise.
            if j < 6 and abs(self.delta[j]) < 1e-15:
                plus[:,j] = minus[:,j] = np.r_[self.S_el[:,j],np.zeros(3)]
                valid[j], mismatch[j] = True, 0.
                mechanical_valid[j] = True
                continue
            if j == 8:
                plus[:,j] = minus[:,j] = 0.
                valid[j], mismatch[j] = True, 0.
                continue
            dq = np.zeros(9)
            dq[j] = h
            try:
                rp = self.solve(q+dq, old, dissipative, seed=base["x"])
                rm = self.solve(q-dq, old, dissipative, seed=base["x"])
            except RuntimeError as exc:
                failures.append(f"column {j}: {exc}")
                continue
            plus[:,j] = (np.r_[rp["eps"], rp["m"]]-y0)/h
            minus[:,j] = (y0-np.r_[rm["eps"], rm["m"]])/h
            # Compare actual response increments in dimensionless output units.
            outscale = np.r_[np.full(6, .06), np.full(3, self.p.msat)]
            dp, dm = plus[:,j]*h/outscale, minus[:,j]*h/outscale
            mismatch[j] = np.linalg.norm(dp-dm)/max(np.linalg.norm(dp), np.linalg.norm(dm), 1e-7)
            valid[j] = mismatch[j] <= self.s.derivative_rtol
            if j < 6:
                mechanical_error = np.linalg.norm(dp[:6]-dm[:6])/max(np.linalg.norm(dp[:6]),np.linalg.norm(dm[:6]),1e-7)
                mechanical_valid[j] = mechanical_error <= self.s.derivative_rtol
        J = .5*(plus+minus)
        S = J[:6,:6]
        C = np.full((6,6), np.nan)
        condition = float("inf")
        if np.all(np.isfinite(S)):
            condition = float(np.linalg.cond(S))
            if np.all(mechanical_valid) and condition < self.s.max_compliance_condition:
                C = np.linalg.solve(S, np.eye(6))
        return {"J": J, "J_plus": plus, "J_minus": minus, "valid": valid,
                "mechanical_valid": mechanical_valid, "mismatch": mismatch,
                "C": C, "condition": condition, "failures": failures}


def simulate(stages=None, parameters=None, settings=None, initial_stress=None,
             initial_field=None, progress=True):
    """Return histories and derivative arrays; no plots or writes on import."""
    stages = USER_STAGES if stages is None else stages
    settings = USER_SETTINGS if settings is None else settings
    parameters = USER_PARAMETERS if parameters is None else parameters
    if initial_stress is None:
        initial_stress = USER_INITIAL_STRESS_MPA
    if initial_field is None:
        initial_field = USER_INITIAL_MU0_H_T
    mat = Material(parameters, settings)
    times, loads, dissipative, labels = make_path(stages, initial_stress, initial_field)
    n = len(times)
    data = {"time": times, "load": loads, "dissipative": dissipative,
            "state": np.empty((n,5)), "state_n": np.empty((n,5)),
            "strain_K": np.empty((n,6)), "magnetization": np.empty((n,3)),
            "potential": np.empty(n), "kkt": np.empty(n), "candidates": np.empty(n,int),
            "J": np.full((n,9,9), np.nan), "J_plus": np.full((n,9,9), np.nan),
            "J_minus": np.full((n,9,9), np.nan), "C_K": np.full((n,6,6), np.nan),
            "derivative_valid": np.zeros((n,9),bool), "derivative_sampled": np.zeros(n,bool),
            "mechanical_derivative_valid": np.zeros((n,6),bool),
            "derivative_mismatch": np.full((n,9),np.nan),
            "compliance_condition": np.full(n,np.nan),
            "C_el_K": mat.C_el, "S_el_K": mat.S_el, "a": mat.a}
    old = np.array(settings.initial_state, dtype=float)
    derivative_failures = []
    for i, (t, q, diss) in enumerate(zip(times, loads, dissipative)):
        base = mat.solve(q, old, bool(diss))
        data["state_n"][i] = old
        data["state"][i] = base["x"]
        data["strain_K"][i] = base["eps"]
        data["magnetization"][i] = base["m"]
        for key in ("potential", "kkt", "candidates"):
            data[key][i] = base[key]
        if settings.tangent_every and (i % settings.tangent_every == 0 or i == n-1):
            tangent = mat.sensitivities(q, old, base, bool(diss))
            for key in ("J", "J_plus", "J_minus"):
                data[key][i] = tangent[key]
            data["C_K"][i] = tangent["C"]
            data["derivative_valid"][i] = tangent["valid"]
            data["mechanical_derivative_valid"][i] = tangent["mechanical_valid"]
            data["derivative_mismatch"][i] = tangent["mismatch"]
            data["compliance_condition"][i] = tangent["condition"]
            data["derivative_sampled"][i] = True
            derivative_failures += [f"point {i}, t={t}: {msg}" for msg in tangent["failures"]]
        old = base["x"].copy()  # commit only the unperturbed converged path state
        if progress and (i % max(1,n//20) == 0 or i == n-1):
            print(f"{i+1:4d}/{n}: t={t:.5g}, xi1={old[0]:.6f}, KKT={base['kkt']:.2e}", flush=True)
    metadata = {"parameters": asdict(parameters), "settings": asdict(settings),
                "stages": [asdict(s) for s in stages], "initial_stress_mpa": initial_stress,
                "initial_mu0_h_t": initial_field, "labels": labels,
                "derivative_failures": derivative_failures,
                "state_order": ["xi1","eta1","eta2","theta1","theta3"],
                "kelvin_order": ["11","22","33","sqrt(2)*12","sqrt(2)*23","sqrt(2)*13"],
                "J_rows": ["epsilon_K[0:6]", "magnetization[0:3]"],
                "J_columns": ["sigma_K[0:6] in Pa", "beta=mu0*H_ext[0:3] in T"],
                "J_blocks": {"S": "J[:6,:6], 1/Pa", "P": "J[:6,6:], 1/T",
                             "Q": "J[6:,:6], (A/m)/Pa", "X": "J[6:,6:], (A/m)/T"},
                "C_K": "inverse of S at fixed external field; NaN if invalid/ill-conditioned",
                "validity_flags": "derivative_valid tests entire response column; mechanical_derivative_valid tests only strain response for stiffness inversion",
                "derivative_definition": "algorithmic finite differences at fixed state_n and current endpoint; not path slopes",
                "field_warning": "beta is mu0*H_ext, NOT induction B; constant demagnetization tensor retained",
                "potential_units": "J/m^3, stress-controlled incremental potential including dissipation"}
    return data, metadata


def export_results(data, metadata, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"results.npz", **data)
    (out/"settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (out/"history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time"]+["sigma"+k+"_MPa" for k in STRESS_KEYS]
                        +["mu0_Hext"+k+"_T" for k in FIELD_KEYS]
                        +["eps"+k for k in STRESS_KEYS]+["m"+k+"_A_per_m" for k in FIELD_KEYS]
                        +["xi1","xi2","eta1","eta2","theta1_rad","theta3_rad",
                          "potential_J_per_m3","KKT_scaled","valid_candidates","stage"])
        for i, t in enumerate(data["time"]):
            x = data["state"][i]
            writer.writerow([t]+list(data["load"][i,:6]/KELVIN_FACTORS/1e6)
                            +list(data["load"][i,6:])+list(data["strain_K"][i]/KELVIN_FACTORS)
                            +list(data["magnetization"][i])+[x[0],1-x[0],*x[1:],
                             data["potential"][i],data["kkt"][i],data["candidates"][i],metadata["labels"][i]])
    with (out/"tangents.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time","sampled","C11_GPa","C22_GPa","C33_GPa",
                         "E1_free_transverse_GPa","E2_free_transverse_GPa","E3_free_transverse_GPa",
                         "E1_forward_GPa","E1_backward_GPa",
                         "mu0_dm1_d_beta1","mu0_dm2_d_beta2","mu0_dm3_d_beta3",
                         "S_condition"]+["valid_"+str(j) for j in range(9)])
        for i,t in enumerate(data["time"]):
            S = data["J"][i,:6,:6]
            E = np.full(3,np.nan)
            for j in range(3):
                if data["mechanical_derivative_valid"][i,j] and np.isfinite(S[j,j]) and S[j,j] > 0:
                    E[j] = 1/S[j,j]/1e9
            directional_E = []
            for key in ("J_plus","J_minus"):
                sij = data[key][i,0,0]
                directional_E.append(1/sij/1e9 if np.isfinite(sij) and sij > 0 else np.nan)
            writer.writerow([t,int(data["derivative_sampled"][i]),*list(np.diag(data["C_K"][i])[:3]/1e9),
                             *list(E),*directional_E,*list(MU0*np.diag(data["J"][i,6:,6:])),
                             data["compliance_condition"][i],*list(data["derivative_valid"][i].astype(int))])
    flagged = data["derivative_sampled"] & ~np.all(data["derivative_valid"],axis=1)
    if np.any(flagged):
        warnings.warn(f"{flagged.sum()} sampled points have nonunique/noisy/failed derivative columns. Inspect one-sided matrices and flags.")


def plot_cross_sensitivities(data, out):
    """Plot P=d(eps)/d(beta) and Q=d(m)/d(sigma), beta=mu0*H_ext.

    Display physical tensor components rather than Kelvin components:
    P_tensor = P_K / Kelvin factor; Q_per_MPa = Q_K * Kelvin factor * 1e6.
    Unreliable sampled derivatives remain gaps. Unsampled times are omitted.
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampled = np.asarray(data["derivative_sampled"], dtype=bool)
    if not np.any(sampled):
        return
    t = data["time"][sampled]
    J = data["J"][sampled]
    valid = data["derivative_valid"][sampled]
    P = J[:, :6, 6:] / KELVIN_FACTORS[None, :, None]
    Q = J[:, 6:, :6] * KELVIN_FACTORS[None, None, :] * 1e6

    fig_p, axes_p = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig_q, axes_q = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    fig_p.suptitle(r"P: strain response to external magnetic field (constant stress)")
    fig_q.suptitle(r"Q: magnetization response to stress (constant external field)")
    for i, ij in enumerate(STRESS_KEYS):
        ap, aq = axes_p.flat[i], axes_q.flat[i]
        for k in range(3):
            p_values = np.where(valid[:, 6+k], P[:, i, k], np.nan)
            q_values = np.where(valid[:, i], Q[:, k, i], np.nan)
            ap.plot(t, p_values, label=rf"$\beta_{k+1}$")
            aq.plot(t, q_values, label=rf"$m_{k+1}$")
        ap.set_ylabel(rf"$\partial\varepsilon_{{{ij}}}/\partial\beta_k$ [1/T]")
        aq.set_ylabel(rf"$\partial m_k/\partial\sigma_{{{ij}}}$ [(A/m)/MPa]")
        for ax in (ap, aq):
            ax.set_xlabel("time")
            ax.grid(True)
            ax.legend()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig_p.savefig(out / "cross_P.png", dpi=150)
    fig_q.savefig(out / "cross_Q.png", dpi=150)


def plot_results(data, out, show=True):
    import matplotlib.pyplot as plt
    out = Path(out)
    t = data["time"]
    fig, axes = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for j,k in enumerate(STRESS_KEYS):
        axes[0,0].plot(t,data["load"][:,j]/KELVIN_FACTORS[j]/1e6,label=k)
        axes[1,0].plot(t,data["strain_K"][:,j]/KELVIN_FACTORS[j],label=k)
    for j,k in enumerate(FIELD_KEYS):
        axes[0,1].plot(t,data["load"][:,6+j],label=k)
        axes[1,1].plot(t,MU0*data["magnetization"][:,j],label=k)
    axes[2,0].plot(t,data["state"][:,0],label="xi1")
    axes[2,0].plot(t,1-data["state"][:,0],label="xi2")
    # Original style field-response plot, while time plots also support general paths.
    axes[2,1].plot(data["load"][:,7],data["strain_K"][:,0]-data["strain_K"][0,0],label="delta eps11")
    axes[2,1].plot(data["load"][:,7],data["strain_K"][:,1]-data["strain_K"][0,1],label="delta eps22")
    labels = ["stress [MPa]","external mu0 H [T]","strain [-]","mu0 m [T]","variant fractions [-]","strain relative to initial state [-]"]
    for ax,label in zip(axes.flat,labels):
        ax.set_ylabel(label);ax.set_xlabel("time");ax.grid(True);ax.legend()
    axes[2,1].set_xlabel("external mu0 H2 [T]")
    fig.savefig(out/"response.png",dpi=150)
    if np.any(data["derivative_sampled"]):
        fig2, ax = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for j in range(3):
            ax[0].plot(t,data["C_K"][:,j,j]/1e9,label=f"C{j+1}{j+1}")
            chi=MU0*data["J"][:,6+j,6+j].copy()
            chi[~data["derivative_valid"][:,6+j]]=np.nan
            ax[1].plot(t,chi,label=f"component {j+1}")
        ax[0].set_ylabel("algorithmic stiffness at fixed external field [GPa]")
        ax[1].set_ylabel("mu0 dm_i / d(mu0 Hext_i) [-]")
        for a in ax:
            a.set_xlabel("time");a.grid(True);a.legend()
        fig2.savefig(out/"sensitivities.png",dpi=150)
        plot_cross_sensitivities(data, out)
    if show:
        plt.show()
    else:
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend",choices=["scipy","cyipopt"],default=USER_SETTINGS.backend)
    parser.add_argument("--out",default="msma_results")
    parser.add_argument("--no-show",action="store_true")
    parser.add_argument("--no-tangents",action="store_true")
    parser.add_argument("--quick",action="store_true",help="Only 3 increments per stage; for smoke tests")
    args = parser.parse_args()
    settings = Settings(**asdict(USER_SETTINGS))
    settings.backend = args.backend
    if args.no_tangents:
        settings.tangent_every = 0
    stages = USER_STAGES
    if args.quick:
        stages = [Stage(**{**asdict(s),"steps":3}) for s in stages]
    data, meta = simulate(stages, settings=settings)
    export_results(data,meta,args.out)
    plot_results(data,args.out,show=not args.no_show)
    print("Results saved to",Path(args.out).resolve())


if __name__ == "__main__":
    main()
