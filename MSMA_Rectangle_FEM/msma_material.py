# SPDX-FileCopyrightText: 2026 Thorsten Bartel, Akshay Balachandran Jeeja, Bjoern Kiefer
# SPDX-License-Identifier: BSD-3-Clause
"""Local MSMA material update for a small-strain u / scalar-potential FEM.

Python >=3.9; numpy + scipy. Matplotlib is optional, only for the demo plots.
Run this file in Spyder to replay material_test_inputs.csv next to this file.
Or: python msma_material.py --csv path/to/material_test_inputs.csv
The original plane-strain example remains available with --demo.

DEFAULT INPUT: eps=[eps_xx,eps_yy,eps_xy] (tensor shear, NOT gamma_xy), h=[hx,hy]
with h the LOCAL TOTAL field in A/m, not mu0*h, not H_external, not induction b.
Plane strain: eps_zz=eps_xz=eps_yz=0. Transformation strain in z and sigma_zz
are retained in the 3-D constitutive calculation. Output sigma is a 2x2 tensor,
b is a 2-vector in T, with full 3-D quantities also available.

UPDATE: result=material.update(eps,h,state_n,compute_tangent=True)
No mutation/automatic commit: set state_n=result.state ONLY after convergence
of a global load step. Reuse the same state_n in all Newton/line-search calls.
The optional seed is only an optimizer initial guess, never the history.

ENERGY at fixed eps,h: minimize psi(eps,p)-mu0*h.m(p)+d(p,p_n).
No demagnetization tensor, no prescribed-stress work. psi is the two-variant
first-order laminate from section 3 of Bartel et al., IJSS 182-183 (2020).
Common isotropic elastic tensors are retained from the homogeneous code.
a is eliminated exactly for each phi; theta2=-theta1, theta4=-theta3.
phi can evolve; Y_phi is configurable and defaults to 0 (reversible orientation,
not a calibrated dissipation parameter). Set evolve_phi=False to reproduce the
fixed orientation of msma_loadpaths.py. xi dissipation retains the paper's
endpoint-discretized hardening, not an integral hardening distance.

TOTAL induction b=mu0*(h+m). The vacuum term -.5*mu0*h.h does not affect the
local minimizer but is required in the total-field potential whose derivative
gives b. For the exact split-potential functional (3), use
b_tilde=b-mu0*H_external in the magnetic residual, and vacuum elements in air.

TANGENTS use an analytic Hessian and active-set static condensation of internal
variables. They are derivatives w.r.t. strain/field, NOT nodal FEM dofs.
Default tangent convention uses engineering strain e=[eps_xx,eps_yy,2 eps_xy]
and stress s=[sigma_xx,sigma_yy,sigma_xy]. Also provided: a tensor-shear Jacobian
for [eps_xx,eps_yy,eps_xy]. For scalar-potential dofs, dh=-grad(N)*dphi.
At switches or flat magnetic states there need not be a unique material tangent.
Finite selected-branch tangents remain available with diagnostics (not NaN masks);
the global solver must use damping/line search and respect those diagnostics.
If a null internal mode couples to the input, or negative curvature survives
the active constraints, no valid condensed tangent is claimed.

kinematics='3d' is an auxiliary validation mode for replaying full strain
histories of the homogeneous code. Its 6-vector input uses tensor shear in
order xx,yy,zz,xy,yz,xz; its engineering tangent uses gamma in shear slots.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from itertools import product

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, lsq_linear

MU0 = 4*np.pi*1e-7
SQRT2 = np.sqrt(2.)
FACTORS = np.array([1.,1.,1.,SQRT2,SQRT2,SQRT2])
PLANE = np.array([0,1,3])
G = np.array([[1.,-1.,0.,0.,0.,0.], [1.,1.,0.,0.,0.,0.],
              [-1.,0.,-1.,0.,0.,0.],[-1.,0.,1.,0.,0.,0.]])
G0 = np.array([0.,0.,1.,1.])

# ====================== CSV TEST SETTINGS (Spyder) =======================
CSV_INPUT = "material_test_inputs.csv"  # Relative to this script, or absolute
CSV_HISTORY_MODE = "sequential"  # "sequential" or "reference" (pointwise check)
CSV_COMPUTE_TANGENTS = False      # First compare sigma and b only
CSV_USE_REFERENCE_SEED = False   # Optional diagnostic; NOT imposed solutions
CSV_SHOW_PLOTS = True
# ========================================================================


@dataclass(frozen=True)
class Parameters:
    E: float = 5e9
    nu: float = .3
    msat: float = 514e3
    K: float = .1365e6
    bain_long: float = .0188
    bain_short: float = -.0394
    Y0: float = .03e6
    DY: float = .029e6
    Y_phi: float = 0.  # J/m^3 per radian; explicit reversible default
    evolve_phi: bool = True


@dataclass(frozen=True)
class State:
    xi1: float = .5
    eta1: float = 0.
    eta2: float = 0.
    theta1: float = 0.
    theta3: float = 0.
    phi: float = np.pi/4

    def array(self):
        return np.array([self.xi1,self.eta1,self.eta2,self.theta1,self.theta3,self.phi],float)

    @classmethod
    def from_array(cls,x):
        return cls(*map(float,x))

    @property
    def xi2(self): return 1-self.xi1

    @property
    def alpha(self):
        return .5*np.array([self.xi1+self.eta1,self.xi1-self.eta1,
                           self.xi2+self.eta2,self.xi2-self.eta2])


@dataclass(frozen=True)
class Options:
    maxiter: int = 700
    ftol: float = 1e-13
    energy_scale: float = 1e5
    feasibility_tol: float = 3e-8
    stationarity_tol: float = 3e-5
    active_tol: float = 2e-7
    increment_tol: float = 2e-8
    multistart: bool = True
    curvature_rtol: float = 1e-10
    null_coupling_tol: float = 1e-7


@dataclass
class Tangent:
    C: np.ndarray  # ds/d(engineering strain) [Pa]
    dsigma_dh: np.ndarray  # ds/dh [Pa/(A/m)]
    db_de: np.ndarray  # db/d(engineering strain) [T]
    db_dh: np.ndarray  # [T/(A/m)]
    jacobian_engineering: np.ndarray
    jacobian_tensor_shear: np.ndarray
    valid: bool
    classical_derivative: bool
    diagnostics: dict


@dataclass
class Response:
    sigma: np.ndarray
    b: np.ndarray
    m: np.ndarray
    sigma_full: np.ndarray
    b_full: np.ndarray
    m_full: np.ndarray
    state: State
    a: np.ndarray
    psi: float
    dissipation: float
    incremental_density: float  # psi-mu0*h.m+d, minimized over p
    total_field_density: float  # same plus -.5*mu0*h.h
    tangent: object
    diagnostics: dict

    def split_induction(self,H_external):
        """Return b_tilde for the demagnetization-potential convention of (3)."""
        H_external=np.asarray(H_external,float)
        if H_external.shape != self.b.shape:
            raise ValueError("External field shape must match b")
        return self.b-MU0*H_external


def kelvin_tensor(v):
    v=np.asarray(v)/FACTORS
    return np.array([[v[0],v[3],v[5]],[v[3],v[1],v[4]],[v[5],v[4],v[2]]])


def tensor_kelvin(A):
    return np.array([A[0,0],A[1,1],A[2,2],SQRT2*A[0,1],SQRT2*A[1,2],SQRT2*A[0,2]])


def magnetization(x,ms):
    xi,e1,e2,t1,t3,_=x
    s1,c1,s3,c3=np.sin(t1),np.cos(t1),np.sin(t3),np.cos(t3)
    m=ms*np.array([e1*c1-(1-xi)*s3,xi*s1+e2*c3,0.])
    dm=ms*np.array([[s3,c1,0.,-e1*s1,-(1-xi)*c3,0.],
                    [s1,0.,c3,xi*c1,-e2*s3,0.],[0.,0.,0.,0.,0.,0.]])
    return m,dm


class MSMAMaterial:
    def __init__(self,parameters=None,options=None,kinematics="plane_strain"):
        self.p=parameters or Parameters()
        self.o=options or Options()
        if kinematics not in ("plane_strain","3d"):
            raise ValueError("kinematics must be plane_strain or 3d")
        self.kinematics=kinematics
        p=self.p
        if not all(np.isfinite(v) for v in (p.E,p.nu,p.msat,p.K,p.Y0,p.DY,p.Y_phi,p.bain_long,p.bain_short)):
            raise ValueError("Nonfinite parameters")
        if not (p.E>0 and -1<p.nu<.5 and p.msat>0 and min(p.K,p.Y0,p.DY,p.Y_phi)>=0):
            raise ValueError("Invalid material constants")
        lam=p.E*p.nu/((1+p.nu)*(1-2*p.nu)); mu=p.E/(2*(1+p.nu))
        self.C=2*mu*np.eye(6);self.C[:3,:3]+=lam
        self.e1=np.array([p.bain_short,p.bain_long,p.bain_long,0.,0.,0.])
        self.e2=np.array([p.bain_long,p.bain_short,p.bain_long,0.,0.,0.])
        self.delta=self.e1-self.e2

    def inputs(self,eps,h):
        eps=np.asarray(eps,float);h=np.asarray(h,float)
        if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(h)):
            raise ValueError("Inputs must be finite")
        if self.kinematics=="plane_strain":
            if eps.shape==(3,):
                e=np.zeros(6);e[PLANE]=eps*np.array([1.,1.,SQRT2])
            elif eps.shape==(2,2):
                if not np.allclose(eps,eps.T,rtol=0,atol=1e-12):
                    raise ValueError("Strain tensor must be symmetric")
                A=np.zeros((3,3));A[:2,:2]=eps;e=tensor_kelvin(A)
            elif eps.shape==(3,3):
                if not np.allclose(eps,eps.T,rtol=0,atol=1e-12) or np.max(np.abs(eps[2,:]))>1e-12:
                    raise ValueError("Plane strain requires eps_zz=eps_xz=eps_yz=0")
                e=tensor_kelvin(eps)
            else: raise ValueError("Plane strain eps: [xx,yy,xy] or symmetric 2x2/3x3 tensor")
            if h.shape==(2,): h=np.r_[h,0.]
            if h.shape!=(3,) or abs(h[2])>1e-12:
                raise ValueError("Plane model requires h=[hx,hy], h_z=0")
        else:
            if eps.shape==(6,): e=eps*FACTORS
            elif eps.shape==(3,3) and np.allclose(eps,eps.T,rtol=0,atol=1e-12): e=tensor_kelvin(eps)
            else: raise ValueError("3d eps must be tensor-shear 6-vector or symmetric tensor")
            if h.shape!=(3,):raise ValueError("3d mode expects h[3]")
            if abs(h[2])>1e-12:
                raise ValueError("The retained magnetic microstructure is planar; h_z must be zero")
        return e,h

    def laminate(self,phi):
        n=np.array([np.cos(phi),np.sin(phi),0.]);dn=np.array([-np.sin(phi),np.cos(phi),0.])
        def build(n):
            return np.column_stack([tensor_kelvin(.5*(np.outer(v,n)+np.outer(n,v))) for v in np.eye(3)])
        B,Bp=build(n),build(dn);Bpp=-B
        A=B.T@self.C@B;Ap=Bp.T@self.C@B+B.T@self.C@Bp
        a=np.linalg.solve(A,B.T@self.C@self.delta)
        ap=np.linalg.solve(A,Bp.T@self.C@self.delta-Ap@a)
        rvec=B@a-self.delta
        r=float(rvec@self.C@rvec)
        rp=float(2*rvec@self.C@(Bp@a))
        rpp=float(2*(Bp@a+B@ap)@self.C@(Bp@a)+2*rvec@self.C@(Bpp@a+Bp@ap))
        return r,rp,rpp,a

    def derivatives(self,x,e,h,old,dissipative,branches):
        """Return physical density, gradient and Hessian w.r.t. internal x."""
        xi,eta1,eta2,t1,t3,phi=x;p=self.p
        r,rp,rpp,_=self.laminate(phi)
        v=e-self.e2-xi*self.delta
        m,dm=magnetization(x,p.msat)
        psi=.5*v@self.C@v+.5*xi*(1-xi)*r+p.K*(xi*np.sin(t1)**2+(1-xi)*np.sin(t3)**2)
        f=psi-MU0*h@m
        g=-MU0*dm.T@h
        g[0]+=-self.delta@self.C@v+.5*(1-2*xi)*r+p.K*(np.sin(t1)**2-np.sin(t3)**2)
        g[3]+=p.K*xi*np.sin(2*t1)
        g[4]+=p.K*(1-xi)*np.sin(2*t3)
        g[5]+=.5*xi*(1-xi)*rp
        H=np.zeros((6,6))
        H[0,0]=self.delta@self.C@self.delta-r
        H[0,3]=p.K*np.sin(2*t1)-MU0*p.msat*h[1]*np.cos(t1)
        H[0,4]=-p.K*np.sin(2*t3)-MU0*p.msat*h[0]*np.cos(t3)
        H[0,5]=.5*(1-2*xi)*rp
        H[1,3]=MU0*p.msat*h[0]*np.sin(t1)
        H[2,4]=MU0*p.msat*h[1]*np.sin(t3)
        H[3,3]=2*p.K*xi*np.cos(2*t1)+MU0*p.msat*(h[0]*eta1*np.cos(t1)+h[1]*xi*np.sin(t1))
        H[4,4]=2*p.K*(1-xi)*np.cos(2*t3)+MU0*p.msat*(-h[0]*(1-xi)*np.sin(t3)+h[1]*eta2*np.cos(t3))
        H[5,5]=.5*xi*(1-xi)*rpp
        H=H+np.triu(H,1).T
        d=0.
        if dissipative:
            bxi,bphi=branches
            dx=xi-old[0]
            if bxi==1:
                Y=p.Y0+p.DY*xi;d+=Y*dx;g[0]+=Y+p.DY*dx;H[0,0]+=2*p.DY
            elif bxi==-1:
                Y=p.Y0+p.DY*(1-xi);d-=Y*dx;g[0]+=-Y+p.DY*dx;H[0,0]+=2*p.DY
            if bphi:
                d+=p.Y_phi*bphi*(phi-old[5]);g[5]+=p.Y_phi*bphi
        return float(f+d),g,H,float(psi),float(d)

    def physical_rows(self,x):
        rows=[row for row,c in zip(G,G@x+G0) if c<=self.o.active_tol]
        lo=np.array([0.,-1.,-1.,-np.pi/2,-np.pi/2,-np.inf])
        hi=np.array([1.,1.,1.,np.pi/2,np.pi/2,np.inf])
        for j in range(5):
            if x[j]-lo[j]<=self.o.active_tol:rows.append(np.eye(6)[j])
            if hi[j]-x[j]<=self.o.active_tol:rows.append(-np.eye(6)[j])
        return rows

    def stationarity(self,x,e,h,old,dissipative):
        """Physical KKT residual; handles the two dissipation subdifferentials."""
        bx=0 if abs(x[0]-old[0])<=self.o.increment_tol else int(np.sign(x[0]-old[0]))
        bp=0 if abs(x[5]-old[5])<=self.o.increment_tol else int(np.sign(x[5]-old[5]))
        _,g,_,_,_=self.derivatives(x,e,h,old,dissipative,(bx,bp))
        rows=self.physical_rows(x);cols=[r for r in rows]
        lower=[0.]*len(rows);upper=[np.inf]*len(rows)
        if not self.p.evolve_phi:
            cols.append(np.eye(6)[5]);lower.append(-np.inf);upper.append(np.inf)
        if dissipative and bx==0 and self.p.Y0+self.p.DY>0:
            cols.append(-np.eye(6)[0])
            lower.append(-(self.p.Y0+self.p.DY*(1-x[0]))/self.o.energy_scale)
            upper.append((self.p.Y0+self.p.DY*x[0])/self.o.energy_scale)
        if dissipative and bp==0 and self.p.evolve_phi and self.p.Y_phi>0:
            cols.append(-np.eye(6)[5]);lower.append(-self.p.Y_phi/self.o.energy_scale);upper.append(self.p.Y_phi/self.o.energy_scale)
        if not cols:return np.linalg.norm(g/self.o.energy_scale,np.inf)
        A=np.array(cols).T
        fit=lsq_linear(A,g/self.o.energy_scale,bounds=(lower,upper),tol=1e-12,max_iter=500)
        return float(np.linalg.norm(A@fit.x-g/self.o.energy_scale,np.inf))

    def solve(self,e,h,state_n,seed=None,dissipative=True):
        old=state_n.array();o=self.o;p=self.p
        if not np.all(np.isfinite(old)) or np.min(state_n.alpha)<-o.feasibility_tol or max(abs(old[3]),abs(old[4]))>np.pi/2+o.feasibility_tol:
            raise ValueError("Invalid state_n")
        xbranches=(0,1,-1) if dissipative and p.Y0+p.DY>0 else (None,)
        phibranches=(0,1,-1) if dissipative and p.evolve_phi and p.Y_phi>0 else (None,)
        candidates=[];failures=[]
        for bx,bp in product(xbranches,phibranches):
            lo=np.array([0.,-1.,-1.,-np.pi/2,-np.pi/2,old[5]-np.pi/2])
            hi=np.array([1.,1.,1.,np.pi/2,np.pi/2,old[5]+np.pi/2])
            # Unwrapped phi tracks the nearest equivalent normal; no modulo jumps.
            if bx==0:lo[0]=hi[0]=old[0]
            elif bx==1:lo[0]=old[0]
            elif bx==-1:hi[0]=old[0]
            if not p.evolve_phi or bp==0:lo[5]=hi[5]=old[5]
            elif bp==1:lo[5]=old[5]
            elif bp==-1:hi[5]=old[5]
            starts=[seed.array() if seed is not None else old]
            if o.multistart:
                starts += [old,np.array([.1,0.,0.,.15,-.17,old[5]]),np.array([.9,0.,0.,-.13,.19,old[5]])]
                if p.evolve_phi:starts += [np.array([.5,0.,0.,.11,-.09,old[5]+np.pi/4])]
            seen=[]
            for guess in starts:
                x0=np.clip(np.asarray(guess).copy(),lo,hi)
                x0[1]=np.clip(x0[1],-x0[0],x0[0]);x0[2]=np.clip(x0[2],-(1-x0[0]),1-x0[0])
                if any(np.max(np.abs(x0-z))<1e-12 for z in seen):continue
                seen.append(x0)
                branches=(bx or 0,bp or 0)
                def fg(x):
                    f,g,_,_,_=self.derivatives(x,e,h,old,dissipative,branches)
                    return f/o.energy_scale,g/o.energy_scale
                res=minimize(fg,x0,jac=True,method="SLSQP",bounds=list(zip(lo,hi)),
                             constraints={"type":"ineq","fun":lambda x:G@x+G0,"jac":lambda x:G},
                             options={"ftol":o.ftol,"maxiter":o.maxiter})
                x=res.x
                violation=max(0.,-np.min(G@x+G0),np.max(lo-x),np.max(x-hi))
                kkt=self.stationarity(x,e,h,old,dissipative)
                if res.success and np.isfinite(res.fun) and violation<=o.feasibility_tol and kkt<=o.stationarity_tol:
                    candidates.append((float(res.fun),x.copy(),kkt,branches))
                else:failures.append(f"{res.message}; feasibility={violation:.2e}, KKT={kkt:.2e}")
        if not candidates:raise RuntimeError("No admissible stationary material update: "+" | ".join(failures))
        best=min(candidates,key=lambda c:c[0])
        # Prefer a nearby solution among numerically equal minima. This only
        # resolves optimizer noise/flat modes; it is not artificial dissipation.
        equivalent=[c for c in candidates if c[0]<=best[0]+2e-11]
        anchor=seed.array() if seed is not None else old
        best=min(equivalent,key=lambda c:np.linalg.norm(c[1]-anchor))
        return best,{"kkt_scaled":best[2],"valid_candidates":len(candidates),
                     "failed_starts":len(failures),"global_minimum_guaranteed":False}

    def condensed_tangent(self,x,e,h,old,dissipative):
        o=self.o;p=self.p;notes=[]
        bx=0 if abs(x[0]-old[0])<=o.increment_tol else int(np.sign(x[0]-old[0]))
        bp=0 if abs(x[5]-old[5])<=o.increment_tol else int(np.sign(x[5]-old[5]))
        _,_,H,_,_=self.derivatives(x,e,h,old,dissipative,(bx,bp))
        rows=self.physical_rows(x)
        if dissipative and bx==0 and p.Y0+p.DY>0:rows.append(np.eye(6)[0]);notes.append("xi sticking branch")
        if not p.evolve_phi or (dissipative and bp==0 and p.Y_phi>0):rows.append(np.eye(6)[5])
        Z=null_space(np.array(rows)) if rows else np.eye(6)
        _,dm=magnetization(x,p.msat)
        Hxz=np.zeros((6,9));Hxz[0,:6]=-self.delta@self.C;Hxz[:,6:]=-MU0*dm.T
        Hzz=np.zeros((9,9));Hzz[:6,:6]=self.C;Hzz[6:,6:]=-MU0*np.eye(3)
        M=Z.T@H@Z;R=Z.T@Hxz
        # Dimensionless inputs: typical strain=1, field scale=1e6 A/m.
        # Scale the rank test by energy, not by mixed SI units.
        eig,V=np.linalg.eigh(.5*(M+M.T))
        tol=o.curvature_rtol*max(np.max(np.abs(eig),initial=0.),1.)
        positive=eig>tol;negative=eig < -tol
        null=~positive & ~negative
        input_scales=np.r_[np.ones(6),np.full(3,1e6)]
        nullcoupling=float(np.linalg.norm((V[:,null].T@R)*input_scales)/max(np.linalg.norm(R*input_scales),1.)) if np.any(null) else 0.
        valid=not np.any(negative) and nullcoupling<=o.null_coupling_tol
        if np.any(null):notes.append("flat internal modes; pseudoinverse used on coupled stable subspace")
        if np.any(negative):notes.append("negative reduced curvature")
        if nullcoupling>o.null_coupling_tol:notes.append("null mode couples to input; tangent not defined")
        if valid:
            Minv=(V[:,positive]/eig[positive])@V[:,positive].T
            Hred=Hzz-R.T@Minv@R
            J=Hred.copy();J[6:,:]*=-1
        else:J=np.full((9,9),np.nan)
        # Kelvin -> engineering strain / tensor stress (work conjugate).
        idx=PLANE if self.kinematics=="plane_strain" else np.arange(6)
        nh=2 if self.kinematics=="plane_strain" else 3
        selection=np.r_[idx,6+np.arange(nh)]
        factors=FACTORS[idx]
        output_scale=np.r_[1/factors,np.ones(nh)]
        input_scale=np.r_[1/factors,np.ones(nh)]
        Je=J[np.ix_(selection,selection)]*output_scale[:,None]*input_scale[None,:]
        Jt=Je.copy();Jt[:,:len(idx)]*=np.where(factors==1.,1.,2.)[None,:]
        # At active inequality/dissipation boundaries a selected branch can be
        # differentiated, but the full constitutive map can have another branch.
        classical=valid and not rows and not np.any(null)
        if rows:notes.append("selected active set; use directional check before assuming a classical derivative")
        n=len(idx)
        return Tangent(Je[:n,:n],Je[:n,n:],Je[n:,:n],Je[n:,n:],Je,Jt,valid,classical,
                       {"notes":notes,"reduced_eigenvalues":eig.tolist(),"null_coupling":nullcoupling,
                        "branch":"fixed active constraints and fixed history",
                        "strain_convention":"engineering shear for C/db_de; tensor shear Jacobian separately"})

    def update(self,eps,h,state_n,compute_tangent=True,seed=None):
        """Physical incremental update; dissipation always active."""
        return self._update(eps,h,state_n,compute_tangent,seed,True)

    def equilibrate(self,eps,h,seed=None,compute_tangent=True):
        """Explicit initialization only: no history/dissipation. Not a load step."""
        return self._update(eps,h,seed or State(),compute_tangent,seed,False)

    def _update(self,eps,h,state_n,compute_tangent,seed,dissipative):
        e,h=self.inputs(eps,h)
        best,diagnostics=self.solve(e,h,state_n,seed,dissipative)
        x=best[1];old=state_n.array()
        bx=0 if abs(x[0]-old[0])<=self.o.increment_tol else int(np.sign(x[0]-old[0]))
        bp=0 if abs(x[5]-old[5])<=self.o.increment_tol else int(np.sign(x[5]-old[5]))
        f,_,_,psi,d=self.derivatives(x,e,h,old,dissipative,(bx,bp))
        sigmaK=self.C@(e-self.e2-x[0]*self.delta)
        sigmafull=kelvin_tensor(sigmaK)
        m,_=magnetization(x,self.p.msat);b=MU0*(h+m)
        plane=self.kinematics=="plane_strain"
        tangent=self.condensed_tangent(x,e,h,old,dissipative) if compute_tangent else None
        return Response(sigmafull[:2,:2].copy() if plane else sigmafull.copy(),
                        b[:2].copy() if plane else b.copy(),m[:2].copy() if plane else m.copy(),
                        sigmafull,b,m,State.from_array(x),self.laminate(x[5])[3],psi,d,f,
                        f-.5*MU0*h@h,tangent,diagnostics)

    def check_tangent(self,eps,h,state_n,response=None,delta_eps=1e-7,delta_h=10.):
        """Compare selected analytic branch with one-sided fixed-history updates.

        A disagreement can indicate a switch, degeneracy, solver noise or a
        different nonconvex minimum. No history is committed by this diagnostic.
        """
        base=response or self.update(eps,h,state_n)
        e,h3=self.inputs(eps,h)
        idx=PLANE if self.kinematics=="plane_strain" else np.arange(6)
        nh=2 if self.kinematics=="plane_strain" else 3
        eng=e[idx]*FACTORS[idx];hv=h3[:nh]
        z=np.r_[eng,hv];step=np.r_[np.full(len(idx),delta_eps),np.full(nh,delta_h)]
        def output(r):return np.r_[tensor_kelvin(r.sigma_full)[idx]/FACTORS[idx],r.b_full[:nh]]
        y0=output(base);plus=np.full((len(z),len(z)),np.nan);minus=plus.copy();failures=[]
        for j,ds in enumerate(step):
            dz=np.eye(len(z))[j]*ds
            for sign,target in ((1,plus),(-1,minus)):
                zz=z+sign*dz
                tensorstrain=zz[:len(idx)]/np.where(FACTORS[idx]==1.,1.,2.)
                try:
                    r=self.update(tensorstrain,zz[len(idx):],state_n,False,base.state)
                    target[:,j]=sign*(output(r)-y0)/ds
                except RuntimeError as exc:failures.append(str(exc))
        analytic=base.tangent.jacobian_engineering if base.tangent is not None else None
        return {"analytic":analytic,"plus":plus,"minus":minus,"central":.5*(plus+minus),"failures":failures}


def read_material_csv(filename):
    """Read the generated strain/local-field input CSV; no field conversion.

    eps entries use tensor shear; h entries are already LOCAL TOTAL A/m.
    D columns are provenance only and never enter the material law.
    """
    strain_names=["eps_"+s for s in ("xx","yy","zz","xy","yz","xz")]
    field_names=["h_"+s+"_A_per_m" for s in ("x","y","z")]
    state_names=["xi1","eta1","eta2","theta1_rad","theta3_rad","phi_rad"]
    required=(["time","is_initial_state"]+strain_names+field_names
              +[prefix+s for prefix in ("state_n_","reference_") for s in state_names]
              +["sigma_reference_"+s+"_Pa" for s in ("xx","yy","zz","xy","yz","xz")]
              +["b_reference_"+s+"_T" for s in ("x","y","z")])
    with Path(filename).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        missing=set(required)-set(reader.fieldnames or [])
        if missing:raise ValueError("Missing CSV columns: "+", ".join(sorted(missing)))
        rows=[]
        for line,row in enumerate(reader,start=2):
            try:
                item={k:float(row[k]) for k in required}
                if not np.all(np.isfinite(list(item.values()))):raise ValueError("nonfinite value")
                rows.append(item)
            except (TypeError,ValueError) as exc:
                raise ValueError(f"Invalid numeric data in CSV line {line}") from exc
    if not rows:raise ValueError("CSV is empty")
    if rows[0]["is_initial_state"]!=1 or any(r["is_initial_state"]!=0 for r in rows[1:]):
        raise ValueError("Exactly the first CSV row must have is_initial_state=1")
    if np.any(np.diff([r["time"] for r in rows])<=0):raise ValueError("CSV times must strictly increase")
    return rows


def run_csv(filename,out="material_csv_results",history_mode="sequential",
            compute_tangents=False,use_reference_seed=False,show=False,parameters=None):
    """Replay all CSV rows. Sequential mode carries ONLY computed states.

    The first reference state is the prescribed prepared initial state. In
    reference mode each subsequent row uses its stored reference history,
    which tests individual updates rather than independent history evolution.
    Reference stresses/inductions are used only for errors/plots. An optional
    reference seed changes optimizer starting guesses, not the energy/minimizer.
    Parameters default to the homogeneous code's values, with fixed phi.
    """
    if history_mode not in ("sequential","reference"):
        raise ValueError("history_mode must be sequential or reference")
    rows=read_material_csv(filename)
    parameters=parameters or Parameters(evolve_phi=False)
    material=MSMAMaterial(parameters,kinematics="3d")
    state_keys=("xi1","eta1","eta2","theta1_rad","theta3_rad","phi_rad")
    def state(row,prefix):return State.from_array([row[prefix+k] for k in state_keys])
    old=state(rows[0],"reference_")
    records=[]
    arrays={k:[] for k in ("time","eps","h","state_n","state","sigma","b",
                          "sigma_reference","b_reference","kkt_scaled")}
    if compute_tangents:
        arrays.update({k:[] for k in ("jacobian_engineering","tangent_valid")})
    for i,row in enumerate(rows):
        eps=np.array([row["eps_"+s] for s in ("xx","yy","zz","xy","yz","xz")])
        h=np.array([row["h_"+s+"_A_per_m"] for s in ("x","y","z")])
        previous=state(row,"state_n_") if history_mode=="reference" and i>0 else old
        seed=state(row,"reference_") if use_reference_seed or i==0 else None
        try:
            response=material.update(eps,h,previous,compute_tangent=compute_tangents,seed=seed)
        except RuntimeError as exc:
            raise RuntimeError(f"CSV update failed at row {i}, t={row['time']}: {exc}") from exc
        sigma=tensor_kelvin(response.sigma_full)/FACTORS
        sr=np.array([row["sigma_reference_"+s+"_Pa"] for s in ("xx","yy","zz","xy","yz","xz")])
        br=np.array([row["b_reference_"+s+"_T"] for s in ("x","y","z")])
        values={"time":row["time"],"eps":eps,"h":h,"state_n":previous.array(),
                "state":response.state.array(),"sigma":sigma,"b":response.b_full,
                "sigma_reference":sr,"b_reference":br,"kkt_scaled":response.diagnostics["kkt_scaled"]}
        if compute_tangents:
            values.update(jacobian_engineering=response.tangent.jacobian_engineering,
                          tangent_valid=response.tangent.valid)
        for k,v in values.items():arrays[k].append(v)
        records.append([i,row["time"],*sigma,*sr,*response.b_full,*br,
                        float(np.max(np.abs(sigma-sr))),float(np.max(np.abs(response.b_full-br))),
                        *response.state.array(),response.diagnostics["kkt_scaled"]])
        old=response.state
        if i%max(1,len(rows)//10)==0 or i==len(rows)-1:
            print(f"CSV {i+1}/{len(rows)}: t={row['time']:.6g}, xi1={old.xi1:.8f}",flush=True)
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"csv_response.npz",**arrays)
    with (out/"csv_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["point","time"]
                        +[prefix+s+"_Pa" for prefix in ("sigma_","sigma_reference_") for s in ("xx","yy","zz","xy","yz","xz")]
                        +[prefix+s+"_T" for prefix in ("b_","b_reference_") for s in ("x","y","z")]
                        +["max_sigma_error_Pa","max_b_error_T"]+list(state_keys)+["KKT_scaled"])
        writer.writerows(records)
    # Store conventions/settings so a plotted replay remains reproducible.
    import json
    from dataclasses import asdict
    metadata={"input_csv":str(Path(filename).resolve()),"kinematics":"3d",
              "history_mode":history_mode,"reference_seed":bool(use_reference_seed),
              "parameters":asdict(parameters),"compute_tangents":bool(compute_tangents),
              "field_units":"local total h in A/m, no demagnetization added",
              "strain_order":"xx,yy,zz,xy,yz,xz; tensor shear",
              "initialization":"first CSV reference state is prescribed as prepared initial state",
              "dissipation":"active for every material.update; CSV does not store source stage flags"}
    (out/"csv_run_settings.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Max stress error [Pa]:",np.max(np.abs(arrays["sigma"]-arrays["sigma_reference"])))
    print("Max induction error [T]:",np.max(np.abs(arrays["b"]-arrays["b_reference"])))
    try:import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; CSV and NPZ results saved.");return arrays
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for k in range(2):
        axes[0,k].plot(arrays["time"],arrays["sigma_reference"][:,k]/1e6,"--",label="homogeneous reference")
        axes[0,k].plot(arrays["time"],arrays["sigma"][:,k]/1e6,label="local material")
        axes[0,k].set_ylabel(f"sigma{k+1}{k+1} [MPa]")
        axes[1,k].plot(arrays["time"],arrays["b_reference"][:,k],"--",label="homogeneous reference")
        axes[1,k].plot(arrays["time"],arrays["b"][:,k],label="local material")
        axes[1,k].set_ylabel(f"b{k+1} [T]")
    for ax in axes.flat:ax.set_xlabel("time");ax.grid(True);ax.legend()
    fig.suptitle("CSV replay: "+history_mode)
    fig.savefig(out/"csv_comparison.png",dpi=150)
    if show:plt.show()
    else:plt.close(fig)
    print("CSV comparison outputs:",out.resolve())
    return arrays


def demo(out="material_demo",show=False):
    """Prescribed plane-strain + local-total-field path, no spatial FEM yet."""
    p=Parameters(evolve_phi=True,Y_phi=0.)
    mat=MSMAMaterial(p)
    # A consistent strain seed avoids silently treating zero strain as stress free.
    old=State(xi1=.6,eta1=.6,eta2=.4,theta1=.15,theta3=-.1)
    stages=[(1.,10,[-.01,0.,0.],[0.,2e5]),
            (1.,10,[-.025,.005,.001],[1e5,7e5]),
            (1.,10,[-.01,0.,0.],[0.,0.])]
    eps0=np.array([-.01,0.,0.]);h0=np.zeros(2)
    rows=[];arrays={k:[] for k in ("time","eps","h","sigma","sigma_zz","b","m","state","C","dsigma_dh","db_de","db_dh","tangent_valid","kkt")}
    t=0.
    for duration,steps,epstarget,htarget in stages:
        et,ht=np.asarray(epstarget),np.asarray(htarget)
        for j in range(1,steps+1):
            f=j/steps;eps=eps0+f*(et-eps0);h=h0+f*(ht-h0)
            r=mat.update(eps,h,old);old=r.state
            sv=np.array([r.sigma[0,0],r.sigma[1,1],r.sigma[0,1]])
            values={"time":t+duration*f,"eps":eps,"h":h,"sigma":sv,"sigma_zz":r.sigma_full[2,2],
                    "b":r.b,"m":r.m,"state":r.state.array(),"C":r.tangent.C,
                    "dsigma_dh":r.tangent.dsigma_dh,"db_de":r.tangent.db_de,"db_dh":r.tangent.db_dh,
                    "tangent_valid":r.tangent.valid,"kkt":r.diagnostics["kkt_scaled"]}
            for k,v in values.items():arrays[k].append(v)
            rows.append([values["time"],*eps,*h,*sv,r.sigma_full[2,2],*r.b,*r.m,*r.state.array(),r.tangent.valid])
        eps0,h0=et,ht;t+=duration
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"material_demo.npz",**{k:np.array(v) for k,v in arrays.items()})
    with (out/"material_demo.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["t","eps11","eps22","eps12","h1_A_m","h2_A_m","sigma11_Pa","sigma22_Pa","sigma12_Pa","sigma33_Pa","b1_T","b2_T","m1_A_m","m2_A_m","xi1","eta1","eta2","theta1","theta3","phi","tangent_valid"]);w.writerows(rows)
    try:import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; numerical outputs saved.");return
    time=np.array(arrays["time"]);sig=np.array(arrays["sigma"]);b=np.array(arrays["b"])
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ax[0,0].plot(time,np.array(arrays["eps"]));ax[0,0].set_ylabel("strain (tensor shear)")
    ax[0,1].plot(time,MU0*np.array(arrays["h"]));ax[0,1].set_ylabel("local total mu0*h [T]")
    ax[1,0].plot(time,sig/1e6);ax[1,0].set_ylabel("sigma [MPa]")
    ax[1,1].plot(time,b);ax[1,1].set_ylabel("total induction b [T]")
    for a in ax.flat:a.set_xlabel("time");a.grid(True)
    fig.savefig(out/"material_demo.png",dpi=150)
    if show:plt.show()
    else:plt.close(fig)
    print("Demo outputs:",out.resolve())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out",default=None)
    parser.add_argument("--csv",default=None,help="Path to material_test_inputs.csv")
    parser.add_argument("--demo",action="store_true",help="Run original plane-strain demo instead of CSV")
    parser.add_argument("--history-mode",choices=["sequential","reference"],default=CSV_HISTORY_MODE)
    parser.add_argument("--tangents",action="store_true",default=CSV_COMPUTE_TANGENTS)
    parser.add_argument("--reference-seed",action="store_true",default=CSV_USE_REFERENCE_SEED)
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--no-show",action="store_true")
    args=parser.parse_args()
    show=(CSV_SHOW_PLOTS or args.show) and not args.no_show
    if args.demo:
        if args.csv:parser.error("Choose either --demo or --csv")
        demo(args.out or "material_demo",show)
    else:
        filename=Path(args.csv) if args.csv else Path(CSV_INPUT)
        if not filename.is_absolute():filename=Path(__file__).resolve().parent/filename
        if not filename.is_file():
            raise FileNotFoundError(f"CSV input not found: {filename}. Place material_test_inputs.csv next to this script, set CSV_INPUT, or use --csv PATH / --demo.")
        run_csv(filename,args.out or filename.parent/"material_csv_results",args.history_mode,
                args.tangents,args.reference_seed,show)
