# SPDX-License-Identifier: BSD-3-Clause
"""Q4 FEM prototype for a rectangular MSMA body in a rectangular air box.

Run in Spyder or: python msma_rectangle_fem.py --no-show
Geometry inputs: ASPECT_RATIO=width/height and MESH_DENSITY. All cells in BOTH
regions have identical dx,dy (rectangles, not necessarily squares). Body edges
coincide exactly with cell edges. Height is a fixed reference length (1 mm).

u exists only in the body; phi_tilde exists on the entire grid. Small strain,
plane strain, 2x2 Gauss integration. phi_tilde=0 on all outer box edges.
The applied uniform field enters h=Hext-grad(phi_tilde). The zero boundary
condition is on the DEMAGNETIZATION potential, not the total potential.

USER_STAGES controls symmetric stress amplitudes and optional independent edge
tractions in GLOBAL x,y coordinates. Tension positive, MPa; field input mu0*Hext
in T. Uniform stress amplitudes are converted to t=Sigma*n on all body edges.
Optional edge vectors are ADDED. Consistent edge forces are integrated, not
assigned by node number. Minimal pins remove the three mechanical rigid modes.
Unbalanced user edge loads generate reactions at these pins (reported).

Artificial quadratic regularization of eta1,eta2 is explicit/configurable.
KAPPA_ETA=1000 J/m^3 selects a nearby domain configuration at each physical step.
This changes the incremental model and introduces step-size dependence. Set 0
for the unregularized model; convergence is not guaranteed in flat states.
Local state_n is held fixed throughout Newton/line searches; only a fully
converged step is committed. Failed steps are bisected without committing trials.

TANGENT_MODE='auto' starts with frozen C_el / mu0*I / zero coupling, and switches
to available analytic active-set tangents after 3 iterations or stalled search.
'frozen' uses only frozen tangents; 'consistent' requests analytic tangents.
Convergence always tests the fully updated, coupled balance residuals.

This is a research prototype, not a mesh/box-independent validated simulation.
Output: GiD ASCII .post.msh/.post.res (including strains and body averages),
NPZ checkpoints, CSV history/reactions, settings.json and PNG/PDF path plots.
GiD reads files; no direct program connection required. --postprocess-only
rebuilds these outputs from an existing result directory without solving again.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field, asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, block_diag, diags
from scipy.sparse.linalg import spsolve

from msma_material import MSMAMaterial, Parameters, Options, State, MU0, PLANE
from msma_postprocessing import body_averages, mean_history_values, plot_results

# ============================ USER INPUTS ==============================
ASPECT_RATIO = 9./5.
MESH_DENSITY = "coarse"    # coarse=2, medium=4, fine=8 cells on SHORTER side;
                         # alternatively an integer >=1

@dataclass
class Stage:
    duration: float
    steps: int
    stress_x_mpa: object = None
    stress_y_mpa: object = None
    shear_mpa: object = None
    mu0_h_ext_t: object = None  # [Hx,Hy]*mu0, T
    edge_tractions_mpa: dict = field(default_factory=dict)
    name: str = ""


USER_STAGES = [
    # Zuerst: Spannung von 0 auf -1.4 MPa erhöhen
    Stage(
        duration=1.0,
        steps=10,
        stress_x_mpa=-1.4,
        name="Mechanische Vorbelastung",
    ),
    # Danach: Magnetfeld erhöhen; Spannung bleibt bei -1.4 MPa
    Stage(
        duration=1.0,
        steps=100,
        mu0_h_ext_t=[0.0, 1.0],
        name="Magnetische Belastung",
    ),
]
# Examples (not active):
# Stage(1.,10,stress_x_mpa=-1.4,stress_y_mpa=-.5,mu0_h_ext_t=[.1,.4])
# Stage(1.,10,mu0_h_ext_t=[0.,0.],name="Decrease field; hold stresses")
# Stage(1.,5,edge_tractions_mpa={"right":[-.2,0.],"left":[.2,0.]})
# In the last example the vectors are extra tractions, added to stress_x/y/shear.

KAPPA_ETA = 1000.0          # J/m^3; explicit numerical domain regularization
TANGENT_MODE = "auto"      # auto / frozen / consistent
MATERIAL_PARAMETERS = Parameters(evolve_phi=True,Y_phi=0.)
INITIAL_STATE = State(xi1=.5,eta1=0.,eta2=0.,theta1=0.,theta3=0.,phi=np.pi/4)
SHOW_PLOTS = True

# Fixed geometric defaults: no additional geometry input is necessary.
REFERENCE_HEIGHT_M = 1e-3
OUT_OF_PLANE_DEPTH_M = 1.0  # force and energy per 1 m depth; plane strain
BOX_PADDING = 1.5          # air distance on EACH side / max(width,height)
# The actual box is expanded to the next whole cell in x and y.
# ======================================================================

EDGES=("left","right","bottom","top")
NORMALS={"left":np.array([-1.,0.]),"right":np.array([1.,0.]),
         "bottom":np.array([0.,-1.]),"top":np.array([0.,1.])}
GAUSS=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])/np.sqrt(3.)


@dataclass(frozen=True)
class SolverOptions:
    tangent_mode: str = TANGENT_MODE
    max_iterations: int = 45
    residual_tolerance: float = 3e-6
    max_line_search: int = 11
    max_cutbacks: int = 5
    kappa_eta: float = KAPPA_ETA
    depth_m: float = OUT_OF_PLANE_DEPTH_M


class DomainMaterial(MSMAMaterial):
    """Same solid model with an explicit eta-proximity dissipation potential."""
    def __init__(self,parameters,options,kappa_eta):
        super().__init__(parameters,options,kinematics="3d")
        if not np.isfinite(kappa_eta) or kappa_eta<0:raise ValueError("kappa_eta must be nonnegative")
        self.kappa_eta=float(kappa_eta)

    @lru_cache(maxsize=4096)
    def laminate(self,phi):
        return super().laminate(float(phi))

    def derivatives(self,x,e,h,old,dissipative,branches):
        f,g,H,psi,d=super().derivatives(x,e,h,old,dissipative,branches)
        if dissipative and self.kappa_eta:
            delta=x[1:3]-old[1:3]
            penalty=.5*self.kappa_eta*float(delta@delta)
            f+=penalty;d+=penalty
            g[1:3]+=self.kappa_eta*delta
            H[1:3,1:3]+=self.kappa_eta*np.eye(2)
        return f,g,H,psi,d


def vacuum_material(h,H_external):
    h=np.asarray(h,float);H_external=np.asarray(H_external,float)
    return MU0*h,MU0*(h-H_external),MU0*np.eye(len(h))


@dataclass
class Mesh:
    nodes: np.ndarray
    cells: np.ndarray
    solid: np.ndarray
    body_nodes: np.ndarray
    body_dofs: np.ndarray
    outer_nodes: np.ndarray
    edge_segments: dict
    pins: np.ndarray
    width: float
    height: float
    dx: float
    dy: float
    nx_body: int
    ny_body: int
    nx: int
    ny: int


def make_mesh(aspect_ratio=ASPECT_RATIO,density=MESH_DENSITY,height=REFERENCE_HEIGHT_M,padding=BOX_PADDING):
    if not np.isfinite(aspect_ratio) or aspect_ratio<=0:raise ValueError("width/height must be positive")
    if not np.isfinite(height) or height<=0 or not np.isfinite(padding) or padding<=0:raise ValueError("height and box padding must be positive")
    if isinstance(density,str):
        levels={"coarse":2,"medium":4,"fine":8}
        if density not in levels:raise ValueError("density: coarse / medium / fine, or integer >=1")
        n=levels[density]
    else:
        if isinstance(density,bool) or int(density)!=density or density<1:raise ValueError("density must be an integer >=1")
        n=int(density)
    width=aspect_ratio*height
    if aspect_ratio>=1:nyb=n;nxb=max(1,int(round(n*aspect_ratio)))
    else:nxb=n;nyb=max(1,int(round(n/aspect_ratio)))
    dx,dy=width/nxb,height/nyb
    px=max(1,int(np.ceil(padding*max(width,height)/dx)))
    py=max(1,int(np.ceil(padding*max(width,height)/dy)))
    nx,ny=nxb+2*px,nyb+2*py
    xx=(np.arange(nx+1)-px)*dx-width/2
    yy=(np.arange(ny+1)-py)*dy-height/2
    X,Y=np.meshgrid(xx,yy);nodes=np.column_stack([X.ravel(),Y.ravel()])
    def nid(i,j):return j*(nx+1)+i
    cells=[];solid=[]
    for j in range(ny):
        for i in range(nx):
            cells.append([nid(i,j),nid(i+1,j),nid(i+1,j+1),nid(i,j+1)])
            solid.append(px<=i<px+nxb and py<=j<py+nyb)
    cells=np.array(cells,int);solid=np.array(solid,bool)
    bn=np.unique(cells[solid]);mapping=np.full(len(nodes),-1,int);mapping[bn]=np.arange(len(bn))
    bd=np.full((len(nodes),2),-1,int);bd[bn]=2*mapping[bn,None]+np.arange(2)
    outer=np.array([nid(i,j) for j in range(ny+1) for i in range(nx+1) if i in (0,nx) or j in (0,ny)],int)
    segments={
        "left":[[nid(px,j),nid(px,j+1)] for j in range(py,py+nyb)],
        "right":[[nid(px+nxb,j),nid(px+nxb,j+1)] for j in range(py,py+nyb)],
        "bottom":[[nid(i,py),nid(i+1,py)] for i in range(px,px+nxb)],
        "top":[[nid(i,py+nyb),nid(i+1,py+nyb)] for i in range(px,px+nxb)]}
    bl,br=nid(px,py),nid(px+nxb,py)
    pins=np.array([bd[bl,0],bd[bl,1],bd[br,1]])
    return Mesh(nodes,cells,solid,bn,bd,outer,segments,pins,width,height,dx,dy,nxb,nyb,nx,ny)


def quadrature(dx,dy):
    gradients=[];B=[]
    for xi,eta in GAUSS:
        G=np.array([[-(1-eta),1-eta,1+eta,-(1+eta)],
                    [-(1-xi),-(1+xi),1+xi,1-xi]])/4
        G[0]*=2/dx;G[1]*=2/dy
        b=np.zeros((3,8));b[0,0::2]=G[0];b[1,1::2]=G[1]
        b[2,0::2]=G[1];b[2,1::2]=G[0]
        gradients.append(G);B.append(b)
    return np.array(gradients),np.array(B),dx*dy/4


def load_vector(stage,previous):
    """[sigma_x,sigma_y,tau, extra edge vectors..., beta_external_x,y]."""
    v=np.asarray(previous,float).copy()
    if not np.isfinite(stage.duration) or stage.duration<=0 or isinstance(stage.steps,bool) or int(stage.steps)!=stage.steps or stage.steps<1:
        raise ValueError("Stage duration >0; integer steps >=1")
    for i,value in enumerate((stage.stress_x_mpa,stage.stress_y_mpa,stage.shear_mpa)):
        if value is not None:v[i]=float(value)*1e6
    for edge,traction in stage.edge_tractions_mpa.items():
        if edge not in EDGES:raise ValueError("Unknown edge: "+edge)
        a=np.asarray(traction,float)
        if a.shape!=(2,):raise ValueError("Edge traction needs [tx,ty]")
        v[3+2*EDGES.index(edge):5+2*EDGES.index(edge)]=a*1e6
    if stage.mu0_h_ext_t is not None:
        a=np.asarray(stage.mu0_h_ext_t,float)
        if a.shape!=(2,):raise ValueError("External field needs [mu0*Hx,mu0*Hy]")
        v[-2:]=a
    if not np.all(np.isfinite(v)):raise ValueError("Nonfinite load")
    return v


def nodal_forces(mesh,load,depth=1.):
    F=np.zeros(2*len(mesh.body_nodes))
    Sigma=np.array([[load[0],load[2]],[load[2],load[1]]])
    for j,edge in enumerate(EDGES):
        traction=Sigma@NORMALS[edge]+load[3+2*j:5+2*j]
        for segment in mesh.edge_segments[edge]:
            length=np.linalg.norm(mesh.nodes[segment[1]]-mesh.nodes[segment[0]])
            for node in segment:F[mesh.body_dofs[node]]+=traction*length*depth/2
    return F


def sparse_add(rows,cols,values,dofs1,dofs2,local):
    rows.extend(np.repeat(dofs1,len(dofs2)));cols.extend(np.tile(dofs2,len(dofs1)));values.extend(local.ravel())


class FEM:
    def __init__(self,mesh,parameters=None,options=None,initial_state=INITIAL_STATE):
        self.mesh=mesh;self.options=options or SolverOptions()
        if self.options.tangent_mode not in ("auto","frozen","consistent"):raise ValueError("Invalid tangent mode")
        self.material=DomainMaterial(parameters or MATERIAL_PARAMETERS,
            Options(multistart=True,ftol=1e-14,maxiter=1400),self.options.kappa_eta)
        self.initial_state=initial_state
        self.nu=2*len(mesh.body_nodes);self.nd=self.nu+len(mesh.nodes)
        self.solid_cells=np.flatnonzero(mesh.solid)
        self.G,self.B,self.weight=quadrature(mesh.dx,mesh.dy)
        self.weight*=self.options.depth_m
        self.Cel=self.material.C[np.ix_(PLANE,PLANE)]/np.array([1.,1.,np.sqrt(2)])[:,None]/np.array([1.,1.,np.sqrt(2)])[None,:]
        rows=[];cols=[];values=[]
        for nodes in mesh.cells:
            local=sum(g.T@g*self.weight for g in self.G)
            sparse_add(rows,cols,values,nodes,nodes,local)
        self.L=coo_matrix((values,(rows,cols)),shape=(len(mesh.nodes),len(mesh.nodes))).tocsr()
        rows=[];cols=[];values=[]
        for elem in self.solid_cells:
            ud=mesh.body_dofs[mesh.cells[elem]].ravel()
            local=sum(b.T@self.Cel@b*self.weight for b in self.B)
            sparse_add(rows,cols,values,ud,ud,local)
        self.Ku=coo_matrix((values,(rows,cols)),shape=(self.nu,self.nu)).tocsr()
        self.Kfrozen=block_diag((self.Ku,-MU0*self.L),format="csr")
        fixed=np.r_[mesh.pins,self.nu+mesh.outer_nodes]
        self.free=np.setdiff1d(np.arange(self.nd),fixed)
        self.du_scale=np.r_[np.full(self.nu,.05*mesh.height),np.full(len(mesh.nodes),self.material.p.msat*mesh.height)]

    def initial_guess(self):
        m=self.mesh;p=self.material.p;s=self.initial_state
        bar=self.material.e2+s.xi1*self.material.delta
        ex,ey=bar[0]+p.nu*bar[2],bar[1]+p.nu*bar[2]
        xy=m.nodes[m.body_nodes]
        d=np.zeros(self.nd)
        d[:self.nu:2]=ex*(xy[:,0]+m.width/2)
        d[1:self.nu:2]=ey*(xy[:,1]+m.height/2)
        return d

    def scales(self,force):
        m=self.mesh
        fscale=max(np.max(np.abs(force),initial=0.),self.material.p.E*m.height*self.options.depth_m*1e-3,1.)
        bscale=MU0*self.material.p.msat*max(m.dx,m.dy)*self.options.depth_m
        return np.r_[np.full(self.nu,1/fscale),np.full(len(m.nodes),1/bscale)]

    def evaluate(self,d,load,old_states,seeds=None,consistent=False):
        mesh=self.mesh;phi=d[self.nu:];Hext=load[-2:]/MU0
        R=np.zeros(self.nd);force=nodal_forces(mesh,load,self.options.depth_m)
        R[:self.nu]=-force;R[self.nu:]=-MU0*(self.L@phi)
        states=np.empty_like(old_states)
        fields={k:np.zeros((len(self.solid_cells),4,n)) for k,n in (("sigma",6),("strain",6),("h",3),("b",3),("m",3))}
        rows=[];cols=[];values=[];invalid=0
        for eidx,elem in enumerate(self.solid_cells):
            nodes=mesh.cells[elem];ud=mesh.body_dofs[nodes].ravel();pd=self.nu+nodes
            for q,(G,B) in enumerate(zip(self.G,self.B)):
                eng=B@d[ud];eps=np.array([eng[0],eng[1],0.,eng[2]/2,0.,0.])
                h=np.r_[Hext-G@phi[nodes],0.]
                old=State.from_array(old_states[eidx,q]);seed=State.from_array(seeds[eidx,q]) if seeds is not None else old
                try:r=self.material.update(eps,h,old,compute_tangent=consistent,seed=seed)
                except RuntimeError as exc:raise RuntimeError(f"Material update: element {elem+1}, Gauss point {q+1}: {exc}") from exc
                states[eidx,q]=r.state.array()
                sigma=np.array([r.sigma_full[0,0],r.sigma_full[1,1],r.sigma_full[2,2],r.sigma_full[0,1],r.sigma_full[1,2],r.sigma_full[0,2]])
                for key,value in (("sigma",sigma),("strain",eps),("h",h),("b",r.b_full),("m",r.m_full)):fields[key][eidx,q]=value
                R[ud]+=B.T@sigma[PLANE]*self.weight
                R[pd]+=MU0*G.T@r.m_full[:2]*self.weight
                if consistent:
                    if r.tangent.valid:
                        C=r.tangent.C[np.ix_(PLANE,PLANE)]
                        D=r.tangent.dsigma_dh[PLANE,:2];A=r.tangent.db_de[:2,PLANE];M=r.tangent.db_dh[:2,:2]
                    else:C=self.Cel;D=np.zeros((3,2));A=np.zeros((2,3));M=MU0*np.eye(2);invalid+=1
                    sparse_add(rows,cols,values,ud,ud,B.T@(C-self.Cel)@B*self.weight)
                    sparse_add(rows,cols,values,ud,pd,-B.T@D@G*self.weight)
                    sparse_add(rows,cols,values,pd,ud,G.T@A@B*self.weight)
                    sparse_add(rows,cols,values,pd,pd,-G.T@(M-MU0*np.eye(2))@G*self.weight)
        K=self.Kfrozen
        if consistent:K=K+coo_matrix((values,(rows,cols)),shape=(self.nd,self.nd)).tocsr()
        return R,K,states,fields,invalid

    def solve_step(self,d0,load,old_states):
        d=d0.copy();seeds=old_states.copy();force=nodal_forces(self.mesh,load,self.options.depth_m)
        scale=self.scales(force);free=self.free
        mode=self.options.tangent_mode;use_consistent=mode=="consistent";history=[]
        for iteration in range(self.options.max_iterations+1):
            if mode=="auto" and iteration>=3:use_consistent=True
            R,K,states,fields,invalid=self.evaluate(d,load,old_states,seeds,use_consistent)
            norm=float(np.linalg.norm((R*scale)[free])/np.sqrt(len(free)))
            history.append(norm)
            if norm<self.options.residual_tolerance:
                return d,states,fields,{"iterations":iteration,"residual":norm,"consistent":use_consistent,
                                      "invalid_tangents":invalid,"reactions":R[self.mesh.pins].copy(),"residual_history":history}
            if iteration==self.options.max_iterations:break
            def direction(K):
                A=diags(scale[free])@K[free][:,free]@diags(self.du_scale[free])
                dz=spsolve(A.tocsc(),-(R*scale)[free])
                if not np.all(np.isfinite(dz)):raise RuntimeError("Nonfinite linear solution")
                step=np.zeros(self.nd);step[free]=self.du_scale[free]*dz
                return step
            accepted=False
            for attempt in range(2 if mode=="auto" and not use_consistent else 1):
                step=direction(K)
                for line in range(self.options.max_line_search):
                    alpha=.5**line;trial=d+alpha*step
                    try:
                        Rt,_,st,_,_=self.evaluate(trial,load,old_states,states,False)
                        nt=float(np.linalg.norm((Rt*scale)[free])/np.sqrt(len(free)))
                    except RuntimeError:continue
                    if nt<=norm*(1-1e-4*alpha) or nt<self.options.residual_tolerance:
                        d=trial;seeds=st;accepted=True;break
                if accepted:break
                if mode=="auto" and not use_consistent:
                    R,K,states,fields,invalid=self.evaluate(d,load,old_states,states,True);use_consistent=True
            if not accepted:raise RuntimeError(f"Global line search stalled; scaled residual={norm:.3e}")
        raise RuntimeError(f"Global iterations exhausted; scaled residual={history[-1]:.3e}")


def write_gid_mesh(mesh,path):
    with Path(path).open("w",encoding="utf-8") as f:
        f.write('MESH "Domain" dimension 2 ElemType Quadrilateral Nnode 4\nCoordinates\n')
        for i,(x,y) in enumerate(mesh.nodes,1):f.write(f"{i} {x:.15g} {y:.15g} 0\n")
        f.write("End Coordinates\nElements\n")
        for i,(nodes,solid) in enumerate(zip(mesh.cells,mesh.solid),1):f.write(f"{i} "+" ".join(str(n+1) for n in nodes)+f" {1 if solid else 2}\n")
        f.write("End Elements\n")


class Results:
    def __init__(self,fem,out):
        self.fem=fem;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.records=[]
        write_gid_mesh(fem.mesh,self.out/"rectangle.post.msh")
        self.res=(self.out/"rectangle.post.res").open("w",encoding="utf-8")
        self.res.write('GiD Post Results File 1.0\nGaussPoints "Q4" ElemType Quadrilateral "Domain"\nNumber Of Gauss Points: 4\nNatural Coordinates: Given\n')
        for xy in GAUSS:self.res.write(f"{xy[0]:.15g} {xy[1]:.15g}\n")
        self.res.write("End GaussPoints\n")

    def result(self,name,time,values,gauss=True,element_ids=None,analysis="MSMA"):
        values=np.asarray(values)
        scalar=values.ndim==(2 if gauss else 1)
        location='OnGaussPoints "Q4"' if gauss else 'OnNodes'
        ids=np.arange(1,len(values)+1) if element_ids is None else np.asarray(element_ids)
        if len(ids)!=len(values):raise ValueError("Result IDs and values have different lengths")
        self.res.write(f'Result "{name}" "{analysis}" {time:.15g} {"Scalar" if scalar else "Vector"} {location}\nValues\n')
        for i,value in zip(ids,values):
            if gauss:
                for q,v in enumerate(value):
                    numbers=np.atleast_1d(v)
                    self.res.write((str(i)+" " if q==0 else "")+" ".join(f"{a:.15g}" for a in numbers)+"\n")
            else:self.res.write(str(i)+" "+" ".join(f"{a:.15g}" for a in np.atleast_1d(value))+"\n")
        self.res.write("End Values\n")

    def save(self,time,d,load,states,fields,info,label):
        fem=self.fem;mesh=fem.mesh;Hext=load[-2:]/MU0
        full={k:np.zeros((len(mesh.cells),4,n)) for k,n in (("h",3),("b",3),("m",3),("sigma",6),("strain",6))}
        for elem,nodes in enumerate(mesh.cells):
            for q,G in enumerate(fem.G):
                h=np.r_[Hext-G@d[fem.nu:][nodes],0.]
                b,_,_=vacuum_material(h,np.r_[Hext,0.]);full["h"][elem,q]=h;full["b"][elem,q]=b
        for key in full:full[key][fem.solid_cells]=fields[key]
        means=body_averages(fields)
        displacement=np.zeros((len(mesh.nodes),3));displacement[mesh.body_nodes,:2]=d[:fem.nu].reshape(-1,2)
        self.result("Demagnetization potential",time,d[fem.nu:],False)
        self.result("Displacement (body only)",time,displacement,False)
        for key,name in (("h","Local magnetic field [A/m]"),("b","Induction [T]"),("m","Magnetization [A/m]")):self.result(name,time,full[key])
        for j,name in enumerate(("xx","yy","zz","xy","yz","xz")):
            self.result("Stress "+name+" [Pa]",time,full["sigma"][:,:,j])
            self.result("Strain "+name+" [-]",time,fields["strain"][:,:,j],
                        element_ids=fem.solid_cells+1)
            for key,title,unit in (("strain","strain","-"),("sigma","stress","Pa")):
                averaged=np.full((len(fem.solid_cells),4),means[key][j])
                self.result(f"Body mean {title} {name} [{unit}]",time,averaged,
                            element_ids=fem.solid_cells+1,analysis="Body averages")
        mean_m=np.broadcast_to(means["m"],fields["m"].shape)
        self.result("Body mean magnetization [A/m]",time,mean_m,
                    element_ids=fem.solid_cells+1,analysis="Body averages")
        for j,name in enumerate("xyz"):
            self.result(f"Body mean magnetization {name} [A/m]",time,mean_m[:,:,j],
                        element_ids=fem.solid_cells+1,analysis="Body averages")
        for j,name in enumerate(("xi1","eta1","eta2","theta1","theta3","phi")):
            values=np.zeros((len(mesh.cells),4));values[fem.solid_cells]=states[:,:,j];self.result(name,time,values)
        self.res.flush()
        step=len(self.records)
        np.savez_compressed(self.out/f"step_{step:04d}.npz",time=time,dofs=d,load=load,states=states,
                            nodes=mesh.nodes,cells=mesh.cells,solid=mesh.solid,body_nodes=mesh.body_nodes,
                            displacement=displacement,potential=d[fem.nu:],
                            **full,**{"mean_"+key:value for key,value in means.items()})
        record={"step":step,"time":time,"label":label,"iterations":info["iterations"],"residual":info["residual"],
                "consistent_tangent_used":info["consistent"],"invalid_tangents":info["invalid_tangents"],
                "mean_sigma_xx_Pa":float(np.mean(fields["sigma"][:,:,0])),"mean_sigma_yy_Pa":float(np.mean(fields["sigma"][:,:,1])),
                "mean_bx_T":float(np.mean(fields["b"][:,:,0])),"mean_by_T":float(np.mean(fields["b"][:,:,1])),
                "mean_xi1":float(np.mean(states[:,:,0])),"mean_mx_A_m":float(np.mean(fields["m"][:,:,0])),
                "mean_my_A_m":float(np.mean(fields["m"][:,:,1])),
                "reaction_BL_x_N":float(info["reactions"][0]),"reaction_BL_y_N":float(info["reactions"][1]),
                "reaction_BR_y_N":float(info["reactions"][2])}
        record.update(mean_history_values(means))
        record.update({"mu0_h_ext_x_T":float(load[-2]),"mu0_h_ext_y_T":float(load[-1]),
                       "applied_sigma_xx_Pa":float(load[0]),"applied_sigma_yy_Pa":float(load[1]),
                       "applied_tau_xy_Pa":float(load[2])})
        self.records.append(record)
        with (self.out/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=record);writer.writeheader();writer.writerows(self.records)
        print(f"step {step}: t={time:.6g}, residual={info['residual']:.2e}, iterations={info['iterations']}, mean xi1={record['mean_xi1']:.5f}",flush=True)

    def close(self):self.res.close()


def run(aspect_ratio=ASPECT_RATIO,density=MESH_DENSITY,stages=None,out="rectangle_results",options=None,
        parameters=None,show=False,height=REFERENCE_HEIGHT_M,padding=BOX_PADDING):
    mesh=make_mesh(aspect_ratio,density,height,padding);fem=FEM(mesh,parameters,options)
    stages=USER_STAGES if stages is None else stages
    writer=Results(fem,out)
    settings={"aspect_ratio":aspect_ratio,"density":density,"height_m":height,"width_m":mesh.width,
              "dx_m":mesh.dx,"dy_m":mesh.dy,"padding":padding,"box_bounds_m":[mesh.nodes.min(axis=0).tolist(),mesh.nodes.max(axis=0).tolist()],
              "solver":asdict(fem.options),"material":asdict(fem.material.p),"initial_state":asdict(fem.initial_state),
              "stages":[asdict(s) for s in stages],"box_material_id":2,"solid_material_id":1,
              "potential":"phi_tilde=0 at outer boundary; h=Hext-grad(phi_tilde)",
              "pins":"bottom-left ux=uy=0; bottom-right uy=0; reactions reported",
              "postprocessing_version":2,
              "averaging":"Volume average over body Gauss points only; equal weights for identical affine Q4 cells",
              "model_limitations":"small strain, plane strain, planar magnetization; finite box; explicit eta regularization"}
    (writer.out/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    states=np.tile(fem.initial_state.array(),(len(fem.solid_cells),4,1))
    d=fem.initial_guess();load=np.zeros(13);time=0.
    print(f"Mesh: {mesh.nx_body}x{mesh.ny_body} solid, {mesh.nx}x{mesh.ny} box; {fem.nd} unknowns before constraints",flush=True)
    try:
        d,states,fields,info=fem.solve_step(d,load,states)
        writer.save(0.,d,load,states,fields,info,"Initial equilibrium")
        def advance(d,old,t0,l0,t1,l1,label,level=0):
            try:
                new,trial,fields,info=fem.solve_step(d,l1,old)
            except RuntimeError as exc:
                if level>=fem.options.max_cutbacks:raise RuntimeError(f"Step to t={t1} failed after cutbacks: {exc}") from exc
                tm=.5*(t0+t1);lm=.5*(l0+l1)
                print(f"Cutback at t={t1:.6g}: {exc}",flush=True)
                mid,smid=advance(d,old,t0,l0,tm,lm,label+" [cutback]",level+1)
                return advance(mid,smid,tm,lm,t1,l1,label,level+1)
            writer.save(t1,new,l1,trial,fields,info,label)
            return new,trial
        for stage in stages:
            target=load_vector(stage,load);startload=load.copy();starttime=time
            for i in range(1,int(stage.steps)+1):
                t1=starttime+stage.duration*i/stage.steps;l1=startload+(target-startload)*i/stage.steps
                d,states=advance(d,states,time,load,t1,l1,stage.name)
                time,load=t1,l1
    finally:
        writer.close()
        plot_results(fem,writer,show)
    return fem,writer


def postprocess_saved_results(out="rectangle_results",show=False):
    """Rebuild GiD/CSV/plots from accepted checkpoints, without solving again.

    history.csv is authoritative: stale extra NPZ files are not included.
    The saved strains, fields, states and dofs are preserved numerically.
    """
    out=Path(out)
    settings=json.loads((out/"settings.json").read_text(encoding="utf-8"))
    with (out/"history.csv").open(newline="",encoding="utf-8") as f:
        records=list(csv.DictReader(f))
    if not records:raise ValueError("No converged steps found in history.csv")
    if [int(row["step"]) for row in records]!=list(range(len(records))):
        raise ValueError("history.csv must list consecutive accepted steps starting at zero")
    paths=[out/f"step_{int(r['step']):04d}.npz" for r in records]
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    mesh=make_mesh(settings["aspect_ratio"],settings["density"],
                   settings["height_m"],settings.get("padding",BOX_PADDING))
    with np.load(paths[0],allow_pickle=False) as data:
        if mesh.nodes.shape!=data["nodes"].shape or not np.allclose(mesh.nodes,data["nodes"],rtol=1e-12,atol=1e-15):
            raise ValueError("Saved mesh differs from reconstructed mesh; check saved padding/BOX_PADDING")
        if not np.array_equal(mesh.cells,data["cells"]) or not np.array_equal(mesh.solid,data["solid"]):
            raise ValueError("Saved mesh topology/material assignment differs")
    parameters=Parameters(**settings["material"])
    options=SolverOptions(**settings["solver"])
    fem=FEM(mesh,parameters,options,State(**settings["initial_state"]))
    writer=Results(fem,out)
    try:
        for row,path in zip(records,paths):
            with np.load(path,allow_pickle=False) as data:
                d=data["dofs"].copy();load=data["load"].copy();states=data["states"].copy()
                time=float(data["time"])
                if not np.isclose(time,float(row["time"]),rtol=1e-12,atol=1e-14):
                    raise ValueError(f"Checkpoint time differs from history: {path.name}")
                fields={key:data[key][mesh.solid].copy() for key in ("strain","sigma","m","h","b")}
            info={"iterations":int(row["iterations"]),"residual":float(row["residual"]),
                  "consistent":str(row["consistent_tangent_used"]).lower()=="true",
                  "invalid_tangents":int(row["invalid_tangents"]),
                  "reactions":np.array([float(row[key]) for key in
                    ("reaction_BL_x_N","reaction_BL_y_N","reaction_BR_y_N")])}
            writer.save(time,d,load,states,fields,info,row["label"])
    finally:writer.close()
    settings["postprocessing_version"]=2
    settings["averaging"]="Volume average over body Gauss points only; equal weights for identical affine Q4 cells"
    (out/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    plot_results(fem,writer,show)
    return fem,writer


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aspect",type=float,default=ASPECT_RATIO)
    parser.add_argument("--density",default=MESH_DENSITY)
    parser.add_argument("--out",default="rectangle_results")
    parser.add_argument("--tangent",choices=["auto","frozen","consistent"],default=TANGENT_MODE)
    parser.add_argument("--kappa-eta",type=float,default=KAPPA_ETA)
    parser.add_argument("--no-show",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="Small 2x1 body; one weak field increment")
    parser.add_argument("--postprocess-only",action="store_true",help="Rebuild outputs from NPZ/history/settings in --out; no FEM solve")
    args=parser.parse_args()
    density=int(args.density) if str(args.density).isdigit() else args.density
    stages=[Stage(1.,1,stress_x_mpa=-.1,mu0_h_ext_t=[0.,.002],name="Smoke test")] if args.smoke else USER_STAGES
    if args.postprocess_only:
        postprocess_saved_results(args.out,show=SHOW_PLOTS and not args.no_show)
    else:
        run(args.aspect,1 if args.smoke else density,stages,args.out,
            SolverOptions(tangent_mode=args.tangent,kappa_eta=args.kappa_eta),show=SHOW_PLOTS and not args.no_show)
