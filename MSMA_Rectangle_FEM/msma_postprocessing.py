# SPDX-License-Identifier: BSD-3-Clause
"""Body averages and path plots; no material updates or changes to solver physics."""
from __future__ import annotations
import warnings
import numpy as np

TENSOR_COMPONENTS=("xx","yy","zz","xy","yz","xz")


def body_averages(fields):
    """Volume averages: identical affine Q4 cells, identical 2x2 Gauss weights.

    fields contains BODY Gauss points only. Air is deliberately excluded.
    Tensor shear epsilon_xy is averaged, not engineering shear gamma_xy.
    """
    return {key:np.mean(np.asarray(fields[key]),axis=(0,1))
            for key in ("strain","sigma","m","h","b")}


def mean_history_values(means):
    result={}
    for j,c in enumerate(TENSOR_COMPONENTS):
        result[f"mean_eps_{c}"]=float(means["strain"][j])
        result[f"mean_sigma_{c}_Pa"]=float(means["sigma"][j])
    for j,c in enumerate("xyz"):
        for key,unit in (("m","A_m"),("h","A_m"),("b","T")):
            result[f"mean_{key}{c}_{unit}"]=float(means[key][j])
    return result


def varying_segments(x,absolute_tolerance=1e-12,relative_tolerance=1e-8):
    """Continuous changing parts in TIME order, including reversals.

    Returns inclusive point-index pairs. Flat preload/hold portions are omitted.
    Never sort by the abscissa: sorting would destroy hysteresis loops.
    """
    x=np.asarray(x,float)
    if len(x)<2:return []
    tolerance=max(absolute_tolerance,relative_tolerance*np.max(np.abs(x)))
    changing=np.abs(np.diff(x))>tolerance
    intervals=[];start=None
    for i,value in enumerate(changing):
        if value and start is None:start=i
        if start is not None and (not value or i==len(changing)-1):
            end=i if not value else i+1
            intervals.append((start,end));start=None
    return intervals


def plot_results(fem,writer,show=False):
    try:import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("Matplotlib unavailable: numerical/GiD files saved; plots skipped.")
        return
    from matplotlib.collections import PolyCollection
    figures=[]
    def finish(fig,name):
        figures.append(fig)
        fig.savefig(writer.out/(name+".png"),dpi=180)
        fig.savefig(writer.out/(name+".pdf"))
    def decorate(ax,xlabel,ylabel):
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(True,alpha=.3)
        if ax.get_legend_handles_labels()[0]:ax.legend()
    def minimum_y_range(ax,span):
        # Do not magnify roundoff into an apparent physical response.
        # Raw numerical values remain untouched in the plots, NPZ and CSV.
        lo,hi=ax.get_ylim()
        if hi-lo<span:
            center=.5*(lo+hi);ax.set_ylim(center-.5*span,center+.5*span)
    def path(ax,x,y,label,color):
        ax.plot(x,y,".-",label=label,color=color,markersize=5)
        if len(x)>1:
            # Arrow on an actual segment, preserving the chronological direction.
            k=(len(x)-2)//2
            ax.annotate("",xy=(x[k+1],y[k+1]),xytext=(x[k],y[k]),
                        arrowprops={"arrowstyle":"->","color":color,"lw":1.3})

    m=fem.mesh
    fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
    colors=np.where(m.solid[:,None],np.array([[.95,.65,.25,1.]]),np.array([[.8,.9,1.,1.]]))
    ax.add_collection(PolyCollection(m.nodes[m.cells]*1e3,facecolors=colors,edgecolors="gray",linewidths=.3))
    ax.autoscale();ax.set_aspect("equal");ax.set_xlabel("x [mm]");ax.set_ylabel("y [mm]")
    ax.set_title(f"Identical Q4 cells: {m.nx_body} x {m.ny_body} body, {m.nx} x {m.ny} box")
    finish(fig,"mesh")

    if writer.records:
        records=writer.records
        def col(key):return np.array([r[key] for r in records],float)
        t=col("time");msat=fem.material.p.msat
        eps=np.column_stack([col("mean_eps_"+c) for c in ("xx","yy")])
        stress=np.column_stack([col("mean_sigma_"+c+"_Pa")/1e6 for c in ("xx","yy")])
        mag=np.column_stack([col("mean_m"+c+"_A_m")/msat for c in "xy"])
        beta=np.column_stack([col("mu0_h_ext_"+c+"_T") for c in "xy"])
        fig,axes=plt.subplots(3,2,figsize=(11,10),layout="constrained")
        for j,c in enumerate(("xx","yy")):
            axes[0,0].plot(t,100*eps[:,j],".-",label=c)
            axes[0,1].plot(t,stress[:,j],".-",label=c)
        for j,c in enumerate("xy"):
            axes[1,0].plot(t,mag[:,j],".-",label=c)
            axes[1,1].plot(t,col("mean_b"+c+"_T"),".-",label=c)
        axes[2,0].plot(t,col("mean_xi1"),".-",label="xi1")
        axes[2,1].semilogy(t,np.maximum(col("residual"),1e-16),".-",label="scaled residual")
        for ax,ylabel in zip(axes.flat,("Mean strain [%]","Mean stress [MPa]",
                "Mean magnetization / Ms [-]","Mean induction [T]","Mean variant fraction [-]","Balance residual [-]")):
            decorate(ax,"Path time",ylabel)
        finish(fig,"history")

        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        for j,c in enumerate(("xx","yy")):
            axes[0].plot(t,col("applied_sigma_"+c+"_Pa")/1e6,".-",label=c)
        axes[0].plot(t,col("applied_tau_xy_Pa")/1e6,".-",label="xy")
        for j,c in enumerate("xy"):axes[1].plot(t,beta[:,j],".-",label=c)
        decorate(axes[0],"Path time","Prescribed stress amplitude [MPa]")
        decorate(axes[1],"Path time",r"$\mu_0 H_{ext}$ [T]")
        finish(fig,"loading_history")

        # Figs. 12/14: strain change and magnetization during field ramps.
        field_segments={j:varying_segments(beta[:,j]) for j in range(2)}
        active=[j for j in range(2) if field_segments[j]]
        fig,axes=plt.subplots(max(1,len(active)),2,figsize=(11,4*max(1,len(active))),
                              layout="constrained",squeeze=False)
        if not active:
            for ax in axes.flat:ax.text(.5,.5,"No varying external field in this path",ha="center",transform=ax.transAxes)
        for row,j in enumerate(active):
            for k,(a,b) in enumerate(field_segments[j]):
                idx=slice(a,b+1);x=beta[idx,j]
                for c,color in enumerate(("C0","C1")):
                    # Reference is the first point of EACH continuous field ramp.
                    label=("xx","yy")[c] if k==0 else None
                    path(axes[row,0],x,100*(eps[idx,c]-eps[a,c]),label,color)
                    path(axes[row,1],x,mag[idx,c],("x","y")[c] if k==0 else None,color)
            xlabel=rf"$\mu_0 H_{{ext,{('x','y')[j]}}}$ [T]"
            decorate(axes[row,0],xlabel,"Mean strain change from ramp start [%]")
            decorate(axes[row,1],xlabel,"Mean magnetization / Ms [-]")
            minimum_y_range(axes[row,0],.01);minimum_y_range(axes[row,1],.02)
        fig.suptitle("Field-ramp response: chronological path; both response directions")
        finish(fig,"field_induced_response")

        # Fig. 13: strain and magnetization against the body-mean stress.
        stress_segments={j:varying_segments(stress[:,j],absolute_tolerance=1e-6) for j in range(2)}
        active=[j for j in range(2) if stress_segments[j]]
        fig,axes=plt.subplots(max(1,len(active)),2,figsize=(11,4*max(1,len(active))),
                              layout="constrained",squeeze=False)
        if not active:
            for ax in axes.flat:ax.text(.5,.5,"No varying mean stress in this path",ha="center",transform=ax.transAxes)
        for row,j in enumerate(active):
            for k,(a,b) in enumerate(stress_segments[j]):
                idx=slice(a,b+1);x=stress[idx,j]
                for c,color in enumerate(("C0","C1")):
                    path(axes[row,0],x,100*eps[idx,c],("xx","yy")[c] if k==0 else None,color)
                    path(axes[row,1],x,mag[idx,c],("x","y")[c] if k==0 else None,color)
            xlabel="Mean stress "+("xx","yy")[j]+" [MPa]"
            decorate(axes[row,0],xlabel,"Mean strain [%]")
            decorate(axes[row,1],xlabel,"Mean magnetization / Ms [-]")
            minimum_y_range(axes[row,0],.01);minimum_y_range(axes[row,1],.02)
        fig.suptitle("Stress-ramp response: chronological path")
        finish(fig,"stress_induced_response")

        fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
        for j,color in enumerate(("C0","C1")):
            path(ax,100*eps[:,j],stress[:,j],("xx","yy")[j],color)
        decorate(ax,"Mean strain [%]","Mean stress [MPa]")
        finish(fig,"stress_strain")
    if show:plt.show()
    else:
        for fig in figures:plt.close(fig)
