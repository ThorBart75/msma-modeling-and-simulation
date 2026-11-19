"""Run: python -m unittest -v test_rectangle_fem.py"""
import tempfile
import unittest
from dataclasses import replace, asdict
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from msma_rectangle_fem import (make_mesh, quadrature, nodal_forces, load_vector,
    Stage, FEM, SolverOptions, MATERIAL_PARAMETERS, INITIAL_STATE, Results, MU0,
    postprocess_saved_results)
from msma_postprocessing import body_averages, varying_segments


class RectangleTests(unittest.TestCase):
    def test_uniform_grid_and_exact_aspect(self):
        for aspect in (.3,1.,2.,3.7):
            m=make_mesh(aspect,2)
            corners=m.nodes[m.cells]
            assert_allclose(corners[:,1]-corners[:,0],np.tile([m.dx,0.],(len(m.cells),1)),atol=1e-17)
            assert_allclose(corners[:,3]-corners[:,0],np.tile([0.,m.dy],(len(m.cells),1)),atol=1e-17)
            self.assertAlmostEqual(m.width/m.height,aspect)
            self.assertEqual(m.solid.sum(),m.nx_body*m.ny_body)

    def test_balanced_stress_force_and_moment(self):
        m=make_mesh(2.,2)
        l=load_vector(Stage(1.,2,stress_x_mpa=-1.,stress_y_mpa=.5,shear_mpa=.2),np.zeros(13))
        f=nodal_forces(m,l).reshape(-1,2);x=m.nodes[m.body_nodes]
        assert_allclose(f.sum(axis=0),0.,atol=1e-10)
        self.assertAlmostEqual(np.sum(x[:,0]*f[:,1]-x[:,1]*f[:,0]),0.,places=10)

    def test_single_edge_consistent_forces(self):
        m=make_mesh(2.,2);l=np.zeros(13);l[5:7]=[2e6,-1e6]
        f=nodal_forces(m,l,depth=.7).reshape(-1,2)
        assert_allclose(f.sum(axis=0),np.array([2e6,-1e6])*m.height*.7)
        ids=np.array(m.edge_segments['right']);end=ids[0,0];mid=ids[0,1]
        assert_allclose(f[m.body_dofs[mid,0]//2],2*f[m.body_dofs[end,0]//2])

    def test_stage_hold_and_simultaneous_targets(self):
        a=load_vector(Stage(1.,3,stress_x_mpa=-1.,mu0_h_ext_t=[0.,.1]),np.zeros(13))
        b=load_vector(Stage(.5,5,stress_y_mpa=.4,mu0_h_ext_t=[0.,0.]),a)
        self.assertEqual(b[0],-1e6);self.assertEqual(b[1],.4e6)
        assert_allclose(b[-2:],0.)

    def test_elastic_and_vacuum_patch(self):
        p=replace(MATERIAL_PARAMETERS,bain_long=0.,bain_short=0.,msat=1e-9,K=0.,evolve_phi=False)
        fem=FEM(make_mesh(1.7,2),p,SolverOptions(tangent_mode='frozen',kappa_eta=0.))
        old=np.tile(INITIAL_STATE.array(),(len(fem.solid_cells),4,1));copy=old.copy()
        load=load_vector(Stage(1.,1,stress_x_mpa=-.2,stress_y_mpa=.1,shear_mpa=.03,mu0_h_ext_t=[.01,.02]),np.zeros(13))
        d,s,f,info=fem.solve_step(fem.initial_guess(),load,old)
        assert_allclose(old,copy,rtol=0.,atol=0.)
        for col,target in ((0,-.2e6),(1,.1e6),(3,.03e6)):
            assert_allclose(f['sigma'][:,:,col],target,rtol=1e-8,atol=.01)
        assert_allclose(f['strain'][:,:,[2,4,5]],0.,atol=0.)
        assert_allclose(f['b'][:,:,:2],np.broadcast_to([.01,.02],f['b'][:,:,:2].shape),atol=1e-12)
        assert_allclose(d[fem.nu+fem.mesh.outer_nodes],0.,atol=0.)
        self.assertLessEqual(info['iterations'],1)
        with tempfile.TemporaryDirectory() as folder:
            writer=Results(fem,folder);writer.save(1.,d,load,s,f,info,'Patch test');writer.close()
            text=(Path(folder)/'rectangle.post.res').read_text()
            self.assertIn('OnGaussPoints "Q4"',text)
            self.assertEqual(text.count('\nValues\n'),text.count('End Values'))
            data=np.load(Path(folder)/'step_0000.npz')
            assert_allclose(data['states'],s)
            assert_allclose(data['mean_sigma'][[0,1,3]],[-.2e6,.1e6,.03e6],atol=.01)
            assert_allclose(data['mean_strain'],np.mean(f['strain'],axis=(0,1)))
            self.assertIn('Result "Strain xx [-]" "MSMA"',text)
            self.assertIn('Result "Strain yy [-]" "MSMA"',text)
            marker='Result "Body mean strain xx [-]" "Body averages"'
            self.assertIn(marker,text)
            block=text.split(marker,1)[1].split('Values\n',1)[1].split('End Values',1)[0].splitlines()
            self.assertEqual(len(block),4*len(fem.solid_cells))
            self.assertEqual([int(line.split()[0]) for line in block[::4]],list(fem.solid_cells+1))
            for line in block:self.assertAlmostEqual(float(line.split()[-1]),data['mean_strain'][0],places=12)
            with (Path(folder)/'history.csv').open() as stream:
                header=stream.readline()
            for key in ('mean_eps_xx','mean_eps_yy','mean_sigma_zz_Pa','mean_mz_A_m'):
                self.assertIn(key,header)

    def test_nonuniform_body_averages(self):
        fields={key:np.arange(2*4*n,dtype=float).reshape(2,4,n)
                for key,n in (('strain',6),('sigma',6),('m',3),('h',3),('b',3))}
        means=body_averages(fields)
        assert_allclose(means['strain'],[21,22,23,24,25,26])
        assert_allclose(means['m'],[10.5,11.5,12.5])

    def test_hysteresis_segments_keep_reversals_and_omit_holds(self):
        self.assertEqual(varying_segments([0,0,1,2,1,0,0,3]),[(1,5),(6,7)])
        self.assertEqual(varying_segments([0,0,0]),[])
        self.assertEqual(varying_segments([1,1+1e-12,1]),[])

    def test_reprocess_without_material_updates_preserves_fields(self):
        fem=FEM(make_mesh(2.,1))
        old=np.tile(INITIAL_STATE.array(),(len(fem.solid_cells),4,1));load=np.zeros(13)
        d,s,fields,info=fem.solve_step(fem.initial_guess(),load,old)
        with tempfile.TemporaryDirectory() as folder:
            writer=Results(fem,folder);writer.save(0.,d,load,s,fields,info,'Initial');writer.close()
            settings={'aspect_ratio':2.,'density':1,'height_m':fem.mesh.height,
                      'material':asdict(fem.material.p),'solver':asdict(fem.options),
                      'initial_state':asdict(fem.initial_state)}
            (Path(folder)/'settings.json').write_text(json.dumps(settings))
            path=Path(folder)/'step_0000.npz'
            with np.load(path) as data:before={key:data[key].copy() for key in data.files}
            with patch('msma_rectangle_fem.DomainMaterial.update',side_effect=AssertionError('Unexpected material solve')):
                with patch('msma_rectangle_fem.plot_results'):
                    postprocess_saved_results(folder)
            with np.load(path) as after:
                for key,value in before.items():assert_allclose(after[key],value,rtol=0.,atol=0.)

    def test_unloaded_plane_strain_equilibrium(self):
        fem=FEM(make_mesh(2.,1));old=np.tile(INITIAL_STATE.array(),(len(fem.solid_cells),4,1))
        d,s,f,info=fem.solve_step(fem.initial_guess(),np.zeros(13),old)
        assert_allclose(f['sigma'][:,:,:2],0.,atol=1e-5)
        assert_allclose(f['sigma'][:,:,2],-fem.material.p.E*fem.material.p.bain_long,rtol=1e-10)
        assert_allclose(f['b'],0.,atol=1e-12)

    def test_magnetic_residual_jacobian_and_sign(self):
        # Prescribed constant magnetization isolates global field assembly from the optimizer.
        from types import SimpleNamespace
        fem=FEM(make_mesh(2.,1))
        class ConstantMaterial:
            def update(self,eps,h,old,**kwargs):
                magnetization=np.array([10000.,30000.,0.])
                return SimpleNamespace(state=old,sigma_full=np.zeros((3,3)),m_full=magnetization,
                                       b_full=MU0*(h+magnetization))
        fem.material=ConstantMaterial();old=np.tile(INITIAL_STATE.array(),(len(fem.solid_cells),4,1))
        d=np.zeros(fem.nd);load=np.zeros(13)
        R,K,_,_,_=fem.evaluate(d,load,old)
        rng=np.random.default_rng(9);v=np.zeros(fem.nd);v[fem.nu:]=rng.normal(size=len(fem.mesh.nodes))
        Rp,*_=fem.evaluate(d+v,load,old)
        assert_allclose((Rp-R)[fem.nu:],(K@v)[fem.nu:],atol=1e-16)
        from scipy.sparse.linalg import spsolve
        free=np.setdiff1d(np.arange(len(fem.mesh.nodes)),fem.mesh.outer_nodes)
        phi=np.zeros(len(fem.mesh.nodes));phi[free]=spsolve(fem.L[free][:,free],R[fem.nu:][free]/MU0)
        demag=np.array([-g@phi[fem.mesh.cells[e]] for e in fem.solid_cells for g in fem.G])
        self.assertLess(demag[:,0].mean(),0.);self.assertLess(demag[:,1].mean(),0.)
        d[fem.nu:]=phi;Rf,*_=fem.evaluate(d,load,old)
        assert_allclose(Rf[fem.nu:][free],0.,atol=1e-16)


if __name__=='__main__':unittest.main()
