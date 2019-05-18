#!/usr/bin/env python
"""3D lid-driven cavity benchmark."""

import itertools
import numpy as np
import csv
import os.path
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.geo import EqualSubdomainsGeometry3D


class LDCBlock(Subdomain3D):
    """3D Lid-driven geometry."""

    max_v = 0.043

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        velocity_bc = NTEquilibriumVelocity

        wall_map = ((hz == 0) | (hx == self.gx - 1) | (hx == 0) | (hy == 0) |
                (hy == self.gy - 1))
        self.set_node(wall_map, wall_bc)
        self.set_node((hz == self.gz - 1) & np.logical_not(wall_map),
                velocity_bc((self.max_v, 0.0, 0.0)))

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vx[hz == self.gz - 1] = self.max_v


class LDCSim(LBFluidSim):
    subdomain = LDCBlock


def save_result(filename_base, num_blocks, timing_infos, min_timings,
        max_timings, subdomains):
    f = open('%s_%d' % (filename_base, num_blocks), 'w')
    f.write(str(timing_infos))
    f.close()

    f = open('%s_min_%d' % (filename_base, num_blocks), 'w')
    f.write(str(min_timings))
    f.close()

    f = open('%s_max_%d' % (filename_base, num_blocks), 'w')
    f.write(str(max_timings))
    f.close()

    mlups_total = 0
    mlups_comp = 0

    for ti in timing_infos:
        block = subdomains[ti.subdomain_id]
        mlups_total += block.num_nodes / ti.total * 1e-6
        mlups_comp  += block.num_nodes / ti.comp * 1e-6

    f = open('%s_mlups_%d' % (filename_base, num_blocks), 'w')
    f.write('%.2f %.2f\n' % (mlups_total, mlups_comp))
    f.close()


def run_benchmark(lattice_size: tuple, block_size: int = 128,
                  timesteps: int = 1000, boundary_split: bool = False,
                  check_results: bool = True, disable_cache: bool = True,
                  periodic: tuple = (False, False, True)):
    settings = {
        'verbose': True,
        'mode': 'benchmark',
        'access_pattern': 'AB',
        'force_implementation': 'bgk',
        'init_iters': 10,
        'max_iters': timesteps,
        'block_size': block_size,
        'subdomains': 1,
        'lat_nx': lattice_size[0],
        'lat_ny': lattice_size[1],
        'lat_nz': lattice_size[2],
        'periodic_x': periodic[0],
        'periodic_y': periodic[1],
        'periodic_z': periodic[2],
        'use_intrinsics': True,
        'precision': 'double',
        'node_addresing': 'direct',
        'nocheck_invalid_results_host': check_results,
        'nocheck_invalid_results_gpu': check_results,
        'cuda-disable-l1': disable_cache,
        'cuda-kernel-stats': True,
        'nocuda_cache': True,
        'cuda-nvcc-opts': '-O3',
        'incompressible': True,
    }

    ctrl = LBSimulationController(LDCSim, EqualSubdomainsGeometry3D, settings)
    timing_infos, min_timings, max_timings, _ = ctrl.run()

    print(str(timing_infos))
    print(min_timings)
    print(max_timings)


if __name__ == '__main__':
    cudaBlockSizes = [32, 64, 128, 256]
    cubicLatticeSizes = [3 * (lat_siz,) for lat_siz in reversed(range(16, 320+1, 32))]
    fullyPeriodic = [False, True]
    boundarySplit = [False, True]
    checkResults = [False, True]
    generator = itertools.product(cubicLatticeSizes, cudaBlockSizes, fullyPeriodic, boundarySplit, checkResults)
    for params in generator:
        cubicLatticeSize, cudaBlockSize, isFullyPeriodic, isBoundarySplit, hasToCheckResults = params
        run_benchmark(cubicLatticeSize, block_size=cudaBlockSize,
                      boundary_split=isBoundarySplit,
                      check_results=hasToCheckResults,
                      periodic=(True, True, True) if isFullyPeriodic else (False, False, True))
