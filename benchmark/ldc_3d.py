#!/usr/bin/env python
"""3D lid-driven cavity benchmark."""

import itertools
import numpy as np
import csv
import os.path
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
#from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.geo import LBGeometry3D


class LDCBlock(Subdomain3D):
    """3D Lid-driven geometry."""

    max_v = 0.05

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        velocity_bc = NTRegularizedVelocity

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

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({'lat_nx': 64, 'lat_ny': 64, 'lat_nz': 64, 'grid': 'D3Q19'})


def run_benchmark(lattice_size: tuple, block_size: int = 128,
                  timesteps: int = 1000, boundary_split: bool = False,
                  periodic: tuple = (False, False, True)):
    settings = {
        'verbose': True,
        'mode': 'benchmark',
        'access_pattern': 'AB',
        'model': 'bgk',
        'grid': 'D3Q19',
        'init_iters': 10,
        'max_iters': timesteps,
        'block_size': block_size,
        'lat_nx': lattice_size[0],
        'lat_ny': lattice_size[1],
        'lat_nz': lattice_size[2],
        'periodic_x': periodic[0],
        'periodic_y': periodic[1],
        'periodic_z': periodic[2],
        'use_intrinsics': True,
        'precision': 'double',
        'node_addressing': 'direct',
        'incompressible': True,
	'every': 500,
    }

    ctrl = LBSimulationController(LDCSim, LBGeometry3D, settings)
    timing_infos, min_timings, max_timings, subdomains = ctrl.run()

    mlups_comp = float(0)
    mlups_total = float(0)
    for ti in timing_infos:
        block = subdomains[ti.subdomain_id]
        mlups_comp += (block.num_nodes / ti.comp) * 1e-6
        mlups_total += (block.num_nodes / ti.total) * 1e-6

    data = {
        "cubicBlockSize": lattice_size[0],
        "timesteps": timesteps,
        "cudaBlockSize": block_size,
        "fullyPeriodic": all(periodic) is True,
        "mlupsPerProcess": mlups_comp,
	"boundarySplit": boundary_split,
    }

    csv_exists = False
    if os.path.isfile("singleNode.csv") is True:
        csv_exists = True

    with open("singleNode.csv", "a") as csvfile:
        writer = csv.writer(csvfile)
        if csv_exists is False:
            # Write header
            writer.writerow(list(sorted(data.keys())))

        writer.writerow([data[key] for key in sorted(data.keys())])

    print(f"MLUPS (compute): {mlups_comp}")


if __name__ == '__main__':
    cudaBlockSizes = [32, 64, 128, 256]
    cubicLatticeSizes = [3 * (lat_siz,) for lat_siz in reversed(range(16, 256+1, 16))]
    fullyPeriodic = [False, True]
    boundarySplit = [False, True]
    generator = itertools.product(cubicLatticeSizes, cudaBlockSizes, fullyPeriodic, boundarySplit)
    for params in generator:
        cubicLatticeSize, cudaBlockSize, isFullyPeriodic, isBoundarySplit = params
        run_benchmark(cubicLatticeSize, block_size=cudaBlockSize,
                      boundary_split=isBoundarySplit,
                      periodic=(True, True, True) if isFullyPeriodic else (False, False, True))
