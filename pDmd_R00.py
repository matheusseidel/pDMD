# ----------------------------------------------------------- #
#                            DMD                              #
# ----------------------------------------------------------- #

# Author:               Matheus Seidel (matheus.seidel@coc.ufrj.br)
# Revision:             00
# Last update:          12/04/2023

# Description:
'''  
This code performs a preliminary test for piecewise Dynamic Mode Decomposition (pDMD) on fluid simulation data. 
It uses the PyDMD library to  to generate the DMD modes
and reconstruct the approximation of the original simulation.
The mesh is read by meshio library using vtk files. The simulation data is in h5 format and is read using h5py.
Details about DMD can be found in:
Schmid, P. J., "Dynamic Mode Decomposition of Numerical and Experimental Data". JFM, Vol. 656, Aug. 2010,pp. 5–28. 
doi:10.1017/S0022112010001217
'''

# Last update
'''

'''

# ----------------------------------------------------------- #

from pydmd import DMD
import h5py
import meshio
import os
import numpy as np
import math

# ------------------- Parameter inputs ---------------------- #

ti = 10                 # Initial timestep read
tf = 5000               # Final timestep read (max=5000)
par_svd = 10            # SVD rank
par_tlsq = 10           # TLSQ rank
par_exact = True        # Exact (boolean)
par_opt = True          # Opt (boolean)
Pressure_modes = 1      # Run pressure dmd modes? 1 = Y, 0 = N
Pressure_snaps = 1      # Run pressure dmd reconstruction? 1 = Y, 0 = N
Velocity_modes = 0      # Run velocity dmd modes? 1 = Y, 0 = N
Velocity_snaps = 0      # Run velocity dmd reconstruction? 1 = Y, 0 = N
N = 500                 # Pre-defined number of partitions

# ------------------------- Data ---------------------------- #

Pressure_data_code = 'f_26'
Velocity_data_code = 'f_20'
Pressure_mesh_path = 'Cilindro_hdmf/Mesh_data_pressure.vtk'
Pressure_data_path = 'Cilindro_hdmf/solution_p.h5'
Velocity_mesh_path = 'Cilindro_hdmf/Mesh_data_velocity.vtk'
Velocity_data_path = 'Cilindro_hdmf/solution_u.h5'

# ----------------- Reading pressure data ------------------- #

if Pressure_modes == 1 or Pressure_snaps == 1:
    mesh_pressure = meshio.read(Pressure_mesh_path)
    pressure = mesh_pressure.point_data[Pressure_data_code]
    print('Pressure data shape: ', pressure.shape)

    f = h5py.File(Pressure_data_path, 'r')
    data_pressure = f['VisualisationVector']

    snapshots_p = []

    for t in range(ti, tf):
        timestep_p = f[f'VisualisationVector/{t}']
        print(f'Reading time step number {t}')
        snapshots_p.append(timestep_p)

    print(f'{len(snapshots_p)} pressure snapshots were read')
    print()
    N_snaps = math.floor(len(snapshots_p)/N)

    # ---------------------- Pressure DMD ----------------------- #

    current_t = 0

    for n_part in range(0, N):
        partition_ti = n_part*N_snaps
        partition_tf = (n_part+1)*N_snaps

        dmd = DMD(svd_rank=par_svd, tlsq_rank=par_tlsq, exact=par_exact, opt=par_opt)
        dmd.fit(snapshots_p[partition_ti:partition_tf:1])
        print()
        if n_part == 0:
            os.mkdir(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}')

        if Pressure_modes == 1:
            print('DMD modes matrix shape:')
            print(dmd.modes.shape)
            num_modes=dmd.modes.shape[1]
            
            for n in range(0, num_modes):
                print(f'Writing dynamic mode number {n}')
                mode = dmd.modes.real[:, n]
                mesh_pressure.point_data[Pressure_data_code] = mode
                mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/Partition_{n_part}_Mode_{n}.vtk')
            print()

        if Pressure_snaps == 1:
            print('DMD reconstruction matrix shape:')
            print(dmd.reconstructed_data.real.T.shape)

            for t in range(0, partition_tf-partition_ti):
                print(f'Writing dmd timestep number {t}')
                step = dmd.reconstructed_data.real[:, t]
                mesh_pressure.point_data[Pressure_data_code] = step
                mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/DMD_timestep_{current_t}.vtk')
                current_t = current_t + 1
            print()

        with open(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/Partition_{n_part}_Pressure_eigs.txt', 'w') as eigs_p:
            eigs_txt = str(dmd.eigs.real)
            eigs_p.write(eigs_txt)  
