# ----------------------------------------------------------- #
#                            pDMD                             #
# ----------------------------------------------------------- #

# Author:               Matheus Seidel (matheus.seidel@coc.ufrj.br)
# Revision:             02
# Last update:          15/05/2023

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
The purpose of this version is to plot the error x N graph. 
The rank is constant.
This code is not supposed to save the DMD modes or the approximation.
'''

# Implementar step no número de subsets

# ----------------------------------------------------------- #

from pydmd import DMD
import h5py
import meshio
import os
import numpy as np
import math
import matplotlib.pyplot as plt

# ------------------- Parameter inputs ---------------------- #

ti = 10                 # Initial timestep read
tf = 5000               # Final timestep read (max=5000)
par_svd = 50            # SVD rank
par_tlsq = 10           # TLSQ rank
par_exact = True        # Exact (boolean)
par_opt = True          # Opt (boolean)
Pressure_modes = 1      # Run pressure dmd modes? 1 = Y, 0 = N
Pressure_snaps = 1      # Run pressure dmd reconstruction? 1 = Y, 0 = N
#Velocity_modes = 0     # Run velocity dmd modes? 1 = Y, 0 = N
#Velocity_snaps = 0     # Run velocity dmd reconstruction? 1 = Y, 0 = N
N_i = 1                 # Initial number of subsets
N_f = 500               # Final number of subsets

# ------------------------- Data ---------------------------- #

Pressure_data_code = 'f_26'
Velocity_data_code = 'f_20'
Pressure_mesh_path = 'Cilindro_hdmf/Mesh_data_pressure.vtk'
Pressure_data_path = 'Cilindro_hdmf/solution_p.h5'
Velocity_mesh_path = 'Cilindro_hdmf/Mesh_data_velocity.vtk'
Velocity_data_path = 'Cilindro_hdmf/solution_u.h5'

# ----------------- Reading pressure data ------------------- #

def read_h5_libmesh(filename, dataset):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    data = h5_file[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array

current_t = 0 

if Pressure_modes == 1 or Pressure_snaps == 1:
    mesh_pressure = meshio.read(Pressure_mesh_path)
    pressure = mesh_pressure.point_data[Pressure_data_code]
    print('Pressure data shape: ', pressure.shape)
    
    num_nodes = pressure.shape[0]
    num_time_steps = tf - ti

    snapshots_p = np.zeros((num_nodes, num_time_steps))

    for t in range(ti, tf):
        snapshots = read_h5_libmesh(Pressure_data_path, f'VisualisationVector/{t}')
        print(f'Reading time step number {t}')
        snapshots_p[:, current_t] = snapshots[:, 0]
        current_t = current_t + 1

    print(f'{snapshots_p.shape[1]} pressure snapshots were read')
    print()

    # ---------------------- Pressure DMD ----------------------- #

    N_total = N_f - N_i
    N_dataset = np.arange(N_i, N_f)
    Error_dataset = np.zeros(N_total)

    for N in range(N_i, N_f):
        N_snaps = math.floor(snapshots_p.shape[1]/N)
        print(f'N_snaps = {N_snaps}')
        print(f'Snapshots read shape: {snapshots_p.shape}')

        current_t = 0

        snapshots_approx = np.zeros((num_nodes, num_time_steps))

        for n_part in range(0, N):
            partition_ti = n_part*N_snaps
            partition_tf = (n_part+1)*N_snaps

            print()
            print(f'Running subset N = {n_part}')
            print(f'subset_ti = {partition_ti}')
            print(f'subset_tf = {partition_tf}')
            print()

            dmd = DMD(svd_rank=par_svd, tlsq_rank=par_tlsq, exact=par_exact, opt=par_opt)
            dmd.fit(snapshots_p[:, partition_ti:partition_tf])
            print()
            #if n_part == 0:
            #    directory = os.path.exists(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}')
            #    if directory == True:
            #        print(f'Directory Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N} already exists')
            #    else:
            #        os.mkdir(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}')

            #if Pressure_modes == 1:
            #    print('DMD modes matrix shape:')
            #    print(dmd.modes.shape)
            #    num_modes=dmd.modes.shape[1]
                
                #for n in range(0, num_modes):
                #    #print(f'Writing dynamic mode number {n}')
                #    mode = dmd.modes.real[:, n]
                #    mesh_pressure.point_data[Pressure_data_code] = mode
                #    mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/Subset_{n_part}_Mode_{n}.vtk')
                #print()

            if Pressure_snaps == 1:
                print('DMD reconstruction matrix shape:')
                print(dmd.reconstructed_data.real.T.shape)
                
                for t in range(0, partition_tf-partition_ti):
                    #print(f'Writing dmd timestep number {t}')
                    step = dmd.reconstructed_data.real[:, t]
                    mesh_pressure.point_data[Pressure_data_code] = step
                    #mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/DMD_timestep_{current_t}.vtk')
                    snapshots_approx[:, current_t] = step
                    current_t = current_t + 1
                print()

            #with open(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/Subset_{n_part}_Pressure_eigs.txt', 'w') as eigs_p:
            #    eigs_txt = str(dmd.eigs.real)
            #    eigs_p.write(eigs_txt)

        E = np.linalg.norm(snapshots_p-snapshots_approx, 'fro')/np.linalg.norm(snapshots_p, 'fro')
        print(f'Error for subset {N}: {E}')
        Error_dataset[N-1] = E

    print()
    print(N_dataset)
    print()
    print(Error_dataset)


    plt.plot(N_dataset, Error_dataset, color="blue")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.title('Error x N')
    plt.savefig(f'Error x N - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N_i}-{N_f}')

    #with open(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}_{N}/Error.txt', 'w') as error_p:
    #    error_txt = str(E)
    #    error_p.write(error_txt)
