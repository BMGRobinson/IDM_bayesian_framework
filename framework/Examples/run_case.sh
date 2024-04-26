#!/bin/bash
#SBATCH --account=def-asarkar
#SBATCH --ntasks=26
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=2-00:00
mpirun -n 2 ./run case00.cfg
#mpirun -n 1 ./run se.cfg
#mpirun -n 26 ./run case00cj.cfg
#mpirun -n 10 ./run case00se.cfg  
