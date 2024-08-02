#!/bin/bash
#SBATCH --job-name=opsum_db    # Job name
#SBATCH --partition=public-cpu
#SBATCH --ntasks=1
#SBATCH --mem=100G                     # Job memory request
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/users/k/klug/logs/short_term_opsum/opsum_db_%j.log   # Standard output and error log

cd
echo $SLURM_NODELIST
./tools/redis/bin/redis-server ./tools/redis/redis.conf --port 6380
sleep 5m

# name available from other nodes as:
#squeue -n opsum_db -o "%R" -h