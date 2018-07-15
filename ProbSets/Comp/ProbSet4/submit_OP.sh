#!/bin/bash
# This is a script to submit Option Pricing project to the midway cluster

# set the job
#SBATCH --job-name=Option_Pricing_Project

# send output to hello-world.out
#SBATCH --output=pi.out

# receive an email when job starts, ends, and fails
#SBATCH --mail-type=BEGIN,END,DAIL

#SBATCH --account=osmlab

# this job requests 1 core. Cores can be selected from various nodes.
#SBATCH --ntasks=1

#SBATCH --partition=sandyb

# Run the process
./BS.exec

