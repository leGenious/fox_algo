#!/bin/bash

scp fox_algo.c stromboli:/home/m1434856/Lab2/ex7
scp Makefile stromboli:/home/m1434856/Lab2/ex7
ssh -t stromboli "cd /home/m1434856/Lab2/ex7; make; rm slurm*; sbatch submit.sh; bash"
