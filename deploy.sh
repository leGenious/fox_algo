#!/bin/bash

scp fox_algo.* stromboli:/home/m1434856/Lab2/ex7
ssh -t stromboli "cd /home/m1434856/Lab2/ex7; make fox_algo; rm slurm*; sleep 0.5; sbatch submit.sh; bash"
