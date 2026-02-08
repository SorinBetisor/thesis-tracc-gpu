#!/bin/bash
set -euo pipefail

echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "ls:"
ls -lah

# Compile + run
gcc -O2 -o myprog myprog.c
./myprog 1000000