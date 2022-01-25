#!/bin/bash

# ../../../mininet/mc.sh h1 python3 main.py --alr=0.0 --name ps --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h2 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w1 --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h3 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w2 --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h4 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w3 --config-file ./dpwa_mc.yaml &


OMP_NUM_THREADS=1 taskset -c 0 ../../../mininet/mc.sh ps python3 main.py --state="test" --alr=0.05 --fb=1 --name ps --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 1 ../../../mininet/mc.sh w1 python3 main.py --state="test" --alr=0.05 --fb=1 --lr=0.001 --batch-size 64 --name w1 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 2 ../../../mininet/mc.sh w2 python3 main.py --state="test" --alr=0.05 --fb=1 --lr=0.001 --batch-size 64 --name w2 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 3 ../../../mininet/mc.sh w3 python3 main.py --state="test" --alr=0.05 --fb=1 --lr=0.001 --batch-size 64 --name w3 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 4 ../../../mininet/mc.sh w4 python3 main.py --state="test" --alr=0.05 --fb=1 --lr=0.001 --batch-size 64 --name w4 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 5 ../../../mininet/mc.sh w5 python3 main.py --state="test" --alr=0.05 --fb=1 --lr=0.001 --batch-size 64 --name w5 --config-file ./dpwa_mc.yaml &
