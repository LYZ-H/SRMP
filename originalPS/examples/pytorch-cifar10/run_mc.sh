#!/bin/bash

# ../../../mininet/mc.sh h1 python3 main.py --alr=0.0 --name ps --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h2 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w1 --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h3 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w2 --config-file ./dpwa_mc.yaml &
# ../../../mininet/mc.sh h4 python3 main.py --alr=0.0 --lr=0.001 --batch-size 64 --name w3 --config-file ./dpwa_mc.yaml &


OMP_NUM_THREADS=1 taskset -c 30 ../../../mininet/mc.sh ps python3 main.py --state="test" --alr=0.1 --fb=1 --name ps --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 31 ../../../mininet/mc.sh w1 python3 main.py --state="test" --alr=0.1 --fb=1 --lr=0.0001 --batch-size 64 --name w1 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 32 ../../../mininet/mc.sh w2 python3 main.py --state="test" --alr=0.1 --fb=1 --lr=0.0001 --batch-size 64 --name w2 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 33 ../../../mininet/mc.sh w3 python3 main.py --state="test" --alr=0.1 --fb=1 --lr=0.0001 --batch-size 64 --name w3 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 34 ../../../mininet/mc.sh w4 python3 main.py --state="test" --alr=0.1 --fb=1 --lr=0.0001 --batch-size 64 --name w4 --config-file ./dpwa_mc.yaml &
OMP_NUM_THREADS=1 taskset -c 35 ../../../mininet/mc.sh w5 python3 main.py --state="test" --alr=0.1 --fb=1 --lr=0.0001 --batch-size 64 --name w5 --config-file ./dpwa_mc.yaml &
