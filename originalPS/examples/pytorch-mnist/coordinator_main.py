'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import numpy as np
import os
import argparse
import sys
import pickle

ext_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../")
sys.path.append(ext_path)
from dpwa.coordinator import CoordinatorClass
import time

import logging

def assert_mu(var_mu):
    if var_mu >= 0.5:
       raise Exception('Error: mu should be set to less than 0.5, or the topology algorithm has no solution 0')
    if var_mu <= 0:
       raise Exception('Error: mu should be positve')
    return True

def init_logging(filename):
    # Create the logs directory
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    # Init logging to file
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        filename="./logs/%s" % filename,
                        filemode='w',
                        level=logging.DEBUG)

    # logging.getLogger("dpwa.conn").setLevel(logging.INFO)


LOGGER = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--beta', default=0.95, type=float, help='exponential moving average rate for estimating bandwidth')
parser.add_argument('--mu', default=0.001, type=float, help='mu = learning_rate * rho')
parser.add_argument('--period', default = 12, type=int, help='the period to computing topology')
parser.add_argument('--request_period', default=18,type=int, help='the topology request-period for each worker')
parser.add_argument('--config-file', type=str, default='dpwa.yaml', help='Dpwa configuration file')
parser.add_argument('--name', type=str, default='coordinator', help="This worker's name within config file")
parser.add_argument('--adaptive', type=int, default=1, help="1: adaptive mu; 0: fix mu")

args = parser.parse_args()

init_logging(args.name + ".log")
assert_mu(args.mu)
ctr = CoordinatorClass(args.config_file,args.mu)
node_num = len(ctr.nodes)
bw_matrix = np.zeros((node_num,node_num))
c_matrix = np.zeros((node_num,node_num))
c_matrix[0] = [0.25,0.25,0.25,0.25]
c_matrix[1] = [0.25,0.25,0.25,0.25]
c_matrix[2] = [0.25,0.25,0.25,0.25]
c_matrix[3] = [0.25,0.25,0.25,0.25]
# Initial Graph
graph = np.zeros((node_num,node_num))
graph[0] = [1.0,1.0,1.0,1.0]
graph[1] = [1.0,1.0,1.0,1.0]
graph[2] = [1.0,1.0,1.0,1.0]
graph[3] = [1.0,1.0,1.0,1.0]
   
state = {}
state['beta'] = args.beta
state['request_period'] = args.request_period
state['mu'] = args.mu
 
ctr.set_topo(c_matrix,state)
first_iter = 0
is_adaptive = args.adaptive
while True:
   try:
      r_list = ctr.get_bwmsg()
      if r_list == None:
         if is_adaptive > 0:
            res = ctr.compute_topo_adptive(graph,bw_matrix)
         else:
            res = ctr.compute_topo(graph,bw_matrix)
         if res is None:
            #LOGGER.debug("Warning:compute topology failed")
            #print('==>Warning:compute topology failed')
            #mt = c_matrix
            time.sleep(args.period)
            continue
         else:
            bw_matrix = np.zeros((node_num,node_num))
            mt = res['P']
            state['mu'] = res['mu']
            ctr.set_topo(mt,state)
            print('===>set_topo')
            print(mt)
            print('===>set_state')
            print(state)
            first_iter += 1
            if first_iter >= 10:
               state['request_period'] =500
               args.period =520
            continue
      msg = r_list[0]
      bw_info = r_list[1]
      node_index = bw_info['node_index']
      node_bw = bw_info['bw']
      bw_matrix[node_index] = node_bw
      #print('bw_matrix is ')
      #print(bw_matrix)
   except Exception as e:
          LOGGER.debug("Warning: coordinator_main.py exception %s", str(e))
          print(str(e))
          bw_matrix = np.zeros((node_num,node_num))


