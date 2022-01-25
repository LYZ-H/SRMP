from __future__ import print_function
# Copyright (C) 2016 Huang MaChi at Chongqing University
# of Posts and Telecommunications, China.
# Copyright (C) 2016 Li Cheng at Beijing University of Posts
# and Telecommunications. www.muzixing.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mininet.net import Mininet
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel, info, debug

import logging
import os
import time


def sys_shell(cmd):
    cmd = cmd.replace('\t', "").replace('\n', "")
    debug('# debug, cmd:', cmd)
    return os.system(cmd)


def run(
        client_num=3, 
        of_version='OpenFlow13',
        bw=100,
        delay='1ms',
        loss=0,
        mcast_ip = '239.0.0.1',
        #max_queue_size=1000,
    ):

    net = Mininet(link=TCLink, autoSetMacs=True, autoStaticArp=True, controller=None)

    ip_offset = 100
    
    ip = '10.0.0.{0}'.format(ip_offset)
    ps = net.addHost('ps', ip=ip)
    ap_switch = net.addSwitch('APSwitch', dpid='100000', protocols=of_version)
    net.addLink(ap_switch, ps, bw=bw, delay=delay)

    workers = []
    for i in range(client_num):
        ip = "10.%d.%d.%d" % (0, 0, ip_offset + i + 1)
        w = net.addHost('w{0}'.format(i+1), ip=ip)
        net.addLink(ap_switch, w, bw=bw, delay=delay,)  #loss=loss,
        # TODO: update link bandwidth and latency .....
        #print('###############', i)
        workers.append(w)
    
    net.start()

    protos = ['ip', ]
    # switch
    egress_buckets = []
    default_table_id = 0
    for i, h in enumerate(net.hosts):
        port_at_switch = i + 1
        port_at_host = 0

        ip = h.IP()
        cmd = "ovs-ofctl add-flow %s -O %s \
            'table=%d,idle_timeout=0,hard_timeout=0,priority=50,ip,nw_dst=%s,actions=output:%d'" % (
                ap_switch.name, of_version, default_table_id, ip, port_at_switch)
        debug(cmd)
        ap_switch.cmd(cmd)

        egress_buckets.append('bucket=output:{0}'.format(port_at_switch))
        #print(port_at_host, port_at_switch)
        cmd = "route add -net 224.0.0.0 netmask 224.0.0.0 {0}-eth{1}".format(h.name, port_at_host)
        debug(cmd)
        h.cmd(cmd)

    # multicast/broadcast forwarding rules
    bucket_s = ','.join(egress_buckets)
    mgroup_id = 1
    cmd = "ovs-ofctl add-group %s -O %s \
    'group_id=%d,type=all,%s'" % (ap_switch.name, of_version, mgroup_id, bucket_s)
    debug(cmd)
    ap_switch.cmd(cmd)
                
    for pro in protos:
        cmd = "ovs-ofctl add-flow %s -O %s \
            'table=%d,priority=10,%s,nw_dst=%s,actions=group:%d'" % (
                ap_switch.name, of_version, default_table_id, pro, mcast_ip, mgroup_id) 
                # TODO
        debug(cmd)
        ap_switch.cmd(cmd)

    if 1:
        ps.cmd('nohup OMP_NUM_THREADS=1 taskset -c 0 python3 main.py --name ps --config-file ./dpwa.yaml > ps.log &')
        for i, w in enumerate(workers):
            w.cmd('nohup OMP_NUM_THREADS=1 taskset -c {} python3 main.py --lr=0.01 --batch-size 8 --name w{} --config-file ./dpwa.yaml > w{}.log &'.format(i+1, i+1, i+1))
    
    CLI(net)
    net.stop()
        

if __name__ == '__main__':
    setLogLevel('info')
    # ryu-manager --observe-links ryu.app.gui_topology.gui_topology

    if os.getuid() != 0:
        logging.debug("You are NOT root")
    elif os.getuid() == 0:
        run()
