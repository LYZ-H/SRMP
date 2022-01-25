import os
import time
import subprocess

tests = [[30, 0.1, 1, "np", "./dpwa_mc.yaml"],
         [30, 0.1, 0, "np", "./dpwa_mc.yaml"]]

test_time = 6

for test_num in range(test_time):
    for test in tests:
        loss = test[0]
        alr = test[1]
        feedback = test[2]
        tag = test[3]
        config = test[4]
        
        os.system("pkill mn")
        os.system("mn -c")
        os.system("ps -ef | grep topo | awk '{ print $2 }' | sudo xargs kill -9")
        os.system("ps -ef | grep mininet | awk '{ print $2 }' | sudo xargs kill -9")
        os.system("ps -ef | grep ovs-controller | awk '{ print $2 }' | sudo xargs kill -9")

        process = subprocess.Popen(["python", "net.py" ,str(loss),"&"])   # pass cmd and args to the function

        time.sleep(6)
        
        state = "{}-{}-{}-{}".format(loss, alr, tag, feedback)
        print("------------------------start:{}-{}------------------------".format(test_num, state))
        cmd = ''
        cmd += "OMP_NUM_THREADS=1 taskset -c 31 ../../../mininet/mc.sh ps python3 main.py --alr={} --name=ps --state={} --config-file ./dpwa_mc.yaml &".format(alr, state)
        cmd += "OMP_NUM_THREADS=1 taskset -c 32 ../../../mininet/mc.sh w1 python3 main.py --alr={} --lr=0.0001 --batch-size 64 --fb={} --name=w1 --state={} --config-file {} &".format(alr, feedback, state, config)
        cmd += "OMP_NUM_THREADS=1 taskset -c 33 ../../../mininet/mc.sh w2 python3 main.py --alr={} --lr=0.0001 --batch-size 64 --fb={} --name=w2 --state={} --config-file {} &".format(alr, feedback, state, config)
        cmd += "OMP_NUM_THREADS=1 taskset -c 34 ../../../mininet/mc.sh w3 python3 main.py --alr={} --lr=0.0001 --batch-size 64 --fb={} --name=w3 --state={} --config-file {} &".format(alr, feedback, state, config)
        cmd += "OMP_NUM_THREADS=1 taskset -c 35 ../../../mininet/mc.sh w4 python3 main.py --alr={} --lr=0.0001 --batch-size 64 --fb={} --name=w4 --state={} --config-file {} &".format(alr, feedback, state, config)
        cmd += "OMP_NUM_THREADS=1 taskset -c 36 ../../../mininet/mc.sh w5 python3 main.py --alr={} --lr=0.0001 --batch-size 64 --fb={} --name=w5 --state={} --config-file {} &".format(alr, feedback, state, config)
        os.system(cmd)
        
        while True:
            f = open("state.txt", "a+")
            if len(f.readlines()) >= 5:
                f.truncate(0)
                print("------------------------done:{}-{}------------------------".format(test_num, state))
                os.system("ps -ef | grep main.py | awk '{ print $2 }' | sudo xargs kill -9")
                time.sleep(5)
                fs = open("./result/{}.txt".format(state), "a")
                fs.write("-\n")
                fs.close()
                break
            f.close()
            time.sleep(1)




# test_num = 0

# for test_num, test in enumerate(tests):
#     loss = test[0]
#     alr = test[1]
#     test_time = test[2]
#     feedback = test[3]
#     tag = test[4]
#     config = test[5]
    
#     for i in range(test_time):
#         os.system("pkill mn")
#         os.system("mn -c")
#         os.system("ps -ef | grep topo | awk '{ print $2 }' | sudo xargs kill -9")
#         os.system("ps -ef | grep mininet | awk '{ print $2 }' | sudo xargs kill -9")
#         os.system("ps -ef | grep ovs-controller | awk '{ print $2 }' | sudo xargs kill -9")

#         process = subprocess.Popen(["python", "net.py" ,str(loss),"&"])   # pass cmd and args to the function

#         time.sleep(6)
        
#         state = "{}-{}-{}-{}".format(loss, alr, tag, feedback)
#         print("------------------------start:{}-{}------------------------".format(test_num,state))
#         cmd = ''
#         cmd += "OMP_NUM_THREADS=1 taskset -c 11 ../../../mininet/mc.sh ps python3 main.py --alr={} --name ps --state={} --config-file ./dpwa_mc.yaml &".format(alr, state)
#         cmd += "OMP_NUM_THREADS=1 taskset -c 12 ../../../mininet/mc.sh w1 python3 main.py --alr={} --lr=0.001 --batch-size 64 --fb={} --name w1 --state={} --config-file {} &".format(alr, feedback, state, config)
#         cmd += "OMP_NUM_THREADS=1 taskset -c 13 ../../../mininet/mc.sh w2 python3 main.py --alr={} --lr=0.001 --batch-size 64 --fb={} --name w2 --state={} --config-file {} &".format(alr, feedback, state, config)
#         cmd += "OMP_NUM_THREADS=1 taskset -c 14 ../../../mininet/mc.sh w3 python3 main.py --alr={} --lr=0.001 --batch-size 64 --fb={} --name w3 --state={} --config-file {} &".format(alr, feedback, state, config)
#         cmd += "OMP_NUM_THREADS=1 taskset -c 16 ../../../mininet/mc.sh w4 python3 main.py --alr={} --lr=0.001 --batch-size 64 --fb={} --name w4 --state={} --config-file {} &".format(alr, feedback, state, config)
#         cmd += "OMP_NUM_THREADS=1 taskset -c 17 ../../../mininet/mc.sh w5 python3 main.py --alr={} --lr=0.001 --batch-size 64 --fb={} --name w5 --state={} --config-file {} &".format(alr, feedback, state, config)
#         os.system(cmd)
        
#         while True:
#             f = open("state.txt", "a+")
#             if len(f.readlines()) >= 5:
#                 f.truncate(0)
#                 print("------------------------done:{}-{}------------------------".format(test_num,state))
#                 os.system("ps -ef | grep main.py | awk '{ print $2 }' | sudo xargs kill -9")
#                 time.sleep(5)
#                 fs = open("./result/{}.txt".format(state), "a")
#                 fs.write("-\n")
#                 fs.close()
#                 break
#             f.close()
#             time.sleep(1)
    
