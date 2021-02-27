import numpy as np 
import neuron as N
from matplotlib import pyplot as plt
import time
inter=16
sensor=4
motor=4
n = inter+sensor+motor

dt = 0.001
I_s = 700*(10**-12)

V_th=-0.060
t_ref=0.002
t_tx=0.003
t_stim=0.002
I_in=300*(10**-12)
R=250*(10**6)
C=70*(10**-12)

ws = np.load('./neurons=(16,4,4), 10class/ws_ext_gen=118_acc=98.npy')

net = N.Network(n,ws[0],dt=dt,V_th=V_th,t_ref=t_ref,t_tx=t_tx,t_stim=t_stim,I_in=I_in,R=R,C=C)

def period(dat):
    spike_count = 0
    before = 0
    spike_time = []
    for i in range(len(dat)):
        if before == 0 and dat[i] == 1:
            spike_count+=1
            spike_time.append(i*dt)
        before = dat[i]
    if spike_count>=5:
        return spike_count/(spike_time[-1]-spike_time[0])
    else:
        return 0

start = time.time()

timer = 0


while True:
    
    t_i = time.time_ns()
    print(timer)
    if timer>=1:
        break
    net.next()
    timer+=dt
    t_f = time.time_ns()
    time.sleep((10**6-t_f-t_i)*(10**-9))
    
print(time.time()-start)

