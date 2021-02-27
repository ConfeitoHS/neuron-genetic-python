import numpy as np 
import math

class Neuron:
    
    def __init__(self,E_r=-0.070,V_th=-0.050,R=200*(10**6),C=150*(10**-12),t_ref=0.001,t_stim = 0.001,t_tx=0.001,dt=0.001):
        self.tx_queue = []
        self.V_hist = []
        self.ref_left = 0
        self.stim_left = 0
        self.E_r=E_r
        self.V_th = V_th
        self.tx_step = round(t_tx/dt)
        self.t_ref=t_ref
        self.R = R
        self.C = C
        self.dt = dt
        self.t_stim = t_stim
        self.V_hist.append(E_r)

    def update_V(self, I_in):
        res = 0
        if(self.ref_left<=0):
            temp = self.V_hist[-1] + self.dt * (self.E_r + self.R * I_in - self.V_hist[-1])/ (self.R*self.C)
            self.V_hist.append(temp)
            if(temp>=self.V_th):
                self.ref_left = self.t_ref
                self.stim_left=self.t_stim
                res=1
            else:
                res = 0
        else:
            self.ref_left-=self.dt
            self.V_hist.append(self.E_r)
            res = 0

        if(self.stim_left>0):
            res = 1
            self.stim_left -= self.dt

        return res
    
    def evaluate(self,I_in):
        self.tx_queue.append(self.update_V(I_in))
        if(len(self.tx_queue)>self.tx_step):
            return self.tx_queue.pop(0)
        else:
            return 0

class Network:
    
    def __init__(self,N,W,E_r=-0.070,V_th=-0.050,R=200*(10**6),C=150*(10**-12),t_ref=0.001,t_stim=0.001,t_tx=0.001,dt=0.001,I_in = 150*(10**-12)):
        self.time = 0
        self.neurons = []
        self.I_before = None
        for i in range(N):
            n = Neuron(E_r=E_r,V_th=V_th,t_ref=t_ref,R=R,C=C,t_stim=t_stim,t_tx=t_tx,dt=dt)
            self.neurons.append(n)
        self.N = N
        assert W.shape == (N,N)
        self.W = W
        self.I_in = I_in
        self.I_before = np.zeros((self.N,))
        self.time_step=dt
        self.E_r = E_r
        self.V_th = V_th
        self.R = R
        self.C = C
        self.t_ref = t_ref
        self.t_stim = t_stim
        self.t_tx = t_tx
    
    def next(self,stimulation=None):
        assert stimulation is None or stimulation.shape == (self.N,)
        output = []
        if stimulation is None:
            stimulation = np.zeros((self.N,))

        for n,i,s in zip(self.neurons,self.I_before,stimulation):
            output.append(n.evaluate(i+s))
        output = np.array(output)

        assert output.shape == (self.N,)

        self.I_before = np.dot(self.W,output) * self.I_in
        self.time += self.time_step

        return (self.time,output)

    def reset(self):
        self.time = 0
        self.neurons = []
        for i in range(self.N):
            n = Neuron(E_r=self.E_r,V_th=self.V_th,t_ref=self.t_ref,R=self.R,C=self.C,t_stim=self.t_stim,t_tx=self.t_tx,dt=self.time_step)
            self.neurons.append(n)
        
        

        


