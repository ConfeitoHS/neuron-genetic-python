import numpy as np 
import neuron as N
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import trange
import parmap
import multiprocessing as mp

#settings

inter = 16
sensor = 4
motor = 4

stimulation = 0.2
iteration = 0.05
dt = 0.001
I_s = 700*(10**-12)

min_spikes = 5
p_cross = 0.8
p_mutate = 0.001
thr = 20
Init_gen = 2000
N_gen = 200

ext_score = 10

n = inter+sensor+motor

bits= [ [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0.77,0.77,0,0],
        [0,0.77,0.77,0],
        [0,0,0.77,0.77],
        [0.77,0,0,0.77],
        [0.77,0,0.77,0],
        [0,0.77,0,0.77],]
for i in range(10):
    bits[i].extend([0]*(inter+motor))

bits = np.array(bits)
stims = bits*I_s

num_cores = mp.cpu_count()


def crossover(w1,w2):
    w1 = unpack_feature(w1)
    w2 = unpack_feature(w2)
    shape = w1.shape
    transpose = np.random.randint(2)
    if(transpose):
        w1 = w1.T
        w2 = w2.T
    w1 = w1.flatten()
    w2 = w2.flatten()
    length = w1.shape[0]
    if(np.random.choice(2,p=[1-p_cross,p_cross]) == 1):
        index = np.random.randint(length-1)+1
        for i in range(index,length):
            w1[i],w2[i] = w2[i],w1[i]
    if(np.random.choice(2,p=[1-p_mutate,p_mutate])):
        a,b = np.random.choice(length,2,replace=False)
        w1[a],w1[b]=w1[b],w1[a]
    if(np.random.choice(2,p=[1-p_mutate,p_mutate])):
        a,b = np.random.choice(length,2,replace=False)
        w2[a],w2[b]=w2[b],w2[a]
    w1 = w1.reshape(shape)
    w2 = w2.reshape(shape)
    if(transpose):
        w1 = w1.T
        w2 = w2.T        
    w1 = pack_feature(w1)
    w2 = pack_feature(w2)
    return w1,w2

def pick(scores):
    scores = np.array(scores)
    scores /= sum(scores)
    choices = np.random.choice(len(scores),size=2,p=scores,replace=False).tolist()
    return choices

def period(dat):
    spike_count = 0
    before = 0
    spike_time = []
    for i in range(len(dat)):
        if before == 0 and dat[i] == 1:
            spike_count+=1
            spike_time.append(i*dt)
        before = dat[i]
    if spike_count>=min_spikes:
        return spike_count/(spike_time[-1]-spike_time[0])
    else:
        return 0        

def fitness(w):
    score = 0.0
    net = N.Network(n,np.array(w),dt=dt,V_th=-0.060,t_ref=0.002,t_tx=0.003,t_stim=0.002,I_in=300*(10**-12),R=250*(10**6),C=70*(10**-12))
    
    for i in range(bits.shape[0]):
        net.reset()
        resp = [np.array([0]*motor)]
        t = [0.]
        k=0
        while True:
            if k*dt >= stimulation:
                break
            a,b = net.next(stimulation=np.array(stims[i]))
            resp.append(b[-motor:n])
            t.append(a)
            k+=1
        k=0
        while True:
            if k*dt >= iteration:
                break
            a,b = net.next()
            t.append(a)
            resp.append(b[-motor:n])
            k+=1
        resp = np.array(resp)
        control = np.array([0,0,0,0])
        up = 0
        right = 0
        for j in range(motor):
            r = period(resp[:,j])
            if(r>=50):
                control[j] = r

        up = control[0]-control[2]
        right = control[3]-control[1]
        if up>thr:
            control[0] = 1
        elif up<-thr:
            control[2] = 1
        if right>thr:
            control[3] = 1
        elif right<-thr:
            control[1] = 1
        if(np.dot(control,bits[i][0:4])==0 and np.sum(control)>=1):
            score += 1
            
    return score

def generation(weights):
    '''
    scores = []
    for w in tqdm(weights):
        scores.append(fitness(w))
    '''
    scores = parmap.map(fitness,weights,pm_pbar=True,pm_processes=8)
    w_next = []
    for i in range(N_gen//2):
        a,b=pick(scores)
        w_next.extend(crossover(weights[a],weights[b]))
    return w_next,(sum(scores)/len(scores))

def extraction(weights):
    scores = parmap.map(fitness,weights,pm_pbar=True,pm_processes=8)
    ws = []
    for i in range(len(weights)):
        if scores[i]>=ext_score:
            ws.append(weights[i])
    return ws
def unpack_feature(w):
    return w[sensor:,:inter+sensor]
def pack_feature(w):
    return np.c_[np.r_[np.zeros((sensor,inter+sensor)),w],np.zeros((n,motor))]
saved80 = False
saved85 = False
saved90 = False
saved95 = False
load= True
learn=False
extract=True
if __name__=="__main__":

    ws = []

    if load:
        ws = np.load("./neurons=(16,4,4), 10class/ws_gen=118_i=16_s=4_m=4_98.npy")
    else:
        #랜덤 생성
        for i in range(Init_gen):
            w = np.random.normal(scale=0.27,size=(inter+motor,inter+sensor))
            w = pack_feature(w)
            for i in range(n):
                w[i,i] = 0.0
            ws.append(w)

    if learn:
        iters= []
        scores = []
        gen=0
        while True:
            iters.append(gen)
            print("Generation "+str(gen+1))
            ws,score = generation(ws)
            print(str(gen+1)+" score = "+str(round(score/bits.shape[0]*100,2))+"\n")
            scores.append(score/bits.shape[0]*100)
            if(score/bits.shape[0]*100>=80. and not saved80):
                np.save("ws_gen="+str(gen+1)+"_i="+str(inter)+"_s="+str(sensor)+"_m="+str(motor)+"_80",np.array(ws))
                saved80 = True
            elif(score/bits.shape[0]*100>=85. and not saved85):
                np.save("ws_gen="+str(gen+1)+"_i="+str(inter)+"_s="+str(sensor)+"_m="+str(motor)+"_85",np.array(ws))
                saved85 = True
            elif(score/bits.shape[0]*100>=90. and not saved90):
                np.save("ws_gen="+str(gen+1)+"_i="+str(inter)+"_s="+str(sensor)+"_m="+str(motor)+"_90",np.array(ws))
                saved90 = True
            elif(score/bits.shape[0]*100>=95. and not saved95):
                np.save("ws_gen="+str(gen+1)+"_i="+str(inter)+"_s="+str(sensor)+"_m="+str(motor)+"_95",np.array(ws))
                saved95 = True
            elif(score/bits.shape[0]*100>=98.):
                np.save("ws_gen="+str(gen+1)+"_i="+str(inter)+"_s="+str(sensor)+"_m="+str(motor),np.array(ws))
                break
            gen+=1
        plt.plot(iters,scores)
        plt.show()

    if extract:
        clear=extraction(ws)
        print(len(clear))
        np.save("ws_ext_gen=118_acc=98",clear)

    mp.freeze_support()
