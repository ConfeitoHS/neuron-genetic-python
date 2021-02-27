import numpy as np
import networkx as nx
from  matplotlib import pyplot as plt
import matplotlib

w = np.load("./neurons=(16,4,4), 10class/ws_ext_gen=118_acc=98.npy")[0]

G = nx.DiGraph()

nodes = ['S1','S2','S3','S4']
for i in range(16):
    nodes.append("I"+str(i))
nodes.extend( ['M1','M2','M3','M4'])

for node in nodes:
    G.add_node(node)
labels = {}
for n in nodes:
    labels[n] = n
colors = []
weights = []
view_thr = 0.3
for i in range(24):
    for j in range(24):
        '''
        if w[i,j]>view_thr:
            G.add_edge(nodes[j],nodes[i],weight=w[i,j])
            colors.append("blue")
            weights.append(w[i,j]*5)
        '''
        if w[i,j]<-view_thr:
            G.add_edge(nodes[j],nodes[i],weight=-w[i,j])
            colors.append("red")
            weights.append(-w[i,j]*5)

pos = nx.kamada_kawai_layout(G,scale=2)
pos['S1']=np.array([-3,1.5])
pos['S2']=np.array([-3,0.5])
pos['S3']=np.array([-3,-0.5])
pos['S4']=np.array([-3,-1.5])
pos['M1']=np.array([3,1.5])
pos['M2']=np.array([3,0.5])
pos['M3']=np.array([3,-0.5])
pos['M4']=np.array([3,-1.5])

fig1 = plt.figure(figsize=(16,9))
nx.draw_networkx_nodes(G,pos,node_color="black",node_size=200)
nx.draw_networkx_labels(G,pos,labels,font_color='white',font_size=6)
nx.draw_networkx_edges(G,pos,edge_color=colors,width=weights,alpha=0.5,arrowsize=20,connectionstyle="arc3,rad=0.1")

plt.axis("off")


fig2 = plt.figure()
plt.imshow(w,cmap='seismic_r',vmin=-0.7,vmax=0.7)
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        plt.text(j,i,round(w[i,j],1),color="w",ha="center",va="center")
plt.colorbar()
plt.show()