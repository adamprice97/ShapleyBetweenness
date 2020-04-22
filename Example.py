from matplotlib.pyplot import figure
import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np
import queue
import colorsys

#Input - G: networkX graph
#Output - cSh: array of ShapelyBetweeness of each node
def ShapelyBetweeness(G):
    #Distance between nodes
    d = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    #list of predecessors on all node pairs
    Pred_s = [[] for i in range(G.number_of_nodes()) ] 
    #Length of shortest path on each pair
    sigma = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    #One-side dependency of source node on target node
    delta = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    cSh = np.zeros(G.number_of_nodes()) 
    #Sructs
    Q = queue.Queue()
    S = []
    #Create node list
    nodes = []
    for n in G.nodes.data():
        nodes.append(n[0])
    for s in range(0, G.number_of_nodes()):
        for v in range(0, G.number_of_nodes()):
            Pred_s[v] = []; d[s,v] = float("inf"); sigma[s,v] = 0
        d[s,s] = 1; sigma[s,s] = 1;  
        Q.put(s)
        #Calulate short paths - num and distance
        while Q.empty() == False:
            v = Q.get()
            S.append(v)
            w = list(G.edges(nodes[v], data=True))
            for i in range(0,len(w)):
                if d[s, nodes.index(w[i][1])] == float("inf"):
                    d[s, nodes.index(w[i][1])] = d[s, v] + 1
                    Q.put(nodes.index(w[i][1]))
                if d[s, nodes.index(w[i][1])] == d[s, v] + 1:
                    sigma[s,nodes.index(w[i][1])] += sigma[s,v]
                    Pred_s[nodes.index(w[i][1])].append(v)

        for v in range(0, G.number_of_nodes()-1):
            delta[s,v] = 0

        #Calculate contribution
        while len(S) > 0:
            w = S.pop()
            for v in Pred_s[w]:
                delta[s,v] += (sigma[s,v]/sigma[s,w])*(1/d[s,w] + delta[s,w])
            if w != s:
                cSh[w] += delta[s,w] + (2-d[s,w])/d[s,w]

    for v in range(0, G.number_of_nodes()):
        cSh[v] = cSh[v]/2

    return cSh

G=nx.Graph()
G.add_edge('v1', 'v9')
G.add_edge('v2', 'v9')
G.add_edge('v3', 'v9')
G.add_edge('v4', 'v9')
G.add_edge('v4', 'v5')
G.add_edge('v5', 'v10')
G.add_edge('v6', 'v10')
G.add_edge('v7', 'v10')
G.add_edge('v8', 'v10')
G.add_edge('v10', 'v11')
G.add_edge('v12', 'v9')
G.add_edge('v13', 'v9')
G.add_edge('v14', 'v9')
G.add_edge('v15', 'v9')
G.add_edge('v16', 'v11')
G.add_edge('v17', 'v11')
G.add_edge('v18', 'v11')
G.add_edge('v19', 'v11')

pos = {'v1':[0,0], 'v2':[.1,0], 'v3':[.2,0], 'v4':[.3,0], 'v5':[.4,0], 'v6':[.5,0], 'v7':[.6,0], 'v8':[.7,0], 
       'v9':[.15,-.1], 'v10':[.55,-.066], 'v11':[.55,-.133],
       'v12':[0,-.2], 'v13':[.1,-.2], 'v14':[.2,-.2], 'v15':[.3,-.2], 'v16':[.4,-.2], 'v17':[.5,-.2], 'v18':[.6,-.2], 'v19':[.7,-.2]}

cSh = ShapelyBetweeness(G)
#cSh = list(nx.betweenness_centrality(G, normalized=False).values())
colourMap = []
for i in range(0, len(cSh)): 
    hue = -.5*((cSh[i]-min(cSh))/(max(cSh)-min(cSh)))+0.5
    col = colorsys.hls_to_rgb(hue,.5,1)
    col += (1.,)
    colourMap.append(col)

figure(num=None, figsize=(6, 4))
nx.draw(G, pos, with_labels=True, edge_color=[.25,.25,.25,.8], font_color=[0,0,0,1], node_size=600, node_color=colourMap)

names = []
for n in G.nodes.data():
    names.append(n[0])
zip_ = zip(names, cSh)
dic = dict(zip_)
c = sorted(dic.items(), reverse=True, key=lambda x: x[1])
for elem in c:
    print(elem[0] , " ::" , elem[1] )

plt.show() 

