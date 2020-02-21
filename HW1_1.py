import math
class Node:
    def __init__(self, num):
        self.num = num
        self.d = math.inf
        self.parent = None
class Graph:
    def __init__(self, nodes):
        self.V = nodes
        self.E = [[0 for i in range(len(nodes))] for j in range(len(nodes))]

def relax(G, u, v):
    if v.d>u.d+G.E[u.num][v.num]:
        v.d =u.d+G.E[u.num][v.num]
        v.parent = u

def find_min_distance(G,S):
    mini= math.inf
    min_node=None
    for v in G.V:
        if v.d<mini and S[v.num]==0:
            min_node = v
            mini= v.d
    return min_node

def dij(G,s):
    s.d = 0
    S = [0] * len(G.V)
    for i in range(len(G.V)):
        u =find_min_distance(G, S)
        if u is None:
            return
        S[u.num]=1
        for v in G.V:
            if G.E[u.num][v.num]>0 and S[v.num]==0:
                relax(G,u,v)
    return

def find_list_of_nodes(graph,y):
    last_node = graph.V[y - 1]
    list_of_nodes = []
    while last_node.parent != None:
        list_of_nodes = list_of_nodes + [last_node.num ]
        last_node = last_node.parent
    list_of_nodes = list_of_nodes + [last_node.num]
    return list_of_nodes

n,m,x,y,k = map(int,input().split(" "))
nodes = [Node(i) for i in range(n)]
graph = Graph(nodes)
number_of_unKnown = 0
nodesW = [Node(i) for i in range(n)]
graphW = Graph(nodesW)
tuples = []
unknowns_tuples = []
for i in range(m):
    x1,x2,d= map(int,input().split(" "))
    if d != -1:
        graph.E[x1 - 1][x2 - 1] = d
        graph.E[x2 - 1][x1 - 1] = d
        graphW.E[x1 - 1][x2 - 1] = d
        graphW.E[x2 - 1][x1 - 1] = d
    else:
        graph.E[x1 - 1][x2 - 1] = 1
        graph.E[x2 - 1][x1 - 1] = 1
        unknowns_tuples = unknowns_tuples+[[x1-1,x2-1]]
        graphW.E[x1 - 1][x2 - 1] = math.inf
        graphW.E[x2 - 1][x1 - 1] = math.inf
    tuples = tuples+[[x1,x2,d]]
dij(graph,graph.V[x-1])
dij(graphW,graphW.V[x-1])
if graphW.V[y-1].d == k:
    print("YES")
    for i, j, z in tuples:
        print(i, j, graphW.E[i - 1][j - 1])
elif graphW.V[y-1].d < k:
    print("NO")
else:
    if graph.V[y-1].d == 0 or graph.V[y-1].d == math.inf:
        print("NO")
    else:
        flag = 1
        while flag:
            if graph.V[y - 1].d > k:
                print("NO")
            elif graph.V[y - 1].d == k:
                print("YES")
                for i, j, z in tuples:
                    print(i, j, graph.E[i - 1][j - 1])
                flag = 0
            else:
                needed = k - graph.V[y - 1].d
                list_of_nodes = find_list_of_nodes(graph, y)
                for i in range(len(list_of_nodes)-1):
                    if unknowns_tuples.__contains__([list_of_nodes[i],list_of_nodes[i+1]]) or unknowns_tuples.__contains__([list_of_nodes[i+1],list_of_nodes[i]]):
                        graph.E[list_of_nodes[i]][list_of_nodes[i+1]] +=  needed
                        graph.E[list_of_nodes[i+1]][list_of_nodes[i]] +=  needed
                        for o in nodes:
                            o.d = math.inf
                        dij(graph,graph.V[x-1])
                        if graph.V[y-1].d == k:
                            pass
                        else:
                            graph.E[list_of_nodes[i]][list_of_nodes[i + 1]] = math.inf
                            graph.E[list_of_nodes[i + 1]][list_of_nodes[i]] = math.inf
                            for o in nodes:
                                o.d = math.inf
                            dij(graph,graph.V[x-1])
                        break
