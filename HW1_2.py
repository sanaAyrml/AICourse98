import math
class Node:
    def __init__(self, n):
        self.pattern = [[0 for i in range(n)] for j in range(n)]
        self.parent = None
        self.huristic = 0
        self.hashCode = 0
        self.source = [-1,-1]
        self.dist = [-1,-1]

def hashing(pattern):
    array = []
    for i in range(n):
        for j in range(n):
            array.append(pattern[i][j])
    strPat = str(pattern)
    return hash(strPat)

def chechIfEnd(node):
    flag = 0
    for i in range(n):
        for j in range(n):
            if node.pattern[i][j]!=0:
                flag += 1
    if flag == 1:
        return True
    else:
        return False

def huristic_function(node):
    colors_num = [0 for i in range(m)]
    for i in range(n):
        for j in range(n):
            colors_num[node.pattern[i][j]-1] += 1
    nummm = 0
    for i in colors_num:
        if i != 0:
            nummm += 1
    return nummm-1

def genrate_childeren(node):
    childeren = []
    for x in range(n):
        for y in range(n):
            if node.pattern[x][y] == 0:
                if x+1>-1 and x+1 < n and (node.pattern[x+1][y] != 0):
                    child = find_child(node,x,y,1,0)
                    if child != None:
                        childeren.append(child)
                if y+1>-1 and y+1 < n and(node.pattern[x][y+1] != 0):
                    child =find_child(node,x,y, 0, 1)
                    if child != None:
                        childeren.append(child)
                if y-1>-1 and y-1 < n and (node.pattern[x][y-1] != 0):
                    child = find_child(node,x,y, 0, -1)
                    if child != None:
                        childeren.append(child)
                if x-1>-1 and x-1 < n and(node.pattern[x-1][y] != 0):
                    child=find_child(node,x,y, -1, 0)
                    if child != None:
                        childeren.append(child)
    return childeren
def P(pattern):
    for i in pattern:
        print(i)
    print("------------------")


def find_child(node,x,y,i,j):
    number = 1
    color = node.pattern[x+i][y+j]
    new_node = Node(n)
    new_node.source = [x,y]
    for q in range(n):
        for w in range(n):
            new_node.pattern[q][w]= node.pattern[q][w]
    new_node.parent = node
    while x+number*i>-1 and x+number*i<n and y+number*j>-1 and y+number*j<n and new_node.pattern[x+number*i][y+number*j] == color:
        new_node.pattern[x + number * i][y + number * j] = 0
        number += 1
    if x+number*i>-1 and x+number*i<n and y+number*j>-1 and y+number*j<n:
        if new_node.pattern[x + number * i][y + number * j] != 0:
            new_node.pattern[x][y] = new_node.pattern[x + number * i][y + number * j]
            new_node.pattern[x + number * i][y + number * j] = 0
            new_node.dist = [x+number*i, y+number*j]
            new_node.huristic = huristic_function(new_node)
            new_node.hashCode = hashing(new_node.pattern)
            # P(new_node.pattern)

            return new_node
        else:
            return
    else:
        return

def find_min_huristic(fridge):
    mini= math.inf
    min_node_i=-1
    for node_index in range(len(fridge)):
        if fridge[node_index].huristic < mini:
            min_node_i = node_index
            mini = fridge[node_index].huristic
    return min_node_i

def aStar(start):

    fridge = []
    fridges_hash = []
    closed = []
    closeds_hash = []
    if chechIfEnd(start):
        return None
    fridge.append(start)
    fridges_hash.append(start.hashCode)
    while len(fridge)>0:
        index = find_min_huristic(fridge)
        if fridge == -1:
            return None
        next_node = fridge[index]
        # print("here we go next node")
        # P(next_node.pattern)
        fridge.pop(index)
        fridges_hash.pop(index)
        closed.append(next_node)
        closeds_hash.append(next_node.hashCode)
        if chechIfEnd(next_node):
            path = []
            current = next_node
            while current.parent is not None:
                path.append([current.source[0],current.source[1],current.dist[0],current.dist[1]])
                current = current.parent
            return path
        childeren = genrate_childeren(next_node)
        for child in childeren:
            if closeds_hash.__contains__(child.hashCode) or fridges_hash.__contains__(child.hashCode):
                pass
            else:
                fridges_hash.append(child.hashCode)
                fridge.append(child)
n = int(input())
m = int(input())
node = Node(n)
# P(node.pattern)
for i in range(n):
    node.pattern[i] = list(map(int,input().split(" ")))
# P(node.pattern)
node.huristic = huristic_function(node)
node.hashCode = hashing(node.pattern)
path = aStar(node)
if path!=None:
    for i in range(len(path)-1,-1,-1):
        print(path[i][2]+1,",",path[i][3]+1," ",path[i][0]+1,",",path[i][1]+1)


