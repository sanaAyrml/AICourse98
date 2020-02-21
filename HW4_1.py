class BayesNet:

    def __init__(self,letters,n):
        self.variables = {}
        self.letters = letters
        self.n = n
        for letter in letters:
            self.variables[letter] = Variable(letter)

class Variable:

    def __init__(self, letter):
        self.letter = letter
        self.parents = []
        self.children = []
        self.ifEv = 0
        self.ifActiveUp = 0

n , m , z = list(map(int, input().split()))
bays = BayesNet([i+1 for i in range(n)],n)
for j in range(m):
    c , p = list(map(int, input().split()))
    bays.variables[p].parents.append(c)
    bays.variables[c].children.append(p)
for j in range(z):
    ggg = int(input())
    bays.variables[ggg].ifEv = 1
    for parent in bays.variables[ggg].parents:
        bays.variables[parent].ifActiveUp = 1

q = list(map(int, input().split()))
iter = [(q[0], 1)]
visited = set()
path =[]
flag = 1
while len(iter) > 0:
    flag2 = 0
    (name, direction) = iter.pop()
    node = bays.variables[name]
    if (name, direction) not in visited:
        visited.add((name, direction))
        while (name, direction) in path:
            path.pop()
        path.append((name, direction))
        if node.ifEv == 0 and name == q[1]:
            flag = 0
            break
        if direction == 1 and node.ifEv == 0:
            for parent in node.parents:
                iter.append((parent, 1))
                flag2 = 1
            for child in node.children:
                iter.append((child, 0))
                flag2 = 1
        elif direction == 0:
            if node.ifEv == 1 or node.ifActiveUp == 1:
                for parent in node.parents:
                    iter.append((parent, 1))
                    flag2 = 1
            if node.ifEv == 0:
                for child in node.children:
                    iter.append((child, 0))
                    flag2 = 1
        if flag2 ==0:
            while (name,direction) in path:
                path.pop()
if(flag):
    print("independent")
else:
    for i in path:
        if (i[0],1-i[1]) in path:
            s = path.index((i[0],1-i[1]))
            e = path.index((i[0],i[1]))
            while s!=e:
                path.remove(path[s])
                print(path)
                s -= 1
    print(*[x[0] for x in path], sep=" ")
