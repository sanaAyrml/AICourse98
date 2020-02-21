import queue


def makeSudoko(grid, labels):
    i = 0
    j = 0
    values = dict()
    for cell in labels:
        if grid[j][i] != 0:
            values[cell] = [grid[j][i]]
        else:
            values[cell] = nums
        i = i + 1
        if i > 8:
            i = 0
            j = j + 1
    return values


def make_labels(A, B):
    return [str(a) + str(b) for a in A for b in B]


def AC_3(csp,arcs,cons):
    q = queue.Queue()
    for arc in arcs:
        q.put(arc)
    while not q.empty():
        (x_i, x_j) = q.get()
        if Remove_Inconsistent_Value(x_i, x_j, csp):
            if len(csp[x_i]) == 0:
                return False
            for x_k in (cons[x_i] - set(x_j)):
                q.put((x_k, x_i))
    return True


def Remove_Inconsistent_Value(x_i, x_j,csp):
    removed = False
    for x in csp[x_i]:
        if notConsistent(x_i, x_j,x,cons,csp):
            csp[x_i] = list(filter(lambda w: w != x,csp[x_i]))
            removed = True
    return removed


def notConsistent( x_i, x_j,x,cons,csp):
    if x_j in cons[x_i]:
        for y in csp[x_j]:
            if y != x:
                return False
    return True


array = []
for i in range(9):
    array.append(list(map(int, input().split(" "))))

nums = [i + 1 for i in range(9)]
labels = make_labels(nums, nums)
sudoku = makeSudoko(array, labels)
labelOfConstarinsGroup = ([make_labels(nums, [i]) for i in nums] + [make_labels([j], nums) for j in nums]
                          + [make_labels(k, s) for k in [[1, 2, 3], [4, 5, 6], [7, 8, 9]] for s in
                             [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
cons = dict((l,set()) for l in labels)
for l in labels:
    for c in labelOfConstarinsGroup:
        if l in c:
            for i in c:
                if i != l:
                    cons[l].add(i)
arcs = {(0, 0)}
arcs.pop()
for label in labels:
    for i in cons[label]:
        arcs.add((label, i))
solved = AC_3(sudoku,arcs,cons)
result = []
j = 1
k = 1
o = []
for i in range(90):
   if k < 10 :
       o.append(sudoku[str(j)+str(k)][0])
       k += 1
   else:
       result.append(o)
       o = []
       j += 1
       k = 1
result.append(o)
for r in result:
    print(*[x for x in r], sep=" ")

