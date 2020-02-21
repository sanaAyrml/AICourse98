from random import randint
from random import uniform
def re_shuffle(repeat,puzzle,zero_loc):
    for i in range(repeat):
        puzzle, zero_loc = shuffle(puzzle,zero_loc)
        # print(i,"      ",puzzle, zero_loc)
    return puzzle, zero_loc

def shuffle(puzzle,zero_loc):
    w = randint(0, 3)
    if w == 0:
        puzzle, zero_loc = shuffle_right(puzzle,zero_loc)
    if w == 1:
        puzzle, zero_loc = shuffle_left(puzzle,zero_loc)
    if w == 2:
        puzzle, zero_loc = shuffle_up(puzzle,zero_loc)
    if w == 3:
        puzzle, zero_loc = shuffle_down(puzzle,zero_loc)
    return puzzle, zero_loc

def copy(puzzle):
    new_puzzle = [[],[],[]]
    for i in range(3):
        for j in range(3):
            new_puzzle[i].append(puzzle[i][j])
    return new_puzzle

def copy_zero(zero_loc):
    new_zero = []
    for i in range(2):
        new_zero.append(zero_loc[i])
    return new_zero


def shuffle_right(new_puzzle, zero_loc):
    # new_puzzle = copy(puzzle)
    if zero_loc[1]+1 < 3 :
        new_puzzle[zero_loc[0]][zero_loc[1]] = new_puzzle[zero_loc[0]][zero_loc[1]+1]
        new_puzzle[zero_loc[0]][zero_loc[1]+1] = 0
        zero_loc[1] = zero_loc[1]+1
        return new_puzzle,zero_loc
    else:
        return shuffle_left(new_puzzle, zero_loc)

def shuffle_left(new_puzzle, zero_loc):
    # new_puzzle = copy(puzzle)
    if zero_loc[1]-1 > -1 :
        new_puzzle[zero_loc[0]][zero_loc[1]] = new_puzzle[zero_loc[0]][zero_loc[1]-1]
        new_puzzle[zero_loc[0]][zero_loc[1]-1] = 0
        zero_loc[1] = zero_loc[1] -1
        return new_puzzle,zero_loc
    else:
        return shuffle_right(new_puzzle, zero_loc)

def shuffle_up(new_puzzle, zero_loc):
    # new_puzzle = copy(puzzle)
    if zero_loc[0]-1 > -1 :
        new_puzzle[zero_loc[0]][zero_loc[1]] = new_puzzle[zero_loc[0]-1][zero_loc[1]]
        new_puzzle[zero_loc[0]-1][zero_loc[1]] = 0
        zero_loc[0] = zero_loc[0]-1
        return new_puzzle,zero_loc
    else:
        return shuffle_down(new_puzzle, zero_loc)

def shuffle_down(new_puzzle, zero_loc):
    # new_puzzle = copy(puzzle)
    if zero_loc[0]+1 < 3 :
        new_puzzle[zero_loc[0]][zero_loc[1]] = new_puzzle[zero_loc[0]+1][zero_loc[1]]
        new_puzzle[zero_loc[0]+1][zero_loc[1]] = 0
        zero_loc[0] = zero_loc[0]+1
        return new_puzzle,zero_loc
    else:
        return shuffle_up(new_puzzle, zero_loc)

def huristic(puzzle):
    # print(puzzle)
    result = 0
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] != 0:
                result = result + abs(int((puzzle[i][j]-1)/3) -i ) + abs((puzzle[i][j]-1)%3 -j)
            else:
                result = result + abs(int((8) / 3) - i) + abs((8) % 3 - j)
    return result

def find_best_hill(puzzle,zero_loc,seen):
    # print(seen)
    hn = huristic(puzzle)
    new_puzzle1 = copy(puzzle)
    new_zero1= copy_zero(zero_loc)
    new_puzzle2 = copy(puzzle)
    new_zero2 = copy_zero(zero_loc)
    new_puzzle3 = copy(puzzle)
    new_zero3 = copy_zero(zero_loc)
    new_puzzle4 = copy(puzzle)
    new_zero4 = copy_zero(zero_loc)
    new_puzzle1, new_zero1 = shuffle_right(new_puzzle1,new_zero1)
    # print(new_puzzle1, zero_loc1)
    # print(puzzle)
    new_puzzle2, new_zero2 = shuffle_left(new_puzzle2,new_zero2)
    # print(new_puzzle2, zero_loc2)
    new_puzzle3, new_zero3 = shuffle_down(new_puzzle3,new_zero3)
    # print(new_puzzle3, zero_loc3)
    new_puzzle4, new_zero4 = shuffle_up(new_puzzle4,new_zero4)
    # print(new_puzzle4, zero_loc4)
    h = [0 for i in range(4)]
    h[0] = huristic(new_puzzle1)
    h[1] = huristic(new_puzzle2)
    h[2] = huristic(new_puzzle3)
    h[3] = huristic(new_puzzle4)
    h1 =[]
    for i in range(4):
        h1.append((i,h[i]))
    # print(h1)
    h1.sort(key = sortSecond)
    # print(h1)
    ii = 0
    while ii < 4:
        if h1[ii][1]<= hn:
            numberr = 0
            j = ii
            while j < 4 and h1[ii][1] == h1[j][1] :
                numberr += 1
                j += 1
            w = randint(0, numberr-1)
            for i in range(numberr):
                if h1[((w+i)%numberr)+ii][0] == 0:
                    r_puzzle =copy(new_puzzle1)
                    r_loc = copy_zero(new_zero1)
                if h1[((w+i)%numberr)+ii][0] == 1:
                    r_puzzle = copy(new_puzzle2)
                    r_loc = copy_zero(new_zero2)
                if h1[((w+i)%numberr)+ii][0] == 2:
                    r_puzzle = copy(new_puzzle3)
                    r_loc = copy_zero(new_zero3)
                if h1[((w+i)%numberr)+ii][0] == 3:
                    r_puzzle =copy(new_puzzle4)
                    r_loc = copy_zero(new_zero4)

                if r_puzzle not in seen:
                    seen.append(r_puzzle)
                    return r_puzzle,r_loc
            ii += numberr
        else:
            break


        # print(ii)
    return None

def sortSecond(val):
    return val[1]


def hill_climbing(puzzle,zero_loc,lastState):
    seen = []
    seen.append(puzzle)
    iteration = 300
    for i in range(iteration):
        # print("iteratiiion ------->   ",i)
        if puzzle == lastState:
            return True, i+1
        else:
            b = find_best_hill(puzzle,zero_loc,seen)
            if b == None:
                # print("erooooooor")
                return False, None
            else:
                for i in range(2):
                    puzzle = b[0]
                    zero_loc = b[1]
    return False, None

def stochastic_hill_climbing(puzzle,zero_loc,lastState):
    seen = []
    seen.append(puzzle)
    iteration = 300
    for i in range(iteration):
        r = uniform(0, 1)
        # ii =randint(10,20)+i
        # print("iteratiiion ------->   ", i)
        if puzzle == lastState:
            return True, i + 1
        else:
            if r > 0.7:
                mabda = copy(puzzle)
                m_z = copy_zero(zero_loc)
                # print(seen)
                # print(mabda)
                f = 0
                w = randint(0, 3)
                for j in range(4):
                    if (w+j)%4 == 0:
                        puzzle, zero_loc = shuffle_right(puzzle, zero_loc)
                        # print(puzzle)
                        # if puzzle not in seen:
                        seen.append(puzzle)
                            # f = 1
                        break
                        # else:
                        #     puzzle = copy(mabda)
                        #     zero_loc = copy_zero(m_z)
                    elif (w+j)%4 == 1:
                        puzzle, zero_loc = shuffle_left(puzzle, zero_loc)
                        # print(puzzle)
                        # if puzzle not in seen:
                        seen.append(puzzle)
                            # f =1
                        break
                        # else:
                        #     puzzle = copy(mabda)
                        #     zero_loc = copy_zero(m_z)
                    elif (w+j)%4 == 2:
                        puzzle, zero_loc = shuffle_up(puzzle, zero_loc)
                        # print(puzzle)
                        # if puzzle not in seen:
                        seen.append(puzzle)
                            # f =1
                        break
                        # else:
                        #     puzzle = copy(mabda)
                        #     zero_loc = copy_zero(m_z)
                    elif (w+j)%4 == 3:
                        puzzle, zero_loc = shuffle_down(puzzle, zero_loc)
                        # print(puzzle)
                        # if puzzle not in seen:
                        seen.append(puzzle)
                            # f =1
                        break
                        # else:
                        #     puzzle = copy(mabda)
                        #     zero_loc = copy_zero(m_z)
                # if not f:
                #     return False, 300

            else:
                b = find_best_hill(puzzle, zero_loc, seen)
                if b == None:
                    pass
                    # print("erooooooor")
                else:
                    for i in range(2):
                        puzzle = b[0]
                        zero_loc = b[1]
    return False, 300


# print(puzzle, zero_loc)
# print(puzzle)
success = 0
number_of_moves = 0
success_s = 0
number_of_moves_s = 0
success2 = 0
number_of_moves2 = 0
success_s2 = 0
number_of_moves_s2 = 0
# b, n = hill_climbing(puzzle, zero_loc, last)
# print(b,n)
for i in range(1000):
    puzzle = [[1,2,3],[4,5,6],[7,8,0]]
    last = [[1,2,3],[4,5,6],[7,8,0]]
    zero_loc= [2,2]
    puzzle, zero_loc = re_shuffle(10,puzzle,zero_loc)
    b, n =hill_climbing(puzzle, zero_loc, last)
    if b:
        success += 1
        number_of_moves += n

for i in range(1000):
    puzzle = [[1,2,3],[4,5,6],[7,8,0]]
    last = [[1,2,3],[4,5,6],[7,8,0]]
    zero_loc= [2,2]
    puzzle, zero_loc = re_shuffle(10,puzzle,zero_loc)
    b, n =stochastic_hill_climbing(puzzle, zero_loc, last)
    if b:
        success_s += 1
        number_of_moves_s += n

for i in range(1000):
    puzzle = [[1,2,3],[4,5,6],[7,8,0]]
    last = [[1,2,3],[4,5,6],[7,8,0]]
    zero_loc= [2,2]
    puzzle, zero_loc = re_shuffle(20,puzzle,zero_loc)
    b, n =hill_climbing(puzzle, zero_loc, last)
    if b:
        success2 += 1
        number_of_moves2 += n

for i in range(1000):
    puzzle = [[1,2,3],[4,5,6],[7,8,0]]
    last = [[1,2,3],[4,5,6],[7,8,0]]
    zero_loc= [2,2]
    puzzle, zero_loc = re_shuffle(20,puzzle,zero_loc)
    b, n =stochastic_hill_climbing(puzzle, zero_loc, last)
    if b:
        success_s2 += 1
        number_of_moves_s2 += n
print("hill climbing with 10 shuffle:")
print("moves:",number_of_moves/success,"success:", success/10)
print("stochastic hill climbing with 10 shuffle:")
print("moves:",number_of_moves_s/success_s,"success:", success_s/10)
print("stochastic hill climbing with 20 shuffle:")
print("moves:",number_of_moves2/success2,"success:", success2/10)
print("stochastic hill climbing with 20 shuffle:")
print("moves:",number_of_moves_s2/success_s2,"success:", success_s2/10)
# print(puzzle)
