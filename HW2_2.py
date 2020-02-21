import random


n,m = map(int,input().split(" "))
pop_size = n*6
interfer = []
for i in range(m):
    interfer.append(list(map(int,input().split(" "))))
def sortFit(val):
    return -(val.fitness)

class people():
    def __init__(self,person):
        self.person = person
        self.fitness,self.days = self.cal_fitness()
        self.fitness_per = 0

    def cal_fitness(self):
        result = 0
        inters = 1
        flag = 0
        days = [0]
        for i in range(1,n):
            for j in range(1,inters+1):
                if [self.person[i],self.person[i-j]] in interfer or [self.person[i-j],self.person[i]] in interfer:
                    days.append(i)
                    result += 1
                    inters = 1
                    flag = 1
                    break
            if flag == 0:
                inters += 1
            else:
                flag = 0
        return n-result-1,days

def copy_person(person):
    new_ppp = []
    for i in range(len(person)):
        new_ppp.append(person[i])
    return new_ppp

def selection(population):
    sum = 0
    for i in range(pop_size):
        sum = sum + population[i].fitness
    for i in range(1,pop_size):
        population[i].fitness += population[i-1].fitness
    for i in range(1,pop_size):
        population[i].fitness_per += population[i-1].fitness/sum
    new_population = []

    for i in range(pop_size):
        selected = pop_size-1
        rand = random.uniform(0, 1)
        for j in range(1,pop_size):
            if rand < population[j].fitness_per:
                selected = j - 1
                break
        new_ozv = people(copy_person(population[selected].person))
        new_population.append(new_ozv)
    # for i in new_population:
    #     print(i.person)


    new_population.sort(key = sortFit)
    return new_population

def cross_over(new_population):
    ii = 0
    childeren = []
    while ii < pop_size:
        mother = new_population[ii]
        father = new_population[ii+1]
        # print(ii,"mother", mother.person,mother.days)
        # print(ii+1,"father",father.person,father.days)

        if len(mother.days)>1:
            rand= (random.randint(1,len(mother.days)-1))
            # print(rand)
            child1 = []
            for i in range(mother.days[rand]):
                child1.append(mother.person[i])
            for i in range(n):
                if father.person[i] not in child1:
                    child1.append(father.person[i])

            childeren.append(people(child1))
        else:
            childeren.append(mother)
        # print("child1", childeren[len(childeren)-1].person,childeren[len(childeren)-1].fitness)
        if len(father.days)>1:
            rand = (random.randint(1, len(father.days) - 1))
            # print(rand)
            child2 = []
            for i in range(father.days[rand],n):
                child2.append(father.person[i])
            for i in range(n-1,-1,-1):
                if mother.person[i] not in child2:
                    child2 = [mother.person[i]]+child2
            childeren.append(people(child2))
        else:
            childeren.append(father)
        # print("child2", childeren[len(childeren)-1].person,childeren[len(childeren)-1].fitness)
        ii += 2
    return childeren

def mutation(people):
    e1= random.randint(0,4)
    e2 = random.randint(0, 4)
    temp = people.person[e1]
    people.person[e1] = people.person[e2]
    people.person[e2] = temp

def copy_best(po):
    best_f = po.fitness
    best_person = []
    for i in range(n):
        best_person.append(po.person[i])
    best_days = []
    for i in range(len(po.days)):
        best_days.append(po.days[i])
    return best_f,best_person,best_days



population = []
for i in range(pop_size):
    p = [i for i in range(n)]
    random.shuffle(p)
    PP = people(p)
    population.append(PP)
population.sort(key = sortFit)
best_f ,best_person,best_days = copy_best(population[0])
# print("best_f------->", best_f)
# print(*[x.person for x in population], sep=" ")


for permiutation in range(1000):
    # print(permiutation)
    new_population = cross_over(selection(population))
    # print("newwww generatiooooooon")
    # for i in new_population:
    #     print(i.person , i.fitness)
        # print(selected)


    for i in new_population:
        e = random.uniform(0,1)
        if e < 0.2:
            # print("before mut", i.person)
            mutation(i)
            # print("after mut", i.person)
    population = new_population
    population.sort(key = sortFit)
    # print("newwww generatiooooooon2")
    # for i in new_population:
    #     print(i.person , i.fitness)
    # print("best_f------->", population[0].fitness)
    if population[0].fitness> best_f:
        # print("bettter better")
        best_f, best_person, best_days = copy_best(population[0])


# new_population = cross_over(population)
# print(*[x.person for x in new_population], sep=" ")
# e = random.uniform(0,1)
# for i in new_population:
#     if e < 0.1:
#         print("before mut", i.person)
        # mutation(i)
        # print("after mut", i.person)
# population = new_population
# population = sorted(population, key = lambda x : -x.fitness)
# if population[0].fitness> best_f:
#     best_f, best_person, best_days = copy_best(population[0])


print(n-best_f)
ppppp = [[]for i in range(n-best_f)]
j = 0
for i in range(len(best_person)):
    if i in best_days and i!=0:
        j+= 1
    ppppp[j].append(best_person[i])
for i in range(len(ppppp)):
    print(*[x for x in ppppp[i]], sep=" ")
# print(ppppp)

