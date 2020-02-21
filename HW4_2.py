import sys
import random

random.seed(43)


class BayesNet:
    """
    variables: dictionary mapping letter (ex: 'A') to a Variable object
    letters: list of letters in topological order
    query: Query object representing the query variable and evidence variables
    """

    def __init__(self, filepath):
        self.variables = {}
        self.letters = []
        self.query = None
        self.n = 0

        with open(filepath) as reader:
            self.letters = reader.readline().rstrip().split(' ')
            print(self.letters)
            for letter in self.letters:
                self.variables[letter] = Variable(letter)

            # Read conditional probabilities until we hit an empty line
            for line in iter(reader.readline, '\n'):
                components = line.rstrip().split(' ')
                self.process_components(components)

            # Finally, read the query
            line = reader.readline()
            query_components = line.rstrip().split(' ')
            dep_var = query_components.pop(0)
            self.query = Query(dep_var, query_components)
            print(self.query.variable,self.query.evidence)
            for i in self.query.evidence:
                print(i)

    def process_components(self, components):
        """
        Updates Bayes Net probabilities and structure.
        components : list describing a conditional probability from the Bayes Net
                     (ex: ['+A', '-B', 'C', '0.5'] -> P(+C | +A, -A) = 0.5)

        """
        print(components)
        probability = float(components.pop())
        dep_var = components.pop()
        parents = [s[1] for s in components]
        print(parents)
        values = tuple((s[0] == "+") for s in components)
        print(values)

        if len(parents) > 0:
            self.variables[dep_var].parents = parents
            self.add_children(dep_var, parents)
            self.variables[dep_var].distribution[values] = probability
        else:
            self.variables[dep_var].probability = probability

    def add_children(self, dep_var, parents):
        """
        Add dep_var to the children list of each parent
        """
        for parent in parents:
            if dep_var not in self.variables[parent].children:
                self.variables[parent].children.append(dep_var)

    def sample(self, probability):
        """
        Use this function when generating random assignments.
        You should not be generating random numbers elsewhere.
        """
        return random.uniform(0, 1) < probability

    def direct_sample(self, trials):
        """
        Example of a direct sampling implementation. Ignores evidence variables.
        You do not need to edit this.
        """
        for i in range(4):
            print(self.variables[self.letters[i]].letter,self.variables[self.letters[i]].distribution,self.variables[self.letters[i]].parents, self.variables[self.letters[i]].children,self.variables[self.letters[i]].probability )
        count = 0
        flag = 1
        total_count = 0
        for i in range(trials):
            values = {}
            for letter in self.letters:
                prob = self.variables[letter].get_prob(values)
                values[letter] = self.sample(prob)
            if values[self.query.variable]:
                count += 1
        return float(count) / trials

    def rejection_sample(self, trials):
        """
        Implement this!
        Returns the estimated probability of the query.
        """

        count = 0
        i = 0
        while i < trials:
            flag = 1
            values = {}
            for letter in self.letters:
                prob = self.variables[letter].get_prob(values)
                values[letter] = self.sample(prob)
            for j in self.query.evidence:
                if values[j] != self.query.evidence[j]:
                    i = i-1
                    flag = 0
                    break
            if flag:
                if values[self.query.variable]:
                    count += 1
            i += 1
        return float(count) / trials

    def likelihood_sample(self, trials):
        """
        Implement this!
        Returns the estimated probability of the query.
        """
        count = 0
        total_count = 0

        for i in range(trials):
            values = {}
            w = 1
            for letter in self.letters:
                prob = self.variables[letter].get_prob(values)
                if letter not in self.query.evidence:
                    values[letter] = self.sample(prob)
                else:
                    values[letter] = self.query.evidence[letter]
                    w = w*prob
            total_count += w
            if values[self.query.variable]:
                count += 1*w
        return float(count)/total_count


    def gibbs_sample(self, trials):
        """
        Implement this!
        Returns the estimated probability of the query.
        """
        for i in range(4):
            print(self.variables[self.letters[i]].letter,self.variables[self.letters[i]].distribution,self.variables[self.letters[i]].parents, self.variables[self.letters[i]].children,self.variables[self.letters[i]].probability )
        count = 0
        values = {}
        for letter in self.letters:
            values[letter] = self.sample(random.uniform(0,1))
        for letter in self.query.evidence:
            values[letter] = self.query.evidence[letter]
        for i in range(trials):
            for letter in self.letters:
                if letter not in self.query.evidence:
                    if values[letter]:
                        pTrue = self.variables[letter].get_prob(values)
                    else:
                        pTrue = 1-self.variables[letter].get_prob(values)
                    p1 = pTrue
                    for j in self.variables[letter].children:
                        pTrue = pTrue * (self.variables[j].get_prob(values))

                    pFalse = 1 - p1
                    temp_values = {}
                    for i in values:
                        temp_values[i] = values[i]
                    temp_values[letter] = not(values[letter])
                    for j in self.variables[letter].children:
                        pFalse = pFalse * (self.variables[j].get_prob(temp_values))
                    if((pTrue + pFalse) == 0):
                        prob = 0.0
                    else:
                        if values[letter]:
                            prob = float(pTrue) / (pTrue + pFalse)
                        else:
                            prob = float(pFalse) / (pTrue + pFalse)
                    values[letter] = self.sample(prob)
            if values[self.query.variable]:
                count += 1
        return float(count) / trials




class Variable:
    """
    letter: the letter (ex: 'A')
    distribution: dictionary mapping ordered values of parents to float probabilities,
                  ex: (True, True, False) -> 0.5
    parents: list of parents (ex: ['C', 'D'])
    children: list of children (ex: ['E'])
    probability: probability of node if the node has no parents
    """

    def __init__(self, letter):
        self.letter = letter
        self.distribution = {}  # Maps values of parents (ex: True, True, False) to
        self.parents = []
        self.children = []
        self.probability = 0.0  # Only for variables with no parents

    def get_prob(self, values):
        if len(self.parents) == 0:
            return self.probability
        else:
            key = tuple([values[letter] for letter in self.parents])
            return self.distribution[key]


class Query:
    """
    self.variable: the dependent variable associated with the query
    self.evidence: mapping from evidence variables to True or False
    """

    def __init__(self, variable, evidence):
        self.variable = variable
        self.evidence = {}
        for s in evidence:  # ex: "+B" or "-C"
            self.evidence[s[1]] = (s[0] == "+")


if __name__ == '__main__':
    filename = sys.argv[1]
    trials_num = int(sys.argv[2])
    sampling_type = int(sys.argv[3])
    bayes_net = BayesNet(filename)

    if sampling_type == 0:
        print(bayes_net.direct_sample(trials_num))
    elif sampling_type == 1:
        print(bayes_net.rejection_sample(trials_num))
    elif sampling_type == 2:
        print(bayes_net.likelihood_sample(trials_num))
    elif sampling_type == 3:
        print(bayes_net.gibbs_sample(trials_num))
