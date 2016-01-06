import random
import math
import sys
import operator
import os.path

words = {}
pos = ['noun', 'verb', 'det', 'adv', 'adj', 'adp', 'num', 'pron', 'prt', 'x', '.', 'conj']
pos_probabilities = {x: {x: 0 for x in pos+['start']} for x in pos}
occurance = {}
pos_occurance = {x: 0 for x in pos+['count']}
test_samples = []
weights = {}
edges = {}
average_count = 0
MCMC_SAMPLES = 5
transitions = {}

class Solver:

    def train(self, data):
        for idx, i in enumerate(data):
            sys.stdout.write(" Training: %f %% \r" % (float(idx*100)/float(len(data))))
            sys.stdout.flush()
            sentence = {x: i[1][y] for y,x in enumerate(i[0])}
            prev_pos = None
            for word, part in sentence.iteritems():
                if word in words:
                    occurance[word] += 1
                    words[word][part] += 1
                else:
                    occurance[word] = 1
                    words[word] = {x: 0 for x in pos}
                    words[word][part] = 1
                pos_occurance[part] += 1
                pos_occurance['count'] += 1
                if prev_pos:
                    pos_probabilities[part][prev_pos] += 1
                else:
                    pos_probabilities[part]['start'] += 1
                prev_pos = part
                self.perceptron_train(word, part)
            self.average_perceptron(sentence.iterkeys())
        pass

    def perceptron_train(self, word, part):
        global average_count
        if word not in weights:
            weights[word] = {x: 0 for x in pos}
            edges[word] = {x: 0 for x in pos}
            edges[word][part] = 1
        else:
            guess = self.predict(word,True)
            if guess != part:
                edges[word][part] += 1
                edges[word][guess] -= 1
        average_count += 1

    def average_perceptron(self, sentence):
        global average_count
        for word in sentence:
            for part in pos:
                weights[word][part] = float(average_count*weights[word][part]+edges[word][part])/float(average_count+1)
        average_count += 1

    def predict(self, word, train=False):
        if train == True:
            current_weight = edges[word]
        else:
            if word not in weights:
                return "reserve"
            current_weight = weights[word]
        return max(current_weight.iterkeys(), key=(lambda key: current_weight[key]))

    def resolve_reserves(self, result, best):
        if best == False:
            return 0
        for i in range(0,len(result)):
            if result[i] == "reserve":
                if i+1 <= len(result) or (i+1 <= len(result) and result[i+1] == "reserve"):
                    if result[i-1] == 'reserve':
                        result[i] = max([[pos_probabilities[x]["noun"],x] for x in pos])[1]
                    else:
                        result[i] = max([[pos_probabilities[x][result[i-1]],x] for x in pos])[1]
                elif i == 0:
                    result[i] = max([[pos_probabilities[x][result[i+1]]*pos_probabilities[x]['start'],x] for x in pos])[1]
                else:
                    result[i] = max([[pos_probabilities[result[i+1]][x]*pos_probabilities[x][result[i-1]],x] for x in pos])[1]

    def naive(self, sentence, best=False):
        result = []
        for word in sentence:
            if word in words.keys():
                result.append(max(words[word].iterkeys(), key=(lambda key: words[word][key])))
            else:
                subs = self.resolve_by_substring(word)
                if subs != None:
                    result.append(subs)
                else:
                    result.append('noun')
            if best == True and len(set(words[word].values())) == 1:
                result[-1] = 'reserve'
        self.resolve_reserves(result,best)
        return result

    def mcmc(self, sentence, sample_count):
        for test_number in range(0,100):
            sample = [self.random_toss(self.compute_buff(sentence[0]))]
            for word in sentence[1:]:
                sample.append(self.random_toss(self.compute_buff(word,sample[-1])))
            test_samples.append(sample)
        return test_samples[100-sample_count:]

    def best(self, sentence):
        result = []
        for word in sentence:
            result.append(self.predict(word))
        self.resolve_reserves(result,True)
        return result

    def max_marginal(self, sentence):
        result = []
        probs = []
        i=0
        global test_samples
        for i in range(0,len(sentence)):
            sample_probs = {x: len([y[i] for y in test_samples if y[i] == x]) for x in pos}
            result.append(max(sample_probs.iterkeys(), key=(lambda key: sample_probs[key])))
            probs.append(float(sample_probs[result[-1]])/100.0)
        test_samples = []
        return result

    def viterbi(self, sentence):
        transitions = self.generate_transitions()
        probs = []
        prev_builds = []
        start_prob = {part: float(pos_probabilities[part]['start'])/float(pos_occurance[part]) for part in pos}
        emissions = self.generate_emissions(sentence[0]) if sentence[0] in words else {x: float(1.0) for x in pos}
        probs.append({v: float(emissions[v])*float(start_prob[v]) for v in pos})
        prev_state = probs[0]
        result = ['noun']
        for word in sentence[1:]:
            emissions = self.generate_emissions(word) if word in words else {x: float(0.00000078) for x in pos}
            prev_builds.append({part: {k: float(prev_state[k])*float(transitions[part][k]) for k in pos} for part in pos})
            probs.append({v: emissions[v]*max(prev_builds[-1][v].itervalues()) for v in pos})
            new_result = max(probs[-1].iterkeys(), key=(lambda key: probs[-1][key]))
            result.append(new_result)
            prev_state = probs[-1]
        for idx in range(len(result)-1, 0, -1):
            result[idx-1] = max(prev_builds[idx-1][result[idx]].iterkeys(), key=(lambda key: prev_builds[idx-1][result[idx]][key]))
        for i in range(0,len(sentence)):
            if len(set(words[sentence[i]].values())) == 1:
                result[i] == "reserve"
        self.resolve_reserves(result,True)
        return result

    def generate_transitions(self):
        return {idx: {y: float(x)/float(pos_occurance[idx]-pos_probabilities[idx]['start']) for y,x in val.iteritems()} for idx, val in pos_probabilities.iteritems()}

    def generate_emissions(self, word):
        return {part: float(words[word][part])/float(occurance[word]) for part in pos}

    def compute_buff(self, word, prev=None):
        res = 0.0
        result = {}
        self.create_or_order(word)
        for x in words[word].keys():
            if prev is None:
                result[x] = float(words[word][x])/float(occurance[word])
            else:
                result[x] = (float(words[word][x])/float(occurance[word]))*(float(pos_probabilities[x][prev])/float(pos_occurance[x]))
        result = {x: (result[x]/sum([result[y] for y in result])) for x in result}
        return dict(sorted(result.items(), key=lambda i:float(i[1])))

    def create_or_order(self, word):
        if word not in words:
            words[word] = {x: float(1)/float(12) for x in pos}
            occurance[word] = 1

    def random_toss(self, bias):
        toss = random.random()
        for idx,val in bias.iteritems():
            if toss < val:
                return idx
        return max(bias.iterkeys(), key=(lambda key: bias[key]))

    def resolve_by_substring(self, word):
        full_diff = 0
        result = None
        for train_word in words.keys():
            diff = abs(len(word)-len(train_word))+len(set(list(train_word))-set(list(word)))
            if diff < 5:
                if diff < full_diff:
                    result = max(words[train_word].iterkeys(), key=(lambda key: words[train_word][key]))
        return result

    def display_sentence(self, arr):
        return ' '.join(arr)

    def solve(self, sentence):
        print "Naive Bayes"
        print self.display_sentence(self.naive(sentence))
        print "\nGibbs Sampling"
        mcmc_result = self.mcmc(sentence, MCMC_SAMPLES)
        for i in range(0,MCMC_SAMPLES):
            print self.display_sentence(mcmc_result[i])
        print "\nMax Marginal inference"
        print self.display_sentence(self.max_marginal(sentence))
        print "\nViterbi"
        print self.display_sentence(self.viterbi(sentence))
        print "\nAveraged Perceptron"
        print self.display_sentence(self.best(sentence))
        print "\n"

def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    return exemplars

def main():
    solver = Solver()
    train_data = read_data("train_file.pos")
    solver.train(train_data)
    while(True):
        response = raw_input("Enter the Sentence (Enter X to exit): ")
        if response == "X":
            break
        else:
            sentence = response.split(" ")
        solver.solve(sentence)

if __name__ == "__main__": main()