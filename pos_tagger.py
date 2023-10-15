from importlib.machinery import all_suffixes
from multiprocessing import Pool
from re import S
import numpy as np
import time
from utils import *
import math



""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        #print(k,'/',n)
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        # TODO
        # Laplace smoothing
        self.k = 0.002

        # TODO 
        # Linear interpolation
        self.lambda_1 = 0
        self.lambda_2 = 0
        self.lambda_3 = 0

    def linear_interpolation(self):
        # TODO
        self.lambda_1 = 0
        self.lambda_2 = 0
        self.lambda_3 = 0

        # Formatted after algorithm covered in Brants' paer
        for tag_1 in range(self.num_tags):
            for tag_2 in range(self.num_tags):
                for tag_3 in range(self.num_tags):
                    if self.trigrams_count[tag_1, tag_2, tag_3] > 0:
                        case_1 = 0 if (self.bigrams_count[tag_1, tag_2] - 1) == 0 else (self.trigrams_count[tag_1, tag_2, tag_3] - 1) / (self.bigrams_count[tag_1, tag_2] - 1)
                        case_2 = 0 if (self.unigrams_count[tag_2] - 1) == 0 else (self.bigrams_count[tag_1, tag_2] - 1) / (self.unigrams_count[tag_2] - 1)
                        # TODO for case 3 is N correct (the denominator)
                        case_3 = 0 if np.sum(self.unigrams) == 0 else (self.unigrams_count[tag_3] - 1) / np.sum(self.unigrams)

                        max_val = max(case_1, case_2, case_3)

                        if max_val == case_1:
                            self.lambda_3 += self.trigrams_count[tag_1, tag_2, tag_3]
                        elif max_val == case_2:
                            self.lambda_2 += self.trigrams_count[tag_1, tag_2, tag_3]
                        else:
                            self.lambda_1 += self.trigrams_count[tag_1, tag_2, tag_3]


        # Normalization step
        sum = self.lambda_1 + self.lambda_2 + self.lambda_3
        self.lambda_1 /= sum
        self.lambda_2 /= sum
        self.lambda_3 /= sum
                        
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        self.unigrams = np.zeros(len(self.all_tags))
        self.unigrams_count = np.zeros(len(self.all_tags))
        tag_sequences = self.data[1]
        for tag_sequence in tag_sequences:  
            for tag in tag_sequence:
                self.unigrams[self.tag2idx[tag]] += 1
                self.unigrams_count[self.tag2idx[tag]] += 1
        # Smoothing
        self.unigrams += self.k
        self.unigrams /= (np.sum(self.unigrams) + self.num_tags * self.k)

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        self.bigrams_count = np.zeros((len(self.all_tags), len(self.all_tags)))

        self.bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        tag_sequences = self.data[1]
        for tag_sequence in tag_sequences:
            for i in range(len(tag_sequence) - 1):
                tag1_index = self.tag2idx[tag_sequence[i]]
                tag2_index = self.tag2idx[tag_sequence[i + 1]]
                self.bigrams[tag1_index, tag2_index] += 1
                self.bigrams_count[tag1_index, tag2_index] += 1
        # Smoothing
        for tag1 in range(len(self.bigrams)):
            for tag2 in range(len(self.bigrams)):
                self.bigrams[tag1, tag2] = (self.bigrams[tag1, tag2] + self.k) / (self.data[1].count(tag1) + self.k * self.num_tags)
                # self.bigrams[tag1, tag2] = (self.bigrams[tag1, tag2] + (self.k / len(self.all_tags))) / (self.data[1].count(tag1) + self.k)
        # self.bigrams[self.tag2idx['<STOP>'], self.tag2idx['O']] = 0
        # print(np.matrix(self.bigrams))

        # Smoothing
        # self.bigrams += self.k
        # self.bigrams /= (np.sum(self.unigrams) + self.num_tags * self.k)

        # Log space
        # self.bigrams = np.log(self.bigrams)
        # print(self.bigrams)
    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        self.trigrams_count = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        self.trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        tag_sequences = self.data[1]
        for tag_sequence in tag_sequences:
            for i in range(len(tag_sequence) - 2):
                tag1_index = self.tag2idx[tag_sequence[i]]
                tag2_index = self.tag2idx[tag_sequence[i + 1]]
                tag3_index = self.tag2idx[tag_sequence[i + 2]]
                self.trigrams[tag1_index, tag2_index, tag3_index] += 1
                self.trigrams_count[tag1_index, tag2_index, tag3_index] += 1
        for tag1 in range(len(self.trigrams)):
            for tag2 in range(len(self.trigrams)):
                for tag3 in range(len(self.trigrams)):
                    self.trigrams[tag1, tag2, tag3] = (self.trigrams[tag1, tag2, tag3] + self.k) / (self.bigrams_count[tag1, tag2] + self.k * self.num_tags)
        # self.trigrams = np.log(self.trigrams)
        # print(self.trigrams)
    
    def get_fourgrams(self):
        self.fourgrams_count = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        self.fourgrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        tag_sequences = self.data[1]
        for tag_sequence in tag_sequences:
            for i in range(len(tag_sequence) - 3):
                tag1_index = self.tag2idx[tag_sequence[i]]
                tag2_index = self.tag2idx[tag_sequence[i + 1]]
                tag3_index = self.tag2idx[tag_sequence[i + 2]]
                tag4_index = self.tag2idx[tag_sequence[i + 3]]
                self.fourgrams[tag1_index, tag2_index, tag3_index, tag4_index] += 1
                self.fourgrams_count[tag1_index, tag2_index, tag3_index, tag4_index] += 1
        for tag1 in range(self.num_tags):
            for tag2 in range(self.num_tags):
                for tag3 in range(self.num_tags):
                    for tag4 in range(self.num_tags):
                        self.fourgrams[tag1, tag2, tag3, tag4] = (self.fourgrams[tag1, tag2, tag3, tag4] + self.k) / (self.trigrams_count[tag1, tag2, tag3] + self.k * self.num_tags)
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        self.lexical = np.zeros((len(self.all_tags), len(self.vocab)))
        self.tag_count = np.zeros(len(self.all_tags))
        for i, words in enumerate(self.data[0]):
            tags = self.data[1][i]
            for j, word in enumerate(words):
                tag = tags[j]
                self.tag_count[self.tag2idx[tag]] += 1
                self.lexical[self.tag2idx[tag], self.vocab2idx[word]] += 1
                # print(self.lexical[self.tag2idx[tag], self.vocab2idx[word]])
        # for tag in self.all_tags:
        #     # TODO have another loop for words
        #     self.lexical[self.tag2idx[tag], _]
        tag_sums = self.lexical.sum(axis = 1)
        self.lexical = (self.lexical + self.k) / (tag_sums[:, np.newaxis] + self.k)
        # print(self.lexical)

#other models besides beam search and viterbi, get full score just with beam search and viterbi

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        #train_lower = [[string.lower() for string in sublist] for sublist in data[0]]
        #data = (train_lower, data[1])

        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}

        ## TODO
        self.vocab = list(set(w for word in data[0] for w in word))
        self.vocab2idx = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.idx2vocab = {v:k for k,v in self.vocab2idx.items()}
        self.num_tags = len(self.all_tags)


        # TODO for evaluate
        self.word2idx = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.idx2word = {v:k for k,v in self.vocab2idx.items()}

        self.get_unigrams()
        self.get_bigrams()
        self.get_trigrams()

        self.get_emissions()

        self.get_fourgrams()

        self.get_unknown_word_emissions()


        # TODO
        self.unigrams = np.log(self.unigrams)
        self.bigrams = np.log(self.bigrams)
        self.trigrams = np.log(self.trigrams)
        self.lexical = np.log(self.lexical)
        self.fourgrams = np.log(self.fourgrams)
        
        # TODO temp test
        #sequence = ['The', 'dog', 'ran', 'on', 'the', 'street']

        #self.beam_trigram(sequence, k=5)
        #print('viterbi time')
        #self.viterbi_bigram(sequence)
        #self.viterbi_bigram_optimized(sequence)
        #self.viterbi_trigram(sequence)
        # print(self.trigrams)
        # print(self.num_tags)


        # # self.set_lambda_parameters()
        # self.viterbi_trigrams()

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        ## TODO
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        ## TODO
        #beam_solution = self.beam_trigram(sequence, k = 5)[0]
        #beam_solution = self.viterbi_bigram_optimized(sequence)
        #beam_solution = self.viterbi_trigram_optimized(sequence)
        beam_solution = self.viterbi_trigram(sequence)
        #beam_solution = self.viterbi_bigram(sequence)

        return beam_solution

    def get_unknown_word_emissions(self):
        noun_suffixes = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 
                         'ty', 'ment', 'ness', 'ship', 'ion',  'ant', 'ent','ee','or',
                         'age', 'hood','ry', 'eer']
        verb_suffixes = ['ate', 'en', 'ify', 'fy', 'ize', 'ise', 'ing', 'ed']
        adj_suffixes = ['able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious',  'ish', 
                        'ive', 'less', 'y', 'en', 'ese', 'i', 'ian','ly','ous', 'ant']
        adv_suffixes = ['ward','wards','wise']
        random_suffixes = ['ss','lm','ln','ash','esh','ush','osh']
        all_suffix = ['al','ance','ence','ation','etion','ition','otion','ution',
                      'sion','ure','age','ing','ery','er','ist','ist','ity','ment',
                      'ness','or','ship','th','ty','hood','able','ible','al','ant','ary',
                      'ful','ic','ious','ous','ive','less','y','ical','ish','like',
                      'ed','en','er','ing','ize','ise','ify','fy','ate','act','ly',
                      'ward','wise']
        
        #building matrix
        self.all_suffixes = list(set(all_suffix + noun_suffixes + verb_suffixes + adj_suffixes + adv_suffixes + random_suffixes))
        #dataframe with tag as rows, cols as suffixes
        #final row is total
        #go through the training data, if word ends with any of these, +1 to the tag 
        #and +1 to the total
        suf_num = len(self.all_suffixes)
        self.unknown_words = np.zeros((suf_num, self.num_tags)) 
        #^such that pi_vales[suf_num, tagidx] = prob of that suf being tagidx
        for i, words in enumerate(self.data[0]):
            tags = self.data[1][i] #multiple tags?
            for j, word in enumerate(words):
                tag = tags[j]
                idx_tag = self.tag2idx[tag]
                for suf, suffix in enumerate(self.all_suffixes):
                    len_suff = len(self.all_suffixes[suf])
                    len_word = len(word)
                    if word[len_word-len_suff:] == self.all_suffixes[suf]:
                        self.unknown_words[suf, idx_tag] += 1
        
        #make it proportional to the suffix
        #take row sums, divide every element by the row 
        row_sums = self.unknown_words.sum(axis=1) #row sums, aka suffix sums
        for s, suff in enumerate(self.all_suffixes):
            rsum = row_sums[s]
            for t in range(self.num_tags):
                curr = self.unknown_words[s, t]
                if rsum > 0:
                    self.unknown_words[s, t] = curr/rsum

    def handle_unknown_word(self, word):
        #check to see if it has a suffix in self.all_suffixes
        word_len = len(word)
        for s, suff in enumerate(self.all_suffixes):
            suf_len = len(suff)
            if word[word_len-suf_len:] == suff:
                tag_probs = self.unknown_words[s, :]
                tag_chosen_id = np.argmax(tag_probs)
                #print('match!', word, suff, self.idx2tag[tag_chosen_id])
                return tag_chosen_id
        return self.tag2idx['NN']
    
    def handle_unknown_word_tag(self, word, tag_idx):
        #check to see if it has a suffix in self.all_suffixes
        word_len = len(word)
        for s, suff in enumerate(self.all_suffixes):
            suf_len = len(suff)
            if word[word_len-suf_len:] == suff:
                tag_probs = self.unknown_words[s, tag_idx]
                #print('match!', word, suff, self.idx2tag[tag_chosen_id])
                return tag_probs        
        return self.unigrams[self.tag2idx['NN']]
            
        #if yes, pull out argmax tag
        #if no, return noun

    def beam_fourgram(self, sequence, k = 1):
        # Formatted: tag_1, tag_2, tag_3, probability, sequence
        # Make sure everything is formatted correctly
        best_paths = [(self.tag2idx['O'], self.tag2idx['O'], self.tag2idx['O'], 0, [])]

        for word in sequence:
            paths = []
            for (tag_1, tag_2, tag_3, prob, best_seq_so_far) in best_paths:
                for tag in self.all_tags:
                    next_sequence = best_seq_so_far[:]
                    next_tag = self.tag2idx[tag]
                    # TODO handle unknown word else condition
                    if not word in self.vocab2idx:
                        emission_prob = self.handle_unknown_word_tag(word, next_tag)
                    else:
                        emission_prob = self.lexical[next_tag, self.vocab2idx[word]]
                    #emission_prob = self.lexical[next_tag, self.vocab2idx[word]] if word in self.vocab else self.unigrams[self.tag2idx['NN']]
                    next_prob = prob + self.fourgrams[tag_1, tag_2, tag_3, next_tag] + emission_prob
                    next_sequence.append(next_tag)
                    next_candidate = (tag_2, tag_3, next_tag, next_prob, next_sequence)
                    paths.append(next_candidate)
            # Sort and update top k paths
            # TODO sort second
            # paths.sort(key=self.sortSecond, reverse=True)
            paths.sort(key=lambda x: x[3], reverse=True)
            best_paths = paths[:k]

        # TODO stop
        # TODO edge case stop stop stop

        solution = []
        for path_combination in best_paths:
            tag_path = []
            for tag in path_combination[4]:
                tag_path.append(self.idx2tag[tag])
            solution.append(tag_path)

        return solution
    
    def beam_trigram(self, sequence, k = 1):
        # Formatted: tag_1, tag_2, probability, sequence
        # Make sure everything is formatted correctly
        best_paths = [(self.tag2idx['O'], self.tag2idx['O'], 0, [])]

        for word in sequence:
            paths = []
            for (tag_1, tag_2, prob, best_seq_so_far) in best_paths:
                for tag in self.all_tags:
                    next_sequence = best_seq_so_far[:]
                    next_tag = self.tag2idx[tag]
                    # TODO handle unknown word else condition
                    if not word in self.vocab2idx:
                        emission_prob = self.handle_unknown_word_tag(word, next_tag)
                    else:
                        emission_prob = self.lexical[next_tag, self.vocab2idx[word]]
                    #emission_prob = self.lexical[next_tag, self.vocab2idx[word]] if word in self.vocab else self.unigrams[self.tag2idx['NN']]
                    next_prob = prob + self.trigrams[tag_1, tag_2, next_tag] + emission_prob
                    next_sequence.append(next_tag)
                    next_candidate = (tag_2, next_tag, next_prob, next_sequence)
                    paths.append(next_candidate)
            # Sort and update top k paths
            # TODO sort second
            # paths.sort(key=self.sortSecond, reverse=True)
            paths.sort(key=lambda x: x[2], reverse=True)
            best_paths = paths[:k]

        # TODO stop
        # TODO edge case stop stop stop

        solution = []
        for path_combination in best_paths:
            tag_path = []
            for tag in path_combination[3]:
                tag_path.append(self.idx2tag[tag])
            solution.append(tag_path)
            print(tag_path)

        return solution
    
    def beam_bigram(self, sequence, k = 1):
        best_paths = [(self.tag2idx['O'], 0, [])]

        for word in sequence:
            paths = []
            for (tag_1, prob, best_seq_so_far) in best_paths:
                for tag in self.all_tags:
                    next_sequence = best_seq_so_far[:]
                    next_tag = self.tag2idx[tag]
                    # TODO handle unknown word else condition
                    #emission_prob = self.lexical[next_tag, self.vocab2idx[word]] if word in self.vocab else self.unigrams[self.tag2idx['NN']]
                    if not word in self.vocab2idx:
                        emission_prob = self.handle_unknown_word_tag(word, next_tag)
                    else:
                        emission_prob = self.lexical[next_tag, self.vocab2idx[word]]
                    next_prob = prob + self.bigrams[tag_1, next_tag] + emission_prob
                    next_sequence.append(next_tag)
                    next_candidate = (next_tag, next_prob, next_sequence)
                    paths.append(next_candidate)
            # Sort and update top k paths
            paths.sort(key=lambda x: x[1], reverse=True)
            best_paths = paths[:k]

        solution = []
        for path_combination in best_paths:
            tag_path = []
            for tag in path_combination[2]:
                tag_path.append(self.idx2tag[tag])
            solution.append(tag_path)
            print(tag_path)

        return solution

    def viterbi_trigram(self, sequence):
        self.pi_values = np.matrix(np.ones((len(self.all_tags) * len(self.all_tags), len(sequence) + 1)) * -math.inf)
        self.backpointers = np.matrix(np.ones((len(self.all_tags) * len(self.all_tags), len(sequence) + 1)) * -math.inf)

        start_tag_idx = self.tag2idx['O']
        for i in range(self.num_tags):
            self.pi_values[start_tag_idx * self.num_tags + i, 0] = 0
        
        # idx2pair = {}
        # for tag1 in range(self.num_tags):
        #     for tag2 in range(self.num_tags):
        #         idx = tag1 * self.num_tags + tag2
        #         idx2pair[idx] = (tag1, tag2)

        for k in range(1, len(sequence) + 1):
            #print('k=',k,'------------------------')
            for u in range(self.num_tags): #k-1
                for v in range(self.num_tags): #k
                    idx = []
                    tagxtag_idx = []
                    #only choosing the pairs in k-1 that end with u, saving indxs in idx
                    for iterating in range(self.num_tags):
                        val = iterating * self.num_tags + u # all (w,u) in k-1
                        idx.append(iterating) 
                        tagxtag_idx.append(val)
                    pi_prob = np.asarray(np.take(self.pi_values, tagxtag_idx, axis = 0)[:, k-1]).reshape(-1)
                    q_prob = self.trigrams[:, u, v]
                    lex_prob = None
                    if not sequence[k-1] in self.vocab2idx:
                        lex_prob = self.handle_unknown_word_tag(sequence[k-1], v)
                    else:
                        lex_prob = self.lexical[v, self.vocab2idx[sequence[k-1]]]
                    total_prob = np.add(pi_prob, q_prob)
                    total_prob = total_prob + lex_prob
                    max_p = np.max(total_prob)
                    argmax = np.argmax(total_prob)
                    self.pi_values[u * self.num_tags + v, k] = max_p
                    self.backpointers[u * self.num_tags + v, k] = argmax
        final_max = -math.inf
        final_argmax = None
        second_argmax = None
        for x in range(self.num_tags):
            for y in range(self.num_tags):
                last_backpointer_max = self.pi_values[x*self.num_tags+y, len(sequence)] + self.trigrams[x,y,self.tag2idx['.']]
                if (last_backpointer_max > final_max):
                    final_max = last_backpointer_max
                    final_argmax = y
                    second_argmax = x
        final_y = np.zeros(len(sequence))
        final_y[len(sequence) - 1] = final_argmax
        final_y[len(sequence) - 2] = second_argmax
        for k in range(len(sequence), 2, -1): #goes from n to 3
            #backpointers values should be 0, 1, ... n, leading to n+1 vals
            #this is bc the 0th one is the starting one
            # so at k=0 this should access backpointers [_,1] 
            idx = final_y[k-2]*self.num_tags+final_y[k-1]
            w = self.backpointers[int(idx),k]
            final_y[k-3] = w
        converter = lambda t: self.idx2tag[t]
        converter_func = np.vectorize(converter)
        final_tags = list(converter_func(final_y))
        #print(final_tags)
        return final_tags

    def viterbi_bigram_optimized(self, sequence):
        self.pi_values = np.matrix(np.ones((len(self.all_tags), len(sequence) + 1)) * -math.inf)  #+1 to acct for starting at 1 (0 is start)
        # TODO dtype
        self.backpointers = np.matrix(np.ones((len(self.all_tags), len(sequence) + 1)) * -math.inf)

        self.pi_values[self.tag2idx['O'], 0] = 0
        # Following algorithm described in Collin's notes
        for k in range(1, len(sequence) + 1): #k is the current word 1-6
            for u in range(self.num_tags):
                pi_val_prob = np.asarray(self.pi_values[:,k-1]).reshape(-1)
                big_prob = self.bigrams[:, u]
                e_prob = None
                if not sequence[k-1] in self.vocab2idx:
                    e_prob = self.handle_unknown_word_tag(sequence[k-1], u) 
                else:
                    e_prob = self.lexical[u, self.vocab2idx[sequence[k-1]]]
                if not isinstance(e_prob, float):
                    e_prob = e_prob.flatten()
                    e_prob = e_prob[0]
                total_prob = np.add(pi_val_prob, big_prob)
                total_prob = total_prob + e_prob
                max_p = np.max(total_prob)
                argmax = np.argmax(total_prob)
                self.pi_values[u, k] = max_p
                self.backpointers[u, k] = argmax
        final_max = -math.inf
        final_argmax = None
        for i in range(self.num_tags):
            last_backpointer_max = self.pi_values[i, len(sequence)] + self.bigrams[i,self.tag2idx['.']]
            if (last_backpointer_max > final_max):
                final_max = last_backpointer_max
                final_argmax = i
        final_y = np.zeros(len(sequence))
        final_y[len(sequence) - 1] = final_argmax
        for k in range(len(sequence) - 1, 0, -1): #goes from n-1 to 1 
            next_y = self.backpointers[int(final_y[k]),k+1]
            final_y[k-1] = next_y
        converter = lambda t: self.idx2tag[t]
        converter_func = np.vectorize(converter)
        final_tags = list(converter_func(final_y))
        #print(final_tags)
        return final_tags

if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    # print('length_train_documents',len(train_data[0]))
    # print('length_dev_documents',len(dev_data[0]))
    # print('length_test_documents',len(test_data[0]))
    
    # one_dim_list_tr = [n for one_dim in train_data[0] for n in one_dim]
    # one_dim_list_d = [n for one_dim in dev_data[0] for n in one_dim]
    # one_dim_list_t = [n for one_dim in test_data[0] for n in one_dim]

    # print('length_train_words',len(one_dim_list_tr))
    # print('length_dev_words',len(one_dim_list_d))
    # print('length_test_words',len(one_dim_list_t))

    # lower_tr_set = set([x.lower() for x in one_dim_list_tr])
    # lower_d_set = set([x.lower() for x in one_dim_list_d])
    # lower_t_set = set([x.lower() for x in one_dim_list_t])

    # print('vocab size tr',len(lower_tr_set))
    # print('vocab size d',len(lower_d_set))
    # print('vocab size t',len(lower_t_set))

    # d_unknown = list(set(lower_d_set).difference(lower_tr_set))
    # t_unknown = list(set(lower_t_set).difference(lower_tr_set))

    # print('unknown dev',len(d_unknown))
    # print('unknown test',len(t_unknown))
        

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    # print('Trigram')
    # evaluate(dev_data, pos_tagger)

    # #Predict tags for the test set
    test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence))
    for i in range(len(test_data)):
        sentence = test_data[i]
        print(i,'/',len(test_data))
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # # Write them to a file to update the leaderboard
     # # Write them to a file to update the leaderboard
    # with open("predictions.txt", "w") as file:
    #     file.write("id,tag")
    #     file.write("\n")
    #     for idx, tag in enumerate(test_predictions):
    #         file.write(f"{idx}, \"{tag}\n")

    # test_predictions = ['hi', 'my', 'name', 'is', 'julia']
    csv_data = {'id': list(range(len(test_predictions))), 'tag': test_predictions}
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv('test_y.csv', index=False)
