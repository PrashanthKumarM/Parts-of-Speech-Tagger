# Parts of Speech Tagger

 Totally we have implemented 5 algorithms to detect the parts of speech.
 The training is fast as it takes in only the counts in various data structures.
 These are converted to required probabilities during the test phase.
 Hence all unwanted computations are avoided.
 Additionally the Best algorithm we have chosen is a modified
 Averaged Perceptron, which requires a different way of training.
 Yields better results than viterbi but takes a little longer on training.

# Naive Bayes:
 Detects the part of speech for a word based on
 maximum of the probabilities of part of speech
 given the word. 

# Gibbs Sampler:
 We get a sequence of observations based on random coin tosses
 biased by the joint probability distribution of
 probability of POS given the word and probability of POS
 based on previous POS. We get relative probability of this and
 then the coin toss is limited to POS by this. We do 100 samples
 and present the final 5 samples. I have gone up till 2000 samples.
 There wasn't a considerable change (accuracy improvement of 0.3).
 So I decided the "naive" way to pick the better tradeoff, given time. #BayesIsComing #PunIntended

# Viterbi (MAP):
 follows the formula emission * max(previous*transition)
 transitions are done in generate_transitions method and
 emissions are got from generate_emissions method. 
 Then the results are backtracked from the results of the final
 word. Unknown words are reserved and updated later based on the
 flow of the sentence (previous and next words).

# Max Marginal:
 Marginal probabilities of words with the parts of speech is
 inferred from the samples generated and then used to detect
 the part of speech. Unknown words are given a negligible probability
 value for each part of speech and then picked.

# Averaged Perceptron:
 The version included in this is a modified form of the averaged perceptron
 We have weight vectors and edges leading to the weights.
 On training, we predict the word and on a wrong prediction, reduce the
 wrong edge and increase the correct edge. Then based on the number of 
 averages, we average and update the weights for POS on a word
 Hence on prediction we take the edge with maximum weight and present it.
 Since this is a completely supervised learning we had to include reserves
 technique, where an unknown word in reserved and later updated based on the
 previous and the next word.

We ran this code for 2 sets of test data. a "tiny" set with 3 test cases and a "large" set with 2000 test cases. The training data used by us consists of 44000 cases with results.
The training data should be of the format (In case you decide to use a better one):

Word < Part of Speech > Word < Part of Speech > Word ....

In sense, each word should be followed by a space and then the appropriate part of speech.

The program should be run as:

				 $ python pos_tagger.py


Then enter the sentances that need to be tagged. Make sure to have spaces between each word and symbols too. The number of samples for Gibbs sampling can be varied by changing the constant MCMC_SAMPLES.

# Results (tiny):
                 					Words correct:     Sentences correct: 
				 0. Ground truth:      100.00%              100.00%
				        1. Naive:       97.62%               66.67%
				      2. Sampler:       92.86%                0.00%
				 3. Max marginal:       95.24%               33.33%
				          4. MAP:       95.24%               33.33%
				         5. Best:      100.00%              100.00%
# Results (large):
                						Words correct:     Sentences correct: 
					 0. Ground truth:      100.00%              100.00%
					       1. Naive:       92.91%               41.90%
					     2. Sampler:       90.95%               34.75%
					 3. Max marginal:       91.92%               37.85%
					         4. MAP:       92.44%               40.15%
					        5. Best:       93.52%               44.50%

# Time taken:

					 tiny (without Perceptron):
					 7.24 secs
					 tiny (with Perceptron):
					 15.24 secs

					 large (without Perceptron):
					 212.36 secs
					 large (with Perceptron):
					 314.57 secs

(Based on the problem statement in CS B551 Elements of Artificial Intelligence by Professor David J Crandall , Indiana University, Bloomington)