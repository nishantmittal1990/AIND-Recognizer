import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def calculate_best_score(self, score_bics):
        """
        Calculating min of list of lists by comparing each item by value at index 0
        
        :param score_bics: 
        :return: 
        """
        return min(score_bics, key=lambda x: x[0])

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Implementing model selection based in BIC scores

        # Initialize default value of BIC score
        score_bics = []

        # Looping through number of states from min-n_components to max_n_components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)  # GaussianHMM

                #  Computing log likelihood score
                log_likelihood = model.score(self.X, self.lengths)

                # Number of features used to train the model
                number_data_points = sum(self.lengths)
                # number_free_parameters = init_state_occupation_probs + transition_probs + emission_probs
                #                      = (number of states) + (no of states * (number of states-1)) +
                #                           (no of states * number of data points *2)
                # where number of states = n_components i.e. possibile state a hidden variable may be in - Value lies b/w
                # min_n_components and max_n_components
                # and number of data points = no. of features used to train the model
                number_free_parameters = (n_components ** 2) + (2 * n_components * number_data_points) -1

                # Bayesian information criteria: BIC = -2 * logL + p * logN
                # LogL = log Likelihood, p = no. of free parameters and N = no of data points
                score_bic = (-2 * log_likelihood) + (number_free_parameters * np.log(number_data_points))

                score_bics.append(tuple([score_bic, model]))

            except Exception as exception:
                pass

        return self.calculate_best_score(score_bics)[1] if score_bics else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    
    Unlike the Bayes Factor criterion, the Discriminant Factor Criterion is the difference between 
    the evidence of the model, given the corresponding data set, and the average over anti-evidences 
    of the model. By choosing the model which maximizes the evidence, and minimize the anti evidences, 
    the result is the best generative model for the correct class and the worst generative model for 
    the competitive classes; this scheme thus selects the most discriminant models, resulting in an 
    improved accuracy in regard to the classiﬁcation task.

    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    DIC is sum of 2 terms. The first term is difference between the likelihood of the data log(P(X(i)))
    and the average of anti-likelihood terms, where antilikelihood of the data X(i) against model M 
    is likelihood-like quantity in which model and data belongs to competing catagories.
    
    DIC = log likelihood of the data belonging to model - avg of antilog likelihood of data X vs model M
        = log(P(origional word)) - average(log(P(different words)))
    
    '''

    def best_score_dic(self, score_dics):
        """
        Computing max.
        :param score_dics: 
        :return: 
        """
        return max(score_dics, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #Implemented Model selection based on DIV

        different_words = []
        models = []
        score_dics = []
        # Looping over all_words_sequence i.e. sequence dictionary
        for word in self.words:
            # check if the current word from sequence is word_from_training, else consider it different word
            if word != self.this_word:
                different_words.append(self.hwords[word])

        try:
            # Looping over states ranging from min_n_components to max_n_components
            for n_components in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(n_components) # GaussianHMM
                # Computing score or likelihood for original current word
                log_likelihood_current_word = model.score(self.X, self.lengths)
                models.append((log_likelihood_current_word, model))

        except Exception as exception:
            pass

        for idx, ml in enumerate(models):
            log_likelihood_current_word, model = ml
            log_likelihood_different_word = [ml[1].score(wrd[0], wrd[1]) for wrd in different_words]
            score_dic = log_likelihood_current_word - np.mean(log_likelihood_different_word)
            score_dics.append(tuple([score_dic, ml[1]]))
        return self.best_score_dic(score_dics)[1] if score_dics else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def calculate_best_score(self, score_cv):
        """
        Calculating maximum of list of lists comparing each item by value indexed at 0
        :param score_cv: 
        :return: 
        """
        return max(score_cv, key=lambda x:x[0])

    # Split the data set into K equal folds
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implemented model selection using CV
        kf = KFold(random_state=None, n_splits=3, shuffle=False)

        # Initialize logL and score_cv
        log_likelihoods = []
        score_cvs = []

        # looping over HMM - number of states(min_n_components and max_n_components)
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:

                # Check sufficient data is available to split using KFold
                if len(self.sequences) > 2:

                    # Splitting model data in training and test
                    for train_idx , test_idx in kf.split(self.sequences):

                        # Concatenate sequences referenced in an index list and returns tuple of the new X,lengths
                        # Training sequences split using KFolds are recombined
                        self.X , self.lengths = combine_sequences(train_idx, self.sequences)

                        # Test sequence split using KFolds are recombined
                        X_test, lengths_test = combine_sequences(test_idx, self.sequences)

                        model = self.base_model(n_components)
                        log_likelihood = model.score(X_test, lengths_test)
                else:
                    model = self.base_model(n_components)
                    log_likelihood = model.score(self.X, self.lengths)

                log_likelihoods.append(log_likelihood)

                # Average logLikelihood
                avg_score_cvs = np.mean(log_likelihoods)
                score_cvs.append(tuple([avg_score_cvs, model]))

            except Exception as exception:
                pass

        return self.calculate_best_score(score_cvs)[1] if score_cvs else None
