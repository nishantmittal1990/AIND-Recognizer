import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def calc_best_score(log_likelihood):
        """
        Max of dictionary of values by comparing each item by value at index
        :param log_likelihoods: 
        :return: 
        """
        return max(log_likelihood, key=log_likelihood.get)

    probabilities = []
    guesses = []

    # Iterating through each item in test set
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        word_id_feature_list_sequence, sequence_length = test_set.get_item_Xlengths(word_id)
        log_likelihoods = {}

        # Calculate Log Likelihood score for each word and model and append to probability list
        for word, model in models.items():
            try:
                score = model.score(word_id_feature_list_sequence, sequence_length)
                log_likelihoods[word] = score
            except:
                # As advised, eliminate non-viable models from consideration
                log_likelihoods[word] = float("-inf")
                continue
        # Appended probability list with log_likelihood score
        probabilities.append(log_likelihoods)
        # Appended guesses with word with maximum log likelihood score
        guesses.append(calc_best_score(log_likelihoods))

    return probabilities, guesses