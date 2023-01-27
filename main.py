from tests.test_model_search import TestModelSearch

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers
# (i.e. use logistic regression), each with 3 different sets of
# hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf
# and parampters settings
# 5. Please set up your code to be run and save the results to the directory
# that its executed from
# 6. Investigate grid search function


if __name__ == '__main__':
    test_obj = TestModelSearch()
    test_obj.execute_tests()
