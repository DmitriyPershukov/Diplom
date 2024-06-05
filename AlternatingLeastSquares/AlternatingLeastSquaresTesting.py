import threadpoolctl
from AlternatingLeastSquaresDataPreprocessing import preprocess_data
from implicit import evaluation
import numpy as np

from Программа.AlternatingLeastSquares.AlternatingLeastSquaresModel import AlternatingLeastSquares


class ALSEvaluation:

    def __init__(self):
        self.test_log = []
        self._best_map = -1
        self._best_ndcg = -1
        self.best_map_score = None
        self.best_ndcg_score = None

    def calculate_p_at_k(self, user_id, K, model, users_items):
        recs = model.predict(user_id, K = K, filter_out_bought_items = False)
        recommended_items = recs['id'].values.tolist()
        bought_items = np.where(users_items.toarray()[user_id, :] > 0)[0].tolist()
        precission = sum(el in bought_items for el in recommended_items) / K
        return precission

    def calculate_average_p_at_k(self, user_id, K, model, users_items):
        sum = 0
        for i in range(K):
            sum += self.calculate_p_at_k(user_id, K, model, users_items)
        return sum / K

    def calculate_map_at_k(self, K, model, test_users_items, users_items):
        sum = 0
        model.users_items = test_users_items
        user_number = model.users_items.shape[0]
        for i in range(user_number):
            sum += self.calculate_average_p_at_k(i, K, model, users_items)
        return sum / user_number

    def test_model(self, factors_to_test, regularization_factors_to_test, alphas_to_test, iterations_to_test, K):
        user_items_csr, _, _ = preprocess_data()
        train_set, test_set = evaluation.train_test_split(user_items_csr)
        for current_factor in factors_to_test:
            for current_regularization_factor in regularization_factors_to_test:
                for current_alpha in alphas_to_test:
                    for current_iterations_number in iterations_to_test:
                        model = AlternatingLeastSquares(factors= current_factor,
                                                        regularization= current_regularization_factor,
                                                        alpha= current_alpha,
                                                        iterations= current_iterations_number,
                                                        users_items=train_set)
                        model.fit()
                        model_evaluation = self.calculate_map_at_k(K, model, test_set, user_items_csr)
                        model_evaluation_message = "MAP@" + str(K) + ": " + str(model_evaluation)
                        test_log_entry = "Model with factors: {}, regularization: {}, alpha: {}, iterations: {} scored: {}".format(
                                               current_factor,
                                                     current_regularization_factor,
                                                     current_alpha,
                                                     current_iterations_number,
                                                     model_evaluation_message)
                        self.test_log.append(test_log_entry)
                        if model_evaluation > self._best_map:
                            self._best_map = model_evaluation
                            self.best_map_score = test_log_entry

    def get_best_map_score(self):
        return self.best_map_score


    def print_results(self):
        for i in self.test_log:
            print(i)
        print("Hyperparameters that give best map score:")
        print(self.best_map_score)

if __name__ == '__main__':
    model_eval = ALSEvaluation()
    model_eval.test_model([60], [0.1], [1], [1000], 7)
    model_eval.print_results()
    #model_eval.test_model([30, 45, 60, 75, 90], [0.1, 0.01, 0.05], [1, 10, 20, 30, 40], [100, 300, 500, 700, 1000])
    #model_eval.print_results()
    #7 'map': 0.6215661180698026, 'ndcg': 0.7139702007690466
    # MAP@7: 0.7107313738892661