import threadpoolctl

from AlternatingLeastSquaresDataPreprocessing import preprocess_data
from implicit.als import AlternatingLeastSquares
from implicit import evaluation

class ALSEvaluation:

    def __init__(self):
        self.test_log = []
        self._best_map = -1
        self._best_ndcg = -1
        self.best_map_score = None
        self.best_ndcg_score = None

    def test_model(self, factors_to_test, regularization_factors_to_test, alphas_to_test, iterations_to_test):
        user_items_csr, _, _ = preprocess_data()
        train_set, test_set = evaluation.train_test_split(user_items_csr)
        threadpoolctl.threadpool_limits(1, "blas")
        for current_factor in factors_to_test:
            for current_regularization_factor in regularization_factors_to_test:
                for current_alpha in alphas_to_test:
                    for current_iterations_number in iterations_to_test:
                        model = AlternatingLeastSquares(factors= current_factor,
                                                        regularization= current_regularization_factor,
                                                        alpha= current_alpha,
                                                        iterations= current_iterations_number)
                        model.fit(train_set)
                        model_evaluation = evaluation.ranking_metrics_at_k(model, train_set, test_set, K=7)
                        test_log_entry = "Model with factors: {}, regularization: {}, alpha: {}, iterations: {} scored: {}".format(
                                               current_factor,
                                                     current_regularization_factor,
                                                     current_alpha,
                                                     current_iterations_number,
                                                     model_evaluation)
                        self.test_log.append(test_log_entry)
                        if model_evaluation['map'] > self._best_map:
                            self._best_map = model_evaluation['map']
                            self.best_map_score = test_log_entry
                        if model_evaluation['ndcg'] > self._best_ndcg:
                            self._best_ndcg = model_evaluation['ndcg']
                            self.best_ndcg_score = test_log_entry

    def get_best_ndcg_score(self):
        return self.best_ndcg_score

    def get_best_map_score(self):
        return self.best_map_score


    def print_results(self):
        for i in self.test_log:
            print(i)
        print("Hyperparameters that give best map score:")
        print(self.best_map_score)
        print("Hyperparameters that give best ndcg score:")
        print(self.best_ndcg_score)

if __name__ == '__main__':
    model_eval = ALSEvaluation()
    model_eval.test_model([30, 45, 60, 75, 90], [0.1, 0.01, 0.05], [1, 10, 20, 30, 40], [100, 300, 500, 700, 1000])
    model_eval.print_results()
    #7 'map': 0.6215661180698026, 'ndcg': 0.7139702007690466
