from AlternatingLeastSquaresDataPreprocessing import preprocess_data
from implicit.als import AlternatingLeastSquares
from implicit import evaluation

class ALSEvaluation:

    def __init__(self):
        self.test_log = []
        self._best_map = -1
        self._best_auc = -1
        self.best_map_score = None
        self.best_auc_score = None

    def test_model(self, factors_to_test, regularization_factors_to_test, iterations_to_test):
        user_items_csr, _, _ = preprocess_data()
        train_set, test_set = evaluation.train_test_split(user_items_csr)
        for current_factor in factors_to_test:
            for current_regularization_factor in regularization_factors_to_test:
                for current_iterations_number in iterations_to_test:
                    model = AlternatingLeastSquares(factors= current_factor,
                                                    regularization= current_regularization_factor,
                                                    iterations= current_iterations_number)
                    model.fit(train_set)
                    model_evaluation = evaluation.ranking_metrics_at_k(model, train_set, test_set)
                    test_log_entry = "Model with factor: {}, regularization: {}, iterations: {} scored: {}".format(current_factor,
                                                 current_regularization_factor,
                                                 current_iterations_number,
                                                 model_evaluation)
                    self.test_log.append(test_log_entry)
                    if model_evaluation['map'] > self._best_map:
                        self._best_map = model_evaluation['map']
                        self.best_map_score = test_log_entry
                    if model_evaluation['map'] > self._best_auc:
                        self._best_auc = model_evaluation['map']
                        self.best_auc_score = test_log_entry

    def get_best_auc_score(self):
        return self.best_auc_score

    def get_best_map_score(self):
        return self.best_map_score


    def print_results(self):
        for i in self.test_log:
            print(i)
        print("Hyperparameters that give best auc score:")
        print(self.best_auc_score)
        print("Hyperparameters that give best map score:")
        print(self.best_map_score)

if __name__ == '__main__':
    model_eval = ALSEvaluation()
    model_eval.test_model([45], [0.1], [3000])
    model_eval.print_results()
