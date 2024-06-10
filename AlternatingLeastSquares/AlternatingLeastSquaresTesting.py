import threadpoolctl
from AlternatingLeastSquaresDataPreprocessing import preprocess_data
from implicit import evaluation
import numpy as np

from Программа.AlternatingLeastSquares.AlternatingLeastSquaresModel import AlternatingLeastSquares


class ALSEvaluation:

    def __init__(self):
        self.test_log = []
        self._best_map = -1
        self.best_map_score = None
        self._best_f1 = -1
        self.best_f1_score = None

    def precision_at_k(self, user_id, K, model, test_users_items):
        recs = model.predict(user_id, K = K, filter_out_bought_items = True)
        recommended_items = recs['id'].values.tolist()
        relevant_items = np.where(test_users_items.toarray()[user_id, :] > 0)[0].tolist()
        tp = 0
        for i in recommended_items[:K]:
            if i in relevant_items:
                tp += 1
        return tp / K

    def recall_at_k(self, user_id, K, model, test_users_items):
        recs = model.predict(user_id, K=K, filter_out_bought_items=True)
        recommended_items = recs['id'].values.tolist()
        relevant_items = np.where(test_users_items.toarray()[user_id, :] > 0)[0].tolist()
        tp = 0
        for i in recommended_items[:K]:
            if i in relevant_items:
                tp += 1
        return tp / len(relevant_items)

    def f1_score_at_k(self, user_id, K, model, test_users_items):
        precision = self.precision_at_k(user_id, K, model, test_users_items)
        recall = self.recall_at_k(user_id, K, model, test_users_items)
        if recall + precision == 0:
            return 0
        return (2 * precision * recall) / (recall + precision)

    def average_f1_score_at_k(self, K, model, test_users_items):
        sum = 0
        users_to_test = np.where(np.sum(test_users_items, axis=1) > 0)[0]
        user_number = len(users_to_test)
        for i in users_to_test:
            sum += self.f1_score_at_k(i, K, model, test_users_items)
        return sum / user_number

    def average_precision_at_k(self, user_id, K, model, test_users_items):
        sum = 0
        recs = model.predict(user_id, K=K, filter_out_bought_items=True)
        relevant_items = np.where(test_users_items.toarray()[user_id, :] > 0)[0].tolist()
        relevant_items_count = len(relevant_items)
        rel = 0
        for i in range(K):
            if recs['id'].iloc[i] in relevant_items:
                rel = 1
            else:
                rel = 0
            sum += self.precision_at_k(user_id, i + 1, model, test_users_items) * rel
        return sum / min(K, relevant_items_count)

    def mean_average_precision_at_k(self, K, model, test_users_items):
        sum = 0
        users_to_test = np.where(np.sum(test_users_items, axis = 1)>0)[0]
        user_number = len(users_to_test)
        for i in users_to_test:
            sum += self.average_precision_at_k(i, K, model, test_users_items)
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
                        model_evaluation_map = self.mean_average_precision_at_k(K, model, test_set)
                        model_evaluation_message = "MAP@" + str(K) + ": " + str(model_evaluation_map)
                        model_evaluation_f1 = self.average_f1_score_at_k(K, model, test_set)
                        model_evaluation_message += ", F1@" + str(K) + ": " + str(model_evaluation_f1)
                        model_evaluation_message += ", MP@" + str(K) + ": " + str(self.mean_precision_at_k(K, model, test_set))
                        test_log_entry = "Model with factors: {}, regularization: {}, alpha: {}, iterations: {} scored: {}".format(
                                               current_factor,
                                                     current_regularization_factor,
                                                     current_alpha,
                                                     current_iterations_number,
                                                     model_evaluation_message)
                        self.test_log.append(test_log_entry)
                        if model_evaluation_map > self._best_map:
                            self._best_map = model_evaluation_map
                            self.best_map_score = test_log_entry
                        if model_evaluation_f1 > self._best_f1:
                            self._best_f1 = model_evaluation_f1
                            self.best_f1_score = test_log_entry

    def get_best_map_score(self):
        return self.best_map_score

    def get_best_f1_score(self):
        return self.best_f1_score


    def print_results(self):
        for i in self.test_log:
            print(i)
        print("Hyperparameters that give best map score:")
        print(self.best_map_score)
        print("Hyperparameters that give best F1 score:")
        print(self.best_f1_score)

if __name__ == '__main__':
    model_eval = ALSEvaluation()
    model_eval.test_model([45], [0.05], [1], [1000], 10)
    model_eval.print_results()
    #model_eval.test_model([30, 45, 60, 75, 90], [0.1, 0.01, 0.05], [1, 10, 20, 30, 40], [100, 300, 500, 700, 1000])
    # MAP@7: 0.7107313738892661
    # map100:3 -> 0.33
    # f1100:3 -> 0.30
    # mp100:3 -> 0.28
    # 1000:3 -> 0.366
    # map100:5 -> 0.37
    # f1100:5 -> 0.28
    # mp100:5 -> 0.2
    # 1000:5 -> -
    # 100:7 -> 0.32
    # 1000:7 -> -
    # 100:10 -> - 0.36
    # 1000:10 -> 0.45
