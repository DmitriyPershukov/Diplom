import configparser
import os
import pickle
from AlternatingLeastSquaresDataPreprocessing import preprocess_data
import numpy as np
import pandas as pd
from tqdm import tqdm
class AlternatingLeastSquares:
    def __init__(self, factors, regularization, iterations, alpha):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.P = None
        self.U = None

    def _least_squares(self, users_items, X, Y, type):
        YtY = Y.T.dot(Y)
        if type == 'user':
            confidence = 1 + self.alpha * users_items.toarray()
            preferences = users_items.copy()
        else:
            confidence = (1 + self.alpha * users_items.toarray()).T
            preferences = users_items.T.copy()
        newX = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Cu = np.diag(confidence[i,:])
            YtCuY = YtY + Y.T.dot(Cu - np.eye(confidence.shape[1])).dot(Y)
            YtCuYplusLambdaI = YtCuY + self.regularization*np.eye(self.factors)
            newX[i,:] = np.squeeze(np.linalg.inv(YtCuYplusLambdaI).dot(Y.T).dot(Cu) @ preferences[i,:].T)
        return newX
    def fit(self, users_items):
        U = np.random.rand(users_items.shape[0], self.factors)
        P = np.random.rand(users_items.shape[1], self.factors)

        for i in tqdm(range(self.iterations)):
            U = self._least_squares(users_items, U, P, 'user')
            P = self._least_squares(users_items, P, U, 'item')

        self.U = U
        self.P = P

    def predict(self, user_id, K = 7):
        scores = self.U[user_id,:].dot(self.P.T)
        data_payload = {'id':np.arange(self.P.shape[0]).tolist(), 'score': scores.tolist()}
        return pd.DataFrame(data_payload).sort_values(by='score', ascending=False).head(K)

    def save(self, path):
        pickle.dump(self, open(path + '/model.pkl', 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path + '/model.pkl', 'rb'))

if __name__ == '__main__':
    user_items_csr, customers, products = preprocess_data()
    config = configparser.ConfigParser()
    configpath = os.path.join(os.path.dirname(__file__), '../config.ini')
    config.read(configpath)
    model = AlternatingLeastSquares(factors= int(config.get("als_hyperparameters", "factors")),
                                    regularization= float(config.get("als_hyperparameters", "regularization")),
                                    alpha= float(config.get("als_hyperparameters", "alpha")),
                                    iterations= int(config.get("als_hyperparameters", "iterations")))
    model.fit(user_items_csr)
    path = './Model'
    model.save(path)
    pickle.dump(customers, open(path + '/customers.pkl', 'wb'))
    pickle.dump(products, open(path + '/products.pkl', 'wb'))


