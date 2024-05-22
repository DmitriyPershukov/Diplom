import configparser
import os
import pickle

import threadpoolctl

from AlternatingLeastSquaresDataPreprocessing import preprocess_data
from implicit.als import AlternatingLeastSquares

def train_als_model(model_path):
    user_items_csr, customers, products = preprocess_data()
    config = configparser.ConfigParser()
    configpath = os.path.join(os.path.dirname(__file__), '../config.ini')
    config.read(configpath)
    threadpoolctl.threadpool_limits(1, "blas")
    model = AlternatingLeastSquares(factors=int(config.get("als_hyperparameters", "factors")),
                                    regularization=float(config.get("als_hyperparameters", "regularization")),
                                    iterations=int(config.get("als_hyperparameters", "iterations")))
    model.fit(user_items_csr)

    model.save(model_path + '/model')
    pickle.dump(customers, open(model_path + '/customers.pkl', 'wb'))
    pickle.dump(products, open(model_path + '/products.pkl', 'wb'))
    pickle.dump(user_items_csr, open(model_path + '/user_items_csr.pkl', 'wb'))

if __name__ == '__main__':
    train_als_model('./Model')