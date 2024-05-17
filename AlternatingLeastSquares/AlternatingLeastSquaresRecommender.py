import numpy as np
import pickle

import threadpoolctl
from implicit.cpu.als import AlternatingLeastSquares


class ALSRecommender:

    def __init__(self, model_path):
        threadpoolctl.threadpool_limits(1, "blas")
        self.model = self.model = AlternatingLeastSquares.load(model_path + '/model')
        self.products = pickle.load(open(model_path + '/products.pkl', 'rb'))
        self.customer_id = pickle.load(open(model_path + '/customers.pkl', 'rb'))
        self.user_items_csr = pickle.load(open(model_path + '/user_items_csr.pkl', 'rb'))
        self.vectorized_get_db_product_id = np.vectorize(self.get_db_product_id)
        self.vectorized_get_product_name = np.vectorize(self.get_product_name)
    def get_model_product_id(self, database_id):
        return self.products.index[self.products['product_id'] == database_id].tolist()[0]
    def get_db_product_id(self, model_id):
        return self.products.iloc[model_id]['product_id']
    def get_product_name(self, database_id):
        return self.products.loc[self.products['product_id'] == database_id].product_name.values[0]
    def get_model_customer_id(self, database_id):
        return np.where(self.customer_id == database_id)[0][0]

    def recommend(self, customer_database_id):
        customer_model_id = self.get_model_customer_id(customer_database_id)
        recommendations = self.model.recommend(customer_model_id, self.user_items_csr[customer_model_id,:])
        return self.vectorized_get_db_product_id(recommendations[0]), recommendations[1]

    def get_product_name_recommendations(self, customer_database_id):
        recommendations = self.recommend(customer_database_id)
        return self.vectorized_get_product_name(recommendations[0]), recommendations[1]

if __name__ == '__main__':
    model = ALSRecommender('./Model')
    recs = model.get_product_name_recommendations(33289)
    print(recs[0])