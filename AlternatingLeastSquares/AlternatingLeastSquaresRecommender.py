import numpy as np
import pickle
from AlternatingLeastSquaresModel import AlternatingLeastSquares

class ALSRecommender:

    def __init__(self, model_path):
        self.model = AlternatingLeastSquares.load(model_path)
        self.products = pickle.load(open(model_path + '/products.pkl', 'rb'))
        self.customers = pickle.load(open(model_path + '/customers.pkl', 'rb'))
        self.vectorized_get_db_product_id = np.vectorize(self.get_db_product_id)
        self.vectorized_get_product_name = np.vectorize(self.get_product_name)
    def get_model_product_id(self, database_id):
        return self.products.index[self.products['product_id'] == database_id].tolist()[0]
    def get_db_product_id(self, model_id):
        return self.products.iloc[model_id]['product_id']
    def get_product_name(self, database_id):
        return self.products.loc[self.products['product_id'] == database_id].product_name.values[0]
    def get_model_customer_id(self, database_id):
        return self.customers.index[self.customers['customer_id'] == database_id].tolist()[0]
    def get_customer_db_id_by_name(self, customer_name):
        return self.customers.loc[self.customers['customer_name'] == customer_name].customer_id.values[0]

    def recommend(self, customer_database_id):
        customer_model_id = self.get_model_customer_id(customer_database_id)
        recommendations = self.model.recommend(customer_model_id, self.user_items_csr[customer_model_id,:])
        return self.vectorized_get_db_product_id(recommendations[0]), recommendations[1]

    def get_recommendations_with_names(self, customer_name):
        customer_database_id = self.get_customer_db_id_by_name(customer_name)
        recommendations = self.recommend(customer_database_id)
        return self.vectorized_get_product_name(recommendations[0]), recommendations[1]

if __name__ == '__main__':
    model = ALSRecommender('./Model')
    recs = model.get_recommendations_with_names("Mary Dondero")
    print(recs)