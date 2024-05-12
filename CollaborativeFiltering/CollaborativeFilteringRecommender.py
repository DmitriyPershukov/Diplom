import numpy as np
from implicit.nearest_neighbours import CosineRecommender
import pickle

class CollaborativeFilteringRecommender:

    def __init__(self, model_path):
        self.model = CosineRecommender.load(model_path + '/model')
        self.products = pickle.load(open(model_path + '/products.pkl', 'rb'))
        self.vectorized_get_db_id_by_model_id = np.vectorize(self.get_db_id_by_model_id)
        self.vectorized_get_product_name_by_db_id = np.vectorize(self.get_product_name_by_db_id)

    def get_model_id_by_db_id(self, database_id):
        return self.products.index[self.products['product_id'] == database_id].tolist()[0]

    def get_db_id_by_model_id(self, model_id):
        return self.products.iloc[model_id]['product_id']

    def get_product_name_by_db_id(self, database_id):
        return self.products.loc[self.products['product_id'] == database_id].product_name.values[0]

    def recommend(self, database_id, get_scores=False, minimum_score=-1):
        model_id = self.get_model_id_by_db_id(database_id)
        prediction = self.model.similar_items(model_id, filter_items=model_id)
        recommendation_scores = prediction[1]
        elements_to_keep_count = len(recommendation_scores[recommendation_scores > minimum_score])
        recommendation_ids = self.vectorized_get_db_id_by_model_id(prediction[0])[:elements_to_keep_count]
        recommendation_scores = recommendation_scores[:elements_to_keep_count]
        if get_scores:
            return (recommendation_ids, recommendation_scores)
        else:
            return recommendation_ids

    def recommend_product_names(self, database_id, get_scores=False, minimum_score=-1.0):
        recommendations = self.recommend(database_id, get_scores, minimum_score)
        if type(recommendations) is tuple:
            return self.vectorized_get_product_name_by_db_id(recommendations[0]), recommendations[1]
        else:
            return self.vectorized_get_product_name_by_db_id(recommendations)

if __name__ == '__main__':
    model = CollaborativeFilteringRecommender("./Model")
    print(model.recommend_product_names(2445, get_scores=True))