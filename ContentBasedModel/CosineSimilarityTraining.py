from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from PreprocessingData import preprocess_data

def train_cosine_similarity_model():
    products = preprocess_data()
    cv = CountVectorizer()
    vector = cv.fit_transform(products['description'].values.astype('U')).toarray()
    similarity = cosine_similarity(vector)
    pickle.dump(similarity, open('artefacts/model.pkl', 'wb'))
    pickle.dump(products, open('artefacts/products.pkl', 'wb'))

if __name__ == '__main__':
    train_cosine_similarity_model()