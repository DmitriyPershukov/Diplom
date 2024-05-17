import configparser
import os
import implicit.als
import numpy as np
from implicit import evaluation
from scipy.sparse import csr_matrix
import pandas as pd
import sqlalchemy
import pickle


def train_model(model_path):
    config = configparser.ConfigParser()
    configpath = os.path.join(os.path.dirname(__file__), '../config.ini')
    config.read(configpath)
    db_user = config.get('database', 'user')
    db_password = config.get('database', 'password')
    db_host = config.get('database', 'host')
    db_name = config.get('database', 'name')
    conn = sqlalchemy.create_engine("mysql+pymysql://{}:{}@{}/{}".format(db_user, db_password, db_host, db_name))

    users_items = pd.read_sql_query("""with cleaned_products as (
       select id, name
       from products as p
       where is_demo_data = 0 and deleted_at is null
       ),
       cleaned_customers as (
       select id
       from customers
       where is_demo_data = 0
       )
       select distinct c.id as customer_id, p.id as product_id, p.name as product_name
       from
           cleaned_customers c
       join
           baskets b ON c.id = b.for_user
       join 
           basket_product bp on bp.basket_id = b.id
       join
           product_options po on bp.option_id = po.id
       join 
           cleaned_products p on p.id = po.product_id""", conn)

    debug_names = ['DOGS', 'productimagestest', '8', 'sponge', 'First product with 1 image $100 - $80 - $20',
                   'Test Price', 'NEW PRODUCT', 'cat', 'Test PP Item', 'animal shampoo', 'Product name',
                   'Product placeholder', 'cap', 'new product for testing', 'ttt', 'A PRODUCT NAME', 'B PRODUCT NAME',
                   'hudi 3', 'car 1', 'car 2', 'new product', 'My best product 1', 'My best product 2',
                   'My best product 3',
                   'My best product 4', 'My best product 5', 'Removable !!!!! product',
                   'new product1', 'hudi', 'hudi 2', 'Test product', 'new product 33!', 'new product new new',
                   'whatasoft.shop2222', 'Super product', 'Iphone', '123', 'sadasdasdas', 'asdasdas', 'test', 'nan',
                   'test1',
                   'test2', 'shop', 'a', 'test123', 'dsd', 'Product 1', 'sadsad',
                   '1234', 'sdfsdfsdfsdfsdf', '3454frfdfgdfg',
                   'Product with a lot of data', 'Product with SKU and no other data',
                   'Product with options; not visible',
                   'test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123',
                   'the name', 'Product with no options', 'sort', 'Вася', 'asdasdasd',
                   'sdhjsahdas', 'рарврыоа', 'qwqw', 'sadasdas',
                   'Product with a bunch of test',
                   'aaa 333333333 "33" <aka> \'22\' \\\\// \'Yo & Co\'',
                   'empty product', 'dasda', 'Product with price',
                   'Product with no stock', 'AirFort - Tiki Hut',
                   "AirFort - Farmer's Barn", 'AirFort - UFO', '1', '2', 'Search 2',
                   'Search123', 'New product', 'asasdas', 'New product with image',
                   'New product "man suit"', 'New product 1', 'Removable product',
                   'Ordered product', 'second product with images', 'New product233',
                   'Search', 'Iphone 2', 'Iphone 3', 'Iphone 1', 'v', 'Test',
                   'Iphone1', 'Product 22', 'Product 23', '0', 'Empty',
                   'whatasoft.shop', 'TTMZ', 'd', 'whatasoft.shop update']

    users_items = users_items[~users_items['product_name'].isin(debug_names)]
    customer_id = users_items['customer_id'].unique()
    product_id = users_items['product_id'].unique()

    products = users_items[['product_id', 'product_name']].drop_duplicates(subset='product_id').reset_index(drop=True)

    users_items_matrix = np.zeros(shape=(customer_id.size, product_id.size))

    user_indices_to_populate = [np.where(customer_id == i)[0][0] for i in users_items['customer_id']]
    item_indices_to_populate = [np.where(product_id == i)[0][0] for i in users_items['product_id']]

    for i in range(len(user_indices_to_populate)):
        users_items_matrix[user_indices_to_populate[i], item_indices_to_populate[i]] += 1

    users_items_csr_matrix = csr_matrix(users_items_matrix)

    from implicit.nearest_neighbours import CosineRecommender
    model = CosineRecommender(K=6)

    model.fit(users_items_csr_matrix)

    model.save(model_path + '/model')
    pickle.dump(products, open(model_path + '/products.pkl', 'wb'))


if __name__ == '__main__':
    #train_model("./Model")
    config = configparser.ConfigParser()
    configpath = os.path.join(os.path.dirname(__file__), '../config.ini')
    config.read(configpath)
    db_user = config.get('database', 'user')
    db_password = config.get('database', 'password')
    db_host = config.get('database', 'host')
    db_name = config.get('database', 'name')
    conn = sqlalchemy.create_engine("mysql+pymysql://{}:{}@{}/{}".format(db_user, db_password, db_host, db_name))

    users_items = pd.read_sql_query("""with cleaned_products as (
           select id, name
           from products as p
           where is_demo_data = 0 and deleted_at is null
           ),
           cleaned_customers as (
           select id
           from customers
           where is_demo_data = 0
           )
           select distinct c.id as customer_id, p.id as product_id, p.name as product_name
           from
               cleaned_customers c
           join
               baskets b ON c.id = b.for_user
           join 
               basket_product bp on bp.basket_id = b.id
           join
               product_options po on bp.option_id = po.id
           join 
               cleaned_products p on p.id = po.product_id""", conn)

    debug_names = ['DOGS', 'productimagestest', '8', 'sponge', 'First product with 1 image $100 - $80 - $20',
                   'Test Price', 'NEW PRODUCT', 'cat', 'Test PP Item', 'animal shampoo', 'Product name',
                   'Product placeholder', 'cap', 'new product for testing', 'ttt', 'A PRODUCT NAME', 'B PRODUCT NAME',
                   'hudi 3', 'car 1', 'car 2', 'new product', 'My best product 1', 'My best product 2',
                   'My best product 3',
                   'My best product 4', 'My best product 5', 'Removable !!!!! product',
                   'new product1', 'hudi', 'hudi 2', 'Test product', 'new product 33!', 'new product new new',
                   'whatasoft.shop2222', 'Super product', 'Iphone', '123', 'sadasdasdas', 'asdasdas', 'test', 'nan',
                   'test1',
                   'test2', 'shop', 'a', 'test123', 'dsd', 'Product 1', 'sadsad',
                   '1234', 'sdfsdfsdfsdfsdf', '3454frfdfgdfg',
                   'Product with a lot of data', 'Product with SKU and no other data',
                   'Product with options; not visible',
                   'test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123test 123',
                   'the name', 'Product with no options', 'sort', 'Вася', 'asdasdasd',
                   'sdhjsahdas', 'рарврыоа', 'qwqw', 'sadasdas',
                   'Product with a bunch of test',
                   'aaa 333333333 "33" <aka> \'22\' \\\\// \'Yo & Co\'',
                   'empty product', 'dasda', 'Product with price',
                   'Product with no stock', 'AirFort - Tiki Hut',
                   "AirFort - Farmer's Barn", 'AirFort - UFO', '1', '2', 'Search 2',
                   'Search123', 'New product', 'asasdas', 'New product with image',
                   'New product "man suit"', 'New product 1', 'Removable product',
                   'Ordered product', 'second product with images', 'New product233',
                   'Search', 'Iphone 2', 'Iphone 3', 'Iphone 1', 'v', 'Test',
                   'Iphone1', 'Product 22', 'Product 23', '0', 'Empty',
                   'whatasoft.shop', 'TTMZ', 'd', 'whatasoft.shop update']

    users_items = users_items[~users_items['product_name'].isin(debug_names)]
    customer_id = users_items['customer_id'].unique()
    product_id = users_items['product_id'].unique()

    products = users_items[['product_id', 'product_name']].drop_duplicates(subset='product_id').reset_index(drop=True)

    users_items_matrix = np.zeros(shape=(customer_id.size, product_id.size))

    user_indices_to_populate = [np.where(customer_id == i)[0][0] for i in users_items['customer_id']]
    item_indices_to_populate = [np.where(product_id == i)[0][0] for i in users_items['product_id']]

    for i in range(len(user_indices_to_populate)):
        users_items_matrix[user_indices_to_populate[i], item_indices_to_populate[i]] += 1

    users_items_csr_matrix = csr_matrix(users_items_matrix)

    from implicit.nearest_neighbours import CosineRecommender

    model = implicit.als.AlternatingLeastSquares(iterations=400)
    #model = CosineRecommender(K=10)

    model.fit(users_items_csr_matrix)

    for i in range(len(customer_id)):
        print(model.recommend(i, users_items_csr_matrix[i, :])[0])

    # train, test = evaluation.train_test_split(users_items_csr_matrix)
    # model = implicit.als.AlternatingLeastSquares(iterations=400)
    # model.fit(train)
    # xcv = evaluation.ranking_metrics_at_k(model, train, test)
    # print(xcv)