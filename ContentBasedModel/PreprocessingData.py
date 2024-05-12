import pandas as pd
import sqlalchemy
import re
import html as ihtml
from bs4 import BeautifulSoup
import configparser
import os

def clean_text(text):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"\s+", " ", text)
    return text

def preprocess_data():
    config = configparser.ConfigParser()
    configpath = os.path.join(os.path.dirname(__file__), '../config.ini')
    config.read(configpath)
    db_user = config.get('database', 'user')
    db_password = config.get('database', 'password')
    db_host = config.get('database', 'host')
    db_name = config.get('database', 'name')
    conn = sqlalchemy.create_engine("mysql+pymysql://{}:{}@{}/{}".format(db_user, db_password, db_host, db_name))
    products = pd.read_sql_query("SELECT id, name, description FROM products WHERE is_demo_data = 0 OR deleted_at IS NULL", conn)
    products = products.rename(columns={'name':'product_name'})
    debug_names = ['DOGS', 'productimagestest', '8', 'sponge', 'First product with 1 image $100 - $80 - $20', 'Test Price', 'NEW PRODUCT', 'cat', 'Test PP Item', 'animal shampoo', 'Product name', 'Product placeholder', 'cap', 'new product for testing', 'ttt', 'A PRODUCT NAME', 'B PRODUCT NAME', 'hudi 3', 'car 1', 'car 2', 'new product', 'My best product 1', 'My best product 2', 'My best product 3',
       'My best product 4', 'My best product 5', 'Removable !!!!! product',
       'new product1', 'hudi', 'hudi 2', 'Test product', 'new product 33!', 'new product new new', 'whatasoft.shop2222','Super product', 'Iphone', '123', 'sadasdasdas', 'asdasdas', 'test', 'nan', 'test1',
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
    products = products[~products['product_name'].isin(debug_names)]
    products['description'].loc[products['description'].isnull()] = products['product_name']
    for i in range(len(products['description'])):
        products['description'].iloc[i] = clean_text(products['description'].iloc[i])
    products = products.reset_index(drop=True)
    return products
