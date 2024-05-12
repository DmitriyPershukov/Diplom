import pickle
similarity = pickle.load(open('artefacts/model.pkl', 'rb'))
products = pickle.load(open('artefacts/products.pkl', 'rb'))

def get_product_name_by_id(id):
    return products[products['id'] == id].iloc[0].product_name
def recommend_product(id, items_to_recommend):
    index = products[products['id'] == id].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    return products.iloc[list(map(lambda item: item[0], distance[0:items_to_recommend]))].id

if __name__ == '__main__':
    while True:
        try:
            user_input = input("Введите через пробел id продукта и число желаемых рекомендаций\n"
                                           "Для выхода из программы введите exit\n")
            if user_input == "exit":
                print("Завершение работы.")
                break
            splitted_user_input = list(map(int, user_input.split(" ")))
            id, items_to_recommend = splitted_user_input[0], splitted_user_input[1]
            try:
                print("\nРекомендации для продукта: " + get_product_name_by_id(id))
            except:
                print("Продукт с данным id не найден в системе. Попробуйте другой id\n")
                continue
            recommended_products_id = recommend_product(id, items_to_recommend)
            for i in recommended_products_id:
                print(get_product_name_by_id(i))
            print("\n")
        except:
            print("Введены неправильные данные. Попробуйте снова.\n")
            continue