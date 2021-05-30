from flask import Flask
import flask
import pickle
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime


UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'jsonl'}

# columns must be in the exact same order for the model to work
LOGISTIC_REGRESSION_COLUMNS = ['user_id','product_id','price','discount_price',
 'category_path_Gry i konsole;Gry na konsole;Gry Xbox 360','category_path_Komputery;Monitory;Monitory LCD',
 'category_path_Sprzęt RTV;Video;Odtwarzacze DVD','category_path_Sprzęt RTV;Video;Telewizory i akcesoria;Anteny RTV',
 'category_path_Sprzęt RTV;Video;Telewizory i akcesoria;Okulary 3D','category_path_Telefony i akcesoria;Akcesoria telefoniczne;Zestawy głośnomówiące',
 'category_path_Telefony i akcesoria;Telefony komórkowe','category_path_Telefony i akcesoria;Telefony stacjonarne',
 'city_Gdynia','city_Kutno','city_Mielec','city_Radom','city_Warszawa']
XGB_COLUMNS = ['user_id', 'product_id', 'offered_discount', 'price', 'discount_price',
       'category_path_Gry i konsole;Gry komputerowe',
       'category_path_Gry i konsole;Gry na konsole;Gry PlayStation3',
       'category_path_Gry i konsole;Gry na konsole;Gry Xbox 360',
       'category_path_Komputery;Drukarki i skanery;Biurowe urządzenia wielofunkcyjne',
       'category_path_Komputery;Monitory;Monitory LCD',
       'category_path_Komputery;Tablety i akcesoria;Tablety',
       'category_path_Sprzęt RTV;Audio;Słuchawki',
       'category_path_Sprzęt RTV;Przenośne audio i video;Odtwarzacze mp3 i mp4',
       'category_path_Sprzęt RTV;Video;Odtwarzacze DVD',
       'category_path_Sprzęt RTV;Video;Telewizory i akcesoria;Anteny RTV',
       'category_path_Sprzęt RTV;Video;Telewizory i akcesoria;Okulary 3D',
       'category_path_Telefony i akcesoria;Akcesoria telefoniczne;Zestawy głośnomówiące',
       'category_path_Telefony i akcesoria;Akcesoria telefoniczne;Zestawy słuchawkowe',
       'category_path_Telefony i akcesoria;Telefony komórkowe',
       'category_path_Telefony i akcesoria;Telefony stacjonarne',
       'city_Gdynia', 'city_Konin', 'city_Kutno', 'city_Mielec', 'city_Police',
       'city_Radom', 'city_Szczecin', 'city_Warszawa']


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# preparing datasets
products = pd.read_json(os.path.join(UPLOAD_FOLDER, "products.jsonl"), lines=True)
sessions = pd.read_json(os.path.join(UPLOAD_FOLDER, "sessions.jsonl"), lines=True)
users = pd.read_json(os.path.join(UPLOAD_FOLDER, "users.jsonl"), lines=True)

# hash user_id
users['variant'] = pd.util.hash_pandas_object(users['user_id']) % 2
users['variant'].replace({0: "A", 1: "B"}, inplace=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html', original_input={}, result="Please submit the values first", data=get_logs().tail(5))
    if flask.request.method == 'POST':
        user_id = int(flask.request.form['user_id'])
        product_id = int(flask.request.form['product_id'])
        offered_discount = int(flask.request.form['offered_discount'])

        user, product = get_submit_data(user_id, product_id)

        X = pd.concat([user, product], axis=1)
        X["offered_discount"] = offered_discount
        X["discount_price"] = X["offered_discount"] * X["price"]

        variant = "-"

        if user.shape[0] != 0:
            if product.shape[0] != 0:
                prediction, variant = predict(X, user_id)
                if prediction == 0:
                    result = "VIEW"
                else:
                    result = "BUY"
                save(user_id, product_id, offered_discount, variant, result)
            else:
                result = "There is no data for this product"
        else:
            result = "There is no data for this user"

        return flask.render_template('main.html',
                                     original_input={'User id': user_id,
                                                     'Product id': product_id, 'Offered discount': offered_discount, 'Variant': variant},
                                     result=result, data=get_logs().tail(5))


def predict(X, user_id):
    current_user = users[users.user_id == user_id]
    variant = current_user["variant"].to_numpy()[0]

    # check the variant
    if variant == 'A':
        X = X[LOGISTIC_REGRESSION_COLUMNS]
        model = load_model('model/model_lr.pkl')
    else:
        X = X[XGB_COLUMNS]
        model = load_model('model/model_xgb.pkl')
    return model.predict(X), variant


def load_model(path_from_cwd):
    with open(os.path.join(os.getcwd(), path_from_cwd), 'rb') as f:
        model = pickle.load(f)
    return model


def get_submit_data(user_id, product_id):
    """Return user and product from our data"""
    products_data = pd.read_json(os.path.join(UPLOAD_FOLDER, "products.jsonl"), lines=True)
    users_data = pd.read_json(os.path.join(UPLOAD_FOLDER, "users.jsonl"), lines=True)

    users_data = users_data[users_data.columns[~users_data.columns.isin(["name", "street"])]]
    users_data = pd.get_dummies(users_data)

    products_data = products_data[products_data.columns[~products_data.columns.isin(["product_name"])]]
    products_data = pd.get_dummies(products_data)

    user = users_data[users_data.user_id == user_id]
    product = products_data[products_data.product_id == product_id]

    user.reset_index(inplace=True)
    user = user.drop("index", axis=1)
    product.reset_index(inplace=True)
    product = product.drop("index", axis=1)

    return user, product


def save(user_id, product_id, offered_discount, variant, algo_decision):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data = {
        'time': [dt_string],
        'user_id': [user_id],
        'product_id': [product_id],
        'offered_discount': [offered_discount],
        'variant': [variant],
        'algorithm_decision': [algo_decision]
    }
    df = pd.DataFrame(data)
    logs_path = os.path.join(os.getcwd(), 'logs/logs.csv')
    if os.path.exists(logs_path):
        df.to_csv(logs_path,  mode='a', header=False, index=False)
    else:
        df.to_csv(logs_path, mode='w', header=True, index=False)


def get_logs():
    logs_path = os.path.join(os.getcwd(), 'logs/logs.csv')
    return pd.read_csv(logs_path)


if __name__ == '__main__':
    app.run()
