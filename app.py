from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
import flask
import pickle
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime


UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {'jsonl'}

# columns must be in the exact same orfer for the model to work
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

shop_df = sessions.merge(products, on="product_id", how="left")
shop_df = shop_df.merge(users, on='user_id', how = 'left')
shop_df["discount_price"] = (shop_df["price"] * shop_df["offered_discount"])
shop_df = shop_df[shop_df.columns[~shop_df.columns.isin(["product_name", "purchase_id", "name", "street"])]]
shop_df = pd.get_dummies(shop_df)
shop_df_check = shop_df.copy()
y_lr = shop_df_check["event_type_BUY_PRODUCT"]
shop_df_check = shop_df_check[shop_df_check.columns[~shop_df_check.columns.isin(["event_type_BUY_PRODUCT", "event_type_VIEW_PRODUCT", "purchase_id_isnan", "timestamp"])]]
shop_df_xgb = shop_df_check.copy()
y_xgb = y_lr.copy()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html', original_input={}, result="Please submit the values first")
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
                # save(user_id, product_id, offered_discount, variant, result)
            else:
                result = "There is no data for this product"
        else:
            result = "There is no data for this user"

        return flask.render_template('main.html',
                                     original_input={'User id': user_id,
                                                     'Product id': product_id, 'Offered discount': offered_discount, 'Variant': variant},
                                     result=result)

# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(request.url)
#     return render_template('index.html')


@app.route('/run')
def run():
    # Use pickle to load in the pre-trained model.
    with open(os.path.join(os.getcwd(),'model/grid_sessions.pkl'), 'rb') as f:
        model = pickle.load(f)

    X = shop_df[shop_df.columns[~shop_df.columns.isin(
        ["event_type_BUY_PRODUCT", "event_type_VIEW_PRODUCT", "purchase_id_isnan", "timestamp"])]]

    session_ids = X["session_id"]
    X = X.drop('session_id', 1)

    flag = np.where(np.array(model.predict_proba(X))[:, 1] > 0.45, True, False)
    results = X[flag]
    select_columns = ["user_id", "product_id", "offered_discount", "price"]
    results = results[select_columns]
    results = pd.merge(session_ids, results, left_index=True, right_index=True, how="inner").drop_duplicates()
    results.reset_index(inplace=True)
    results = results.drop("index", axis=1)

    return render_template('run.html', data=results)

# # returns single prediction for given parameters (one row)
# def ABtesting(given_user_id, model_lr, model_xgb):
#     # return predictions for given parameters
#     current_user_row = users[users['user_id'] == given_user_id]
#
#     if current_user_row['variant'] == 'A':
#         return model_xgb.predict()
#     else:
#         return model_lr.predict()


def predict(X, user_id):
    current_user = users[users.user_id == user_id]
    variant = current_user["variant"].to_numpy()[0]

    # check the variant
    if  variant == 'A':
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
    logs_path = os.path.join(UPLOAD_FOLDER, 'logs/logs.csv')
    if os.path.exists(logs_path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    print(df)
    print(append_write)
    df.to_csv(logs_path,  mode='a+', header=False)


if __name__ == '__main__':
    app.run()
