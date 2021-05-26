from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
import pickle
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

UPLOAD_FOLDER = r'C:\Users\Bindas\PycharmProjects\flaskProject\uploads'
ALLOWED_EXTENSIONS = {'jsonl'}

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(request.url)
    return render_template('index.html')


@app.route('/run')
def run():
    # Use pickle to load in the pre-trained model.
    with open(f'model/grid_sessions.pkl', 'rb') as f:
        model = pickle.load(f)

    products = pd.read_json(os.path.join(UPLOAD_FOLDER, "products.jsonl"), lines=True)
    sessions = pd.read_json(os.path.join(UPLOAD_FOLDER, "sessions.jsonl"), lines=True)
    users = pd.read_json(os.path.join(UPLOAD_FOLDER, "users.jsonl"), lines=True)

    shop_df = sessions.merge(products, on="product_id", how="left")
    shop_df = shop_df.merge(users, on='user_id', how='left')

    shop_df["discount_price"] = (shop_df["price"] * shop_df["offered_discount"])

    shop_df = shop_df[shop_df.columns[
        ~shop_df.columns.isin(["product_name", "purchase_id", "name", "street", "timestamp", "purchase_id_isnan"])]]
    shop_df = pd.get_dummies(shop_df)

    X = shop_df[shop_df.columns[~shop_df.columns.isin(
        ["event_type_BUY_PRODUCT", "event_type_VIEW_PRODUCT", "purchase_id_isnan", "timestamp"])]]

    session_ids = X["session_id"]
    X = X.drop('session_id', 1)

    flag = np.where(np.array(model.predict_proba(X))[:, 1] > 0.45, True, False)
    results = X[flag]
    select_columns = ["user_id", "product_id", "offered_discount", "price"]
    results = results[select_columns]
    results = pd.merge(session_ids, results, left_index=True, right_index=True, how="inner").drop_duplicates()

    return render_template('run.html', data=results)


if __name__ == '__main__':
    app.run()
