from flask import Flask, request, render_template, send_from_directory, jsonify, flash, redirect, Response
import sqlite3
from PIL import Image
from Preprocessing import convert_to_image_tensor, invert_image
import torch
from Model import SiameseConvNet, distance_metric
from io import BytesIO
import json
import math
import os

imageid=None

app = Flask(__name__)

images_folder = os.path.join('static', 'test_image')

app.config['UPLOAD_FOLDER'] = images_folder


def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load('Models/model_large_epoch_20', map_location=device))
    return model


def connect_to_db():
    conn = sqlite3.connect('user_signatures.db')
    return conn


def get_file_from_db(customer_id):
    cursor = connect_to_db().cursor()
    select_fname = """SELECT sign1,sign2,sign3 from signatures where customer_id = ?"""
    cursor.execute(select_fname, (customer_id,))
    item = cursor.fetchone()
    cursor.connection.commit()
    return item


def main():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signatures (customer_id TEXT PRIMARY KEY,sign1 BLOB, sign2 BLOB, sign3 BLOB)"""
    cursor = connect_to_db().cursor()
    cursor.execute(CREATE_TABLE)
    cursor.connection.commit()
    app.run(debug=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    customer_id = request.form['customerID']
    global imageid
    imageid=customer_id
    print(customer_id)
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        query = """DELETE FROM signatures where customer_id=?"""
        cursor.execute(query, (customer_id,))
        cursor = conn.cursor()
        query = """INSERT INTO signatures VALUES(?,?,?,?)"""
        cursor.execute(query, (customer_id, files[0].read(), files[1].read(), files[2].read()))
        conn.commit()

        return render_template('middle.html')
    except Exception as e:
        print(e)

@app.route('/img1')
def i1():
    images=get_file_from_db(imageid)
    return Response(images[0],mimetype='image/jpg')

@app.route('/img2')
def i2():
    images=get_file_from_db(imageid)
    return Response(images[1],mimetype='image/jpg')

@app.route('/img3')
def i3():
    images=get_file_from_db(imageid)
    return Response(images[2],mimetype='image/jpg')

imageid=None

@app.route('/verify', methods=['POST'])
def verify():
    try:
        customer_id = request.form['customerID']
        file=request.files['newSignature']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        input_image = Image.open(request.files['newSignature'])
        input_image_tensor = convert_to_image_tensor(invert_image(input_image)).view(1, 1, 220, 155)
        anchor_images = [Image.open(BytesIO(x)) for x in get_file_from_db(customer_id)]
        anchor_image_tensors = [convert_to_image_tensor(invert_image(x)).view(-1, 1, 220, 155)
                                for x in anchor_images]
        model = load_model()
        mindist = math.inf
        for anci in anchor_image_tensors:
            f_A, f_X = model.forward(anci, input_image_tensor)
            dist = float(distance_metric(f_A, f_X).detach().numpy())
            mindist = min(mindist, dist)

            if dist <= 0.145139:  # Threshold obtained using Test.py
                return render_template("result.html", match="Yes", threshold=0.145139, distance=mindist, file=path)
            return render_template("result.html", match="No", threshold=0.145139, distance=mindist, file=path)
    except Exception as e:
        print(e)


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run(debug=True)
