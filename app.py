import multiprocessing as mp
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from keras.models import load_model
import zipfile
import os
import shutil
from notebooks.cnn_transfer_learning import train_model, predict_cnn_model

app = Flask(__name__)

# Import the model
class_names = ['apple', 'broccoli', 'grape', 'lemon', 'orange', 'strawberry']
CNN_model = load_model("CNN-output.h5")

global results
results = []
@app.route('/')
def index():
    return render_template('index.html')

def get_result(result):
    results.append(result)


@app.route('/predict', methods=['POST'])
def predict():
    pred1 = pred2 = pred3 = None
    pool = mp.Pool(mp.cpu_count())
    output = []

    if request.files['img_file']:
        img = Image.open(request.files['img_file'])

        img = img.resize((200, 200))
        img = [np.array(img)]

        res1 = pool.apply_async(predict_cnn_model, args=(img), callback=get_result)
        output.append(res1)

    if request.files['img_file2'] and request.files['img_file']:
        print('process2')
        f1 = request.files['img_file']
        f2 = request.files['img_file2']

        path = './datasets/new'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)

        f1.save('./datasets/new/0.jpg')
        f2.save('./datasets/new/1.jpg')

        res2 = pool.apply_async(predict_triplets, args=(), callback=get_result)
        output.append(res2)

        pass

    pool.close()
    pool.join()

    [r.wait() for r in output]

    if len(results) == 2:
        pred1 = results[0]
        pred2 = results[1]
    elif len(results) == 1:
        pred1 = results[0]

    results.clear()
    output.clear()

    return render_template('index.html', prediction1=f" CNN Result : {pred1}",
                            prediction2=f"Triplet Loss Result: {pred2}")


@app.route('/train', methods=['POST'])
def train():
    if request.files['data_file']:
        with zipfile.ZipFile(request.files['data_file'], "r") as zip_ref:
            zip_ref.extractall('./datasets/.')

        model = train_model()

    return render_template('train.html', result='Model training successful')


if __name__ == '__main__':
    app.run()
