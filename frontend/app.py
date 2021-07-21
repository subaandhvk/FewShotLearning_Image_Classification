import multiprocessing as mp
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from keras.models import load_model
import zipfile
from notebooks.simple_fruit_detector_by_cnn import train_model

app = Flask(__name__)

# Import the model
class_names = ['apple', 'broccoli', 'grape', 'lemon']
CNN_model = load_model("./CNN-output.h5")

global results
results = []


@app.route('/')
def index():
    return render_template('index.html')


def predict_cnn(data):
    print('Started function processing')
    data = [data]
    data = np.array(data)
    print(data.shape)

    predictions = CNN_model.predict(data)

    print('Before 1', predictions, predictions.shape)

    predictions = class_names[np.argmax(predictions)]

    print(predictions)

    return predictions


def predict_cnn2(data):
    print('Processing function 2')
    data = [data]
    data = np.array(data)
    print(data.shape)

    predictions = CNN_model.predict(data)
    print('Before 2', predictions, predictions.shape)

    predictions = class_names[np.argmax(predictions)]
    print(predictions)

    return predictions


def get_result(result):
    results.append(result)


@app.route('/predict', methods=['POST'])
def predict():
    if request.files['img_file']:
        print(request.files.values())
        img = Image.open(request.files['img_file'])
        print('*****************************************************************')

        img = img.resize((400, 400))
        img = [np.array(img)]

        output = []
        pool = mp.Pool(mp.cpu_count())
        res1 = pool.apply_async(predict_cnn, args=(img), callback=get_result)
        res2 = pool.apply_async(predict_cnn2, args=(img), callback=get_result)

        output.append(res1)
        output.append(res2)
        pool.close()
        pool.join()

        [r.wait() for r in output]

        print(results)
        if len(results) == 2:
            pred1 = results[0]
            pred2 = results[1]
            # pred3 = results[2]

        return render_template('index.html', prediction1=f" CNN Result : {pred1}",
                               prediction2=f"Constrastive Loss Result: {pred2}", prediction3='Three')


@app.route('/train', methods=['POST'])
def train():
    if request.files['data_file']:
        with zipfile.ZipFile(request.files['data_file'], "r") as zip_ref:
            zip_ref.extractall('../datasets/.')

        # model = train_model()

    return render_template('index.html', prediction1='Model training successful')


if __name__ == '__main__':
    app.run()
