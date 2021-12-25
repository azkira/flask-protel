from flask import Flask, flash, request, redirect, render_template, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import librosa
import librosa.display

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('There is no file')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('The file is not selected')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # this will secure the filename into pred.wav
        filename = secure_filename('pred.wav')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed files is - wav')
        return redirect(request.url)


@app.route('/api/prediction', methods=['GET'])
def get_json():
    med = np.asarray(librosa.load(str('./static/uploads/pred.wav')))
    med = med[0]
    window_size = 1024
    window = np.hanning(window_size)

    stft = librosa.core.spectrum.stft(
        med, n_fft=window_size, hop_length=512, window=window)

    out = np.abs(stft) ** 2
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), x_coords=None, y_coords=None,
                             y_axis='log', x_axis='time', sr=22050, hop_length=512, fmin=None, fmax=None, bins_per_octave=12, ax=ax)
    fig.savefig("./static/uploads/pred.jpg")

    model = tf.keras.models.load_model('model.h5')
    labels = ['food', 'brush', 'isolation']
    img = image.load_img('./static/uploads/pred.jpg', target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediksi = model.predict(images, batch_size=0)
    catlab = labels[np.argmax(prediksi)]
    outp = {
        "indikasi": catlab
    }
    return jsonify(outp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
