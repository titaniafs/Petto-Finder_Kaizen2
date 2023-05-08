from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

#menampilkan model dan gambar yang akan diprediksi
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model.h5')

#klasifikasi/output dari hasil prediksi
class_dict = {0: 'Cat (Kucing)', 1: 'Dog (Anjing)'}

#peraturan gambar yang nantinya akan diprediksi
def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    return class_dict[predicted_bit]

#buat pengaturan template
#NB: PERHATIKAN NAMA FILENYA
#untuk halaman home
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/home")
def home():
    return render_template('index.html')

#untuk halaman about 
@app.route("/about")
def about():
    return render_template('about.html')

#untuk halaman prediksi
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('prediction.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('prediction.html')

#untuk halaman team
@app.route("/team")
def team():
    return render_template('team.html')

#ini bukan untuk halaman, tapi fungsi untuk menyimpan gambar yang akan diprediksi
@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)