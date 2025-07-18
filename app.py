from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model_tomat.h5')
labels = ['Early Blight', 'Late Blight', 'Leaf Mold', 'Healthy']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Keterangan penyakit
deskripsi_penyakit = {
    'Early Blight': "Early blight ditandai dengan bercak gelap konsentris dan menyebar dari daun bawah. Disebabkan oleh jamur *Alternaria solani*.",
    'Late Blight': "Late blight menyebabkan bercak gelap dan basah. Penyakit ini menyebar cepat dan menyebabkan daun menguning dan membusuk.",
    'Leaf Mold': "Leaf mold muncul sebagai bercak kuning di atas daun dan jamur abu-abu di bawah daun. Umumnya terjadi di tempat lembap.",
    'Healthy': "Daun dalam kondisi sehat. Tidak ditemukan gejala penyakit."
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # ✅ Preprocessing untuk ukuran input VGG16
        img = image.load_img(file_path, target_size=(255, 255))  # Ubah ke 255x255
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # ✅ Prediction
        pred = model.predict(img_array)[0]
        top_index = np.argmax(pred)
        result = labels[top_index]

        # Format probabilitas semua kelas
        prediction_list = [(labels[i], f"{p*100:.2f}%") for i, p in enumerate(pred)]

        # ✅ Ambil deskripsi
        description = deskripsi_penyakit.get(result, "Deskripsi tidak tersedia.")

        return render_template(
            'result.html',
            label=result,
            image_path=file_path,
            predictions=prediction_list,
            description=description
        )
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)