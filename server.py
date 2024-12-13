from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)


def is_numeric(column):
    # Fungsi untuk memeriksa apakah kolom numerik atau tidak
    return pd.to_numeric(column, errors='coerce').notnull().all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    num_clusters = int(request.form['numClusters'])  # Mendapatkan jumlah cluster dari request
    if not num_clusters:
        return jsonify({'error': 'No value provided for number of clusters'})

    try:
        num_clusters = int(num_clusters)  # Mengubah nilai menjadi integer
    except ValueError:
        return jsonify({'error': 'Invalid value for number of clusters'})

   
    # Proses file yang diunggah
    data = pd.read_csv(file)

    # Identifikasi kolom-kolom string
    string_columns = data.select_dtypes(include='object').columns.tolist()

    # One-hot encoding untuk kolom-kolom string
    encoded_data = pd.get_dummies(data[string_columns])

    # Menggabungkan data numerik dan hasil one-hot encoding
    numeric_columns = data.select_dtypes(include='number').columns.tolist()
    numeric_data = data[numeric_columns]
    processed_data = pd.concat([numeric_data, encoded_data], axis=1)

    # Pembersihan data dari nilai non-numerik
    processed_data = processed_data.apply(pd.to_numeric, errors='coerce')
    processed_data = processed_data.dropna()  # Menghapus baris dengan nilai yang tidak valid

    if processed_data.empty:
        return jsonify({'error': 'No valid numeric data'})


    #    # Pembersihan data dari nilai non-numerik
    # data = data.apply(pd.to_numeric, errors='coerce')
    # data = data.dropna()  # Menghapus baris dengan nilai yang tidak valid

    # if data.empty:
    #     return jsonify({'error': 'No valid numeric data'})

    
    # Analisis menggunakan K-means
    # kmeans = KMeans(n_clusters=3)  # Ubah sesuai kebutuhan Anda
    kmeans = KMeans(n_clusters=num_clusters)  # Menggunakan jumlah cluster yang diinginkan

    # kmeans.fit(data)
    kmeans.fit(processed_data)
    iterations = []
    for i in range(5):  # Contoh iterasi sebanyak 5 kali
        # kmeans.fit(data)
        kmeans.fit(processed_data)
        iterations.append(kmeans.cluster_centers_.tolist())  # Menyimpan pusat cluster

    return jsonify({'iterations': iterations})

if __name__ == '__main__':
    app.run(debug=True)
