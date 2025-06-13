# inference.py

import joblib

# Load model dan preprocessing
model = joblib.load('models\spam_model.pkl')
tfidf = joblib.load(r'models\tfidf_vectorizer.pkl')
le = joblib.load('models\label_encoder.pkl')

# Fungsi untuk memprediksi
def predict_message(msg):
    X = tfidf.transform([msg])
    y_pred = model.predict(X)
    return le.inverse_transform(y_pred)[0]

if __name__ == '__main__':
    msg = input("Masukkan pesan email: ")
    hasil = predict_message(msg)
    print("Prediksi:", hasil)
