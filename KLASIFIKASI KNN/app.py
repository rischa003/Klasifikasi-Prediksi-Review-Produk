import streamlit as st
import joblib

# Load model & TF-IDF
knn = joblib.load("model_knn.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Judul Website
st.title("🔍 Analisis Sentimen Review Pakaian Anak (KNN)")
st.write("Aplikasi ini memprediksi sentimen review dari TikTok Shop menggunakan algoritma **K-Nearest Neighbor (KNN)**.")

# Input teks
review_input = st.text_area("📝 Masukkan review:", "")

# Tombol prediksi
if st.button("Prediksi"):
    if review_input.strip() == "":
        st.warning("Harap masukkan teks terlebih dahulu!")
    else:
        # Preprocessing: transform pakai TF-IDF
        fitur = tfidf.transform([review_input])
        hasil = knn.predict(fitur)[0]

        # Tampilkan hasil
        if hasil == "positif":
            st.success("✅ Sentimen: Positif")
        elif hasil == "negatif":
            st.error("❌ Sentimen: Negatif")
        else:
            st.info("ℹ️ Sentimen: Netral")
