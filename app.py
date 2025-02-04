import os
import fitz  # PyMuPDF
import streamlit as st
from groq import Groq
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from key import api_key

# API Anahtarını buraya ekleyin
PDF_FOLDER = "pdf_files"

# Streamlit Arayüzü
st.title("📄 Akıllı PDF Arama & Soru Cevaplama")

# PDF Yükleme
uploaded_files = st.file_uploader("PDF Dosyalarını Yükleyin", accept_multiple_files=True, type=["pdf"])

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read()) as doc:
        for page in doc:
            text += page.get_text()
    return text

# PDF'leri kaydet ve işleme al
pdf_texts = {}
if uploaded_files:
    os.makedirs(PDF_FOLDER, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        pdf_texts[file.name] = extract_text_from_pdf(file)

# Daha önce kaydedilmiş PDF'leri oku
for file_name in os.listdir(PDF_FOLDER):
    if file_name.endswith(".pdf") and file_name not in pdf_texts:
        pdf_path = os.path.join(PDF_FOLDER, file_name)
        with open(pdf_path, "rb") as f:
            pdf_texts[file_name] = extract_text_from_pdf(f)

# Kullanıcıdan soru al
query = st.text_input("🔍 Bir soru girin")

if query:
    # BERT modelini yükle
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    # PDF içeriklerini embedding'e çevir
    pdf_embeddings = {file: get_embedding(content) for file, content in pdf_texts.items()}
    
    # Soruyu embedding'e çevir
    query_embedding = get_embedding(query)

    # Cosine Similarity hesapla
    similarities = {file: cosine_similarity(query_embedding, emb)[0][0] for file, emb in pdf_embeddings.items()}
    
    # En benzer PDF'yi seç
    most_similar_pdf = max(similarities, key=similarities.get)
    
    st.subheader(f"📄 En alakalı PDF: {most_similar_pdf}")
    st.write(f"📊 Benzerlik Skoru: {similarities[most_similar_pdf]:.4f}")
    st.text_area("🔍 PDF İçeriği (İlk 500 karakter)", pdf_texts[most_similar_pdf][:500])
    
    # Groq istemcisini başlat
   
    client = Groq(
    api_key= api_key )
    # PDF özetleme
    context = pdf_texts[most_similar_pdf][:2000]
    summary_prompt = f"Context: {context}\nPlease summarize the above text in a short and concise paragraph, highlighting the key points."
    
    summary_response = client.chat.completions.create(
        messages=[{"role": "user", "content": summary_prompt}],
        model="llama3-70b-8192"
    )
    
    summary = summary_response.choices[0].message.content
    st.subheader("📜 PDF Özeti")
    st.write(summary)
    
    # Soruyu cevaplama
    qa_prompt = f"Context: {summary}\nQuestion: {query}\nAnswer:"
    qa_response = client.chat.completions.create(
        messages=[{"role": "user", "content": qa_prompt}],
        model="llama3-70b-8192"
    )
    
    qa_answer = qa_response.choices[0].message.content
    st.subheader("📜 Cevap")
    st.write(qa_answer)
