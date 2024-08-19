import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.query_pipeline import QueryPipeline as QP, Link, InputComponent
from llama_index.query_engine.pandas import PandasInstructionParser
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
import os
import matplotlib.pyplot as plt

# API Anahtarı
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit başlat
st.title("CSV Dosyası Yükleme ve Semantik Arama")
st.write("CSV dosyanızı yükleyin ve sorgunuzu girin.")

# LLM seçimi
model_type = st.selectbox("Kullanmak istediğiniz dil modelini seçin", ["OpenAI"])
llm = OpenAI(model="gpt-3.5-turbo")

# CSV dosyasını yükle
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:")
    st.write(df.head())

    # SentenceTransformer modelini yükleme
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Sütun adlarını vektörleştirme
    column_embeddings = model.encode(df.columns.tolist())

    # FAISS dizini oluşturma (Kosinüs Benzerliği kullanarak semantik arama için normalize)
    faiss.normalize_L2(column_embeddings)
    d = column_embeddings.shape[1]  # Vektör boyutu
    index = faiss.IndexFlatIP(d)  # Inner Product (Kosinüs Benzerliği için)

    index.add(column_embeddings)  # Tüm sütun vektörlerini ekleme

    # Sorgu girişi
    query_str = st.text_input("Sorgunuzu girin", value="")

    if query_str:
        # Sorguyu vektörleştir ve normalize et
        query_embedding = model.encode([query_str])
        faiss.normalize_L2(query_embedding)

        # FAISS üzerinden en yakın sütunları bul
        k = 3  # En yakın 3 sütun
        distances, indices = index.search(query_embedding, k)
        relevant_columns = [df.columns[i] for i in indices[0]]
        
        st.write("İlgili sütun(lar):", relevant_columns)

        # Sorgu, yalnızca seçilen sütunlar üzerinden gerçekleştirilecek
        if relevant_columns:
            selected_columns = relevant_columns  # En yakın sütunları seç
            st.write(f"Sorgu sadece şu sütunlar üzerinden gerçekleştirilecek: {selected_columns}")
            
            # Pandas sorgusu için talimatları dinamik olarak oluştur
            instruction_str = (
                f"Yalnızca {', '.join(selected_columns)} sütunlarını kullanarak sorguyu Pandas ile çalıştırılabilir Python koduna çevirin.\n"
                "Eğer karmaşık işlemler gerekiyorsa, gruplama, toplama, birleştirme veya şekillendirme fonksiyonlarını kullanmayı düşünün.\n"
                "Eğer grafik isteniyorsa matplotlib ile uygun grafiği oluşturmayı düşünün.\n"
                "Eksik verileri uygun şekilde ele alın.\n"
                "Kodu, `eval()` fonksiyonuyla çalıştırılabilecek bir Python ifadesiyle bitirin.\n"
                "SADECE İFADEYİ YAZDIRIN.\n"
                "İfadeyi tırnak içinde yazmayın.\n"
                "Sadece Türkçe dilini kullanın.\n"
            )
            
            pandas_prompt_str = (
                "Python'da bir pandas dataframe ile çalışıyorsunuz.\n"
                "Dataframe'in adı `df`.\n"
                "Bu, `print(df.head())` çıktısının sonucudur:\n"
                "{df_str}\n\n"
                "Bu talimatları izleyin:\n"
                "{instruction_str}\n"
                "Sorgu: {query_str}\n\n"
                "İfade:"
            )

            pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
                instruction_str=instruction_str, df_str=df[selected_columns].head(5)
            )
            pandas_output_parser = PandasInstructionParser(df[selected_columns])
            response_synthesis_prompt = PromptTemplate(
                "Girdiğiniz soruya göre, sorgu sonuçlarından detaylı bir yanıt üretin.\n"
                "Sorgu: {query_str}\n\n"
                "Pandas Talimatları (isteğe bağlı):\n{pandas_instructions}\n\n"
                "Pandas Çıktısı: {pandas_output}\n\n"
                "Yanıt: "
            )

            # QueryPipeline oluşturma
            qp = QP(
                modules={
                    "input": InputComponent(),
                    "pandas_prompt": pandas_prompt,
                    "llm1": llm,
                    "pandas_output_parser": pandas_output_parser,
                    "response_synthesis_prompt": response_synthesis_prompt,
                    "llm2": llm,
                },
                verbose=True,
            )
            qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
            qp.add_links(
                [
                    Link("input", "response_synthesis_prompt", dest_key="query_str"),
                    Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
                    Link(
                        "pandas_output_parser",
                        "response_synthesis_prompt",
                        dest_key="pandas_output",
                    ),
                ]
            )
            qp.add_link("response_synthesis_prompt", "llm2")

            # Sorguyu çalıştır
            response = qp.run(query_str=query_str)
            st.write("Yanıt:")
            st.write(response.message.content)

            # Eğer çıktı matplotlib figürü içeriyorsa, bunu çizdir
            if "plot" in response.message.content:
                exec(response.message.content)
                st.pyplot(fig=plt.gcf())
