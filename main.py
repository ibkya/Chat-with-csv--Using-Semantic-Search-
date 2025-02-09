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
st.title("Chat with Excel and Visualize!")
st.write("Upload csv and let's start!")

# LLM seçimi
model_type = st.selectbox("Choose your model", ["Llama-3.1:8B", "Llama-3.1:70B (Offline - Bellek Sorunu)"])
llm = OpenAI(model="gpt-3.5-turbo")

# CSV dosyası yükleme
uploaded_file = st.file_uploader("Upload an Excel File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded File:")
    st.write(df.head())

    # SentenceTransformer modelini yükleme
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Sütun adlarını vektörleştirme
    column_embeddings = model.encode(df.columns.tolist())
    
    # FAISS dizini oluşturma
    faiss.normalize_L2(column_embeddings)
    d = column_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(column_embeddings)

    # Kullanıcıdan sorgu girişi
    query_str = st.text_input("Type Here:", value="")

    if query_str:
        # Kullanıcı sorgusunu vektörleştirip normalize etme
        query_embedding = model.encode([query_str])
        faiss.normalize_L2(query_embedding)

        # FAISS ile en yakın sütunları bulma
        k = 3
        distances, indices = index.search(query_embedding, k)
        relevant_columns = [df.columns[i] for i in indices[0]]
        
        st.write("Similar Columns:", relevant_columns)

        if relevant_columns:
            selected_columns = relevant_columns
            st.write(f"Similar Columns: {selected_columns}")

            # Pandas sorgusu için talimatları oluşturma
            instruction_str = (
                f"Sorguyu yalnızca şu sütunları kullanarak Pandas ile çalıştırılabilir bir Python koduna çevir: {', '.join(selected_columns)}.\n"
                "Eğer karmaşık işlemler gerekiyorsa, gruplama, toplama, birleştirme veya yeniden şekillendirme gibi fonksiyonları kullanmayı düşün.\n"
                "Eğer bir grafik isteniyorsa, matplotlib kullanarak uygun bir grafik oluşturmayı düşün.\n"
                "Eksik verileri uygun şekilde işle.\n"
                "Kodu `eval()` fonksiyonu ile çalıştırılabilir bir Python ifadesi olarak bitir.\n"
                "SADECE İFADEYİ YAZDIR.\n"
                "İfadeyi tırnak içine alma.\n"
                "Sadece Türkçe dilini kullan.\n"
            )

            pandas_prompt_str = (
                "Python'da bir pandas dataframe ile çalışıyorsun.\n"
                "Dataframe'in adı `df`.\n"
                "Bu, `print(df.head())` çıktısıdır:\n"
                "{df_str}\n\n"
                "Şu talimatları uygula:\n"
                "{instruction_str}\n"
                "Sorgu: {query_str}\n\n"
                "İfade:"
            )

            response_synthesis_prompt = PromptTemplate(
                "Girdiğiniz sorguya göre sonuçlardan detaylı bir yanıt oluştur.\n"
                "Sorgu: {query_str}\n\n"
                "Pandas Talimatları:\n{pandas_instructions}\n\n"
                "Pandas Çıktısı: {pandas_output}\n\n"
                "Yanıt: "
            )

            # Eksik değişkenleri tanımla
            pandas_prompt = PromptTemplate(pandas_prompt_str)
            pandas_output_parser = PandasInstructionParser()

            # QueryPipeline oluştur
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

            fig, ax = plt.subplots()

            # QueryPipeline çalıştır
            response = qp.run(query_str=query_str)
            st.write("Response:")
            st.write(response.message.content)

            # Grafik oluşturma koşulu
            if "visualize" in response.message.content:
                st.session_state['fig'] = fig
                st.pyplot(fig=fig)
                plt.close(fig)
