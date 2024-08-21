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


#API Anahtarı (Size gönderilen linkten direkt sorgularınızı çalıştırabilirsiniz. Projede hali hazırda bir API bağlı.)
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit başlat
st.title("Verilerinizle Konuşun ve Görselleştirin!")
st.write("CSV dosyanızı yükleyin ve sorgunuzu girin.")

#LLM seçimi(İstenilen LLM modeli implemente edilebilir. Şu anlık sunucu ücretsiz olduğu için sadece OpenAI desteklenmektedir.)
model_type = st.selectbox("Kullanmak istediğiniz dil modelini seçin", ["OpenAI", "Llama-3.1:8B(Offline(Bellek sorunu))","Llama-3.1:70B(Offline(Bellek sorunu))"])
llm = OpenAI(model="gpt-3.5-turbo")

#CSV dosyasınızı yükleme kısmı
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:")
    st.write(df.head())

    #SentenceTransformer modelini yükleme bu kodu çalıştırdığınızda hugging face üzerinden direkt indirmeye başlayacaktır.
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    #Sütun adlarını vektörleştiriyoruz çünkü kullanıcının sorgusuna en uygun sütunları çekip özel sorgularımızı sadece o seçilen sütunlar üzerinden gerçekleştireceğiz.
    column_embeddings = model.encode(df.columns.tolist())

    #FAISS dizini oluşturma (Kosinüs Benzerliği kullanarak semantik arama için L2 Regulatörü ile normalize ediyoruz.)
    faiss.normalize_L2(column_embeddings)
    d = column_embeddings.shape[1]  #Vektör boyutunu belirliyoruz.
    index = faiss.IndexFlatIP(d)  #Inner Product (Kosinüs Benzerliği için kullanıyoruz.)

    index.add(column_embeddings)  #Tüm sütun vektörlerini vektörize veritabanımıza ekliyoruz.

    #Kullanıcının arayüz üzerinden bir sorgu girebilmesi için alan açıyoruz.
    query_str = st.text_input("Sorgunuzu girin", value="")

    if query_str:
        #Kullanıcıdan gelen sorguyu vektörleştirip ve normalize ediyoruz.
        query_embedding = model.encode([query_str])
        faiss.normalize_L2(query_embedding)

        #FAISS veritabanımızdan en yakın sütunları sorguluyoruz.
        k = 3  #k paramtresi ile en yakın kaç sütunu bulmak istediğimizi belirtiyoruz.
        distances, indices = index.search(query_embedding, k) #Vektörize veritabanında sorgumuzu gerçekleştiriyoruz.
        relevant_columns = [df.columns[i] for i in indices[0]] #Sütunları bir değişkene atayıp diğer adımda ekrana bastırıyoruz.
        
        st.write("İlgili sütunlar:", relevant_columns)

        #Sadece belirlenen sütunlar üzerinden bir sorgu üretebilmek adına sisteme sadece seçilen sütunları gönderiyoruz.
        if relevant_columns:
            selected_columns = relevant_columns  #En yakın sütunları veriyoruz.
            st.write(f"Sorgu sadece şu sütunlar üzerinden gerçekleştirilecek: {selected_columns}")
            
            #Pandas sorgusu için talimatları dinamik olarak oluşturmak üzere bir prompt engineering yapıyoruz.
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
                "Pandas Talimatları:\n{pandas_instructions}\n\n"
                "Pandas Çıktısı: {pandas_output}\n\n"
                "Yanıt: "
            )

            #Genel işleyişin bir ilerleme mimarisini kurmak için QueryPipeline oluşturuyoruz.
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
            #Gelen sorguyu oluşturduğumuz işleyiş mimarisine aktarıp sistemi çalıştırıyoruz.
            response = qp.run(query_str=query_str)
            st.write("Yanıt:")
            st.write(response.message.content)


#Her zaman bir grafik oluşturmaması için böylesine basit bir sorgu ile her sorguda grafik oluşturmamasını sağladık. Farkındayım kötü bir koşul işlemi ama inanılmaz derecede uğraştım fakat bir türlü llm1'in kod çıktısına ulaşamadım. O yüzden ne kadar optimize çalışmasa da bir çözüm üretmeye çalıştım.

            if "grafik" in response.message.content: 
                st.session_state['fig'] = fig
                st.pyplot(fig=fig)
                plt.close(fig)

