
# CSV Dosyası Yükleme ve Semantik Arama Uygulaması için Dokümantasyon

## Giriş

Bu proje, kullanıcıların CSV dosyalarını yükleyerek semantik arama yapmalarını sağlayan bir Streamlit uygulamasıdır. Uygulama, FAISS (Facebook AI Similarity Search) ve SentenceTransformers kütüphanelerini kullanarak CSV dosyasındaki sütun adlarını vektörleştirir ve kullanıcının sorgusuna en uygun sütunları belirler. Ardından, LLM (Large Language Model) desteğiyle seçilen sütunlar üzerinden sorgu gerçekleştirilir ve ilgili sonuçlar kullanıcılara sunulur.

## Gereksinimler

Aşağıdaki Python paketlerinin yüklü olduğundan emin olun:

- `llama-index==0.9.45.post1`
- `arize-phoenix>=4.0.0`
- `openai`
- `faiss-cpu`
- `sentence-transformers`
- `pandas`
- `streamlit`

Bu paketler, semantik arama, doğal dil işleme ve LLM tabanlı sorgulama işlemlerini gerçekleştirmek için gereklidir.

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install llama-index==0.9.45.post1 arize-phoenix>=4.0.0 openai faiss-cpu sentence-transformers pandas streamlit
   ```

2. OpenAI API anahtarınızı bir çevre değişkeni olarak tanımlayın:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Proje Mimarisi ve Akış

### 1. Streamlit Arayüzü

Uygulama, bir Streamlit arayüzü üzerinden çalışır. Kullanıcılar bir CSV dosyası yükler ve ardından sorgularını girerler. Kullanıcıya ayrıca kullanılacak dil modelini seçme imkanı sunulur.

### 2. LLM Seçimi

Kullanıcı, sorgularını işlemek için kullanılacak dil modelini seçer. Bu projede, açık kaynaklı LLM modelleri kullanılabilirdi fakat hızlı yanıt verebilmesi açısından OpenAI'nin GPT-3.5-turbo modeli tercih edilmiştir.

```python
model_type = st.selectbox("Kullanmak istediğiniz dil modelini seçin", ["OpenAI"])
llm = OpenAI(model="gpt-3.5-turbo")
```

### 3. CSV Dosyasının Yüklenmesi

Kullanıcı, bir CSV dosyası yükler. Yüklenen dosya `pandas` DataFrame formatına dönüştürülür ve ilk birkaç satırı Streamlit arayüzünde gösterilir.

```python
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Yüklenen veri:")
    st.write(df.head())
```

### 4. Semantik Arama İçin FAISS ve SentenceTransformers Kullanımı

Uygulama, CSV dosyasındaki sütun adlarını semantik olarak vektörleştirmek için SentenceTransformers kullanır. Bu vektörler FAISS dizinine eklenir ve kullanıcı sorgusu ile en alakalı sütunlar bulunur.

```python
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
column_embeddings = model.encode(df.columns.tolist())
faiss.normalize_L2(column_embeddings)
index = faiss.IndexFlatIP(d)
index.add(column_embeddings)
```

### 5. Sorgunun Çalıştırılması ve İlgili Sütunların Bulunması

Kullanıcının girdiği sorgu da aynı modelle vektörleştirilir ve FAISS kullanılarak en alakalı sütunlar bulunur. Bu sütunlar üzerinden Pandas sorgusu gerçekleştirilir.

```python
query_embedding = model.encode([query_str])
faiss.normalize_L2(query_embedding)
distances, indices = index.search(query_embedding, k)
relevant_columns = [df.columns[i] for i in indices[0]]
```

### 6. Pandas Sorgusu İçin Talimatların Dinamik Olarak Oluşturulması

Seçilen sütunlar üzerinden yapılacak Pandas sorgusu için talimatlar, OpenAI'nin GPT-3.5-turbo modeli tarafından oluşturulur. Bu talimatlar, CSV verilerinin yapısına göre dinamik olarak ayarlanır.

```python
instruction_str = (
    f"Yalnızca {', '.join(relevant_columns)} sütunlarını kullanarak sorguyu Pandas ile çalıştırılabilir Python koduna çevirin."
)
pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df[selected_columns].head(5)
)
```

### 7. QueryPipeline ile Sorgunun İşlenmesi

Llama Index `QueryPipeline` sınıfı kullanılarak, sorgu işleme aşamaları tanımlanır ve sonuçlar elde edilir. Bu aşamalar, kullanıcının sorgusunun işlenmesini, Pandas kodunun oluşturulmasını ve sonucun kullanıcıya sunulmasını içerir.

```python
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
response = qp.run(query_str=query_str)
st.write("Yanıt:")
st.write(response.message.content)
```

## Kullanım

1. Uygulamayı başlatmak için şu komutu çalıştırın:

   ```bash
   streamlit run CwcsvwRAG.py
   ```

2. Web arayüzünde bir CSV dosyası yükleyin.

3. Sorgunuzu girin. Örneğin, "Konya'dan başarılı kaç istek gelmiştir?" gibi bir sorgu yazabilirsiniz.

4. Uygulama, en uygun sütunları seçerek sorguyu çalıştıracak ve sonuçları gösterecektir.

5. https://zvjg288j9bxi9hgd8vdetb.streamlit.app ilgili linkten sistemin çalışır haline ulaştıktan sonra CSV dosyanızı yükleyip istediğiniz sorguları gerçekleştirebilirsiniz.

## Sonuç

Bu proje, LLM'ler ve semantik arama algoritmalarını kullanarak CSV dosyaları üzerinde esnek ve kullanıcı dostu sorgulama yapılmasını sağlar. FAISS ve SentenceTransformers ile sütunları semantik olarak anlamlandırarak, kullanıcının amacına en uygun verileri bulur ve sunar. Bu tür uygulamalar, büyük veri kümeleriyle çalışırken kullanıcıların daha verimli analiz yapmasını sağlar.
