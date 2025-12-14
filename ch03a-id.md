# Bab 3. RAG Bagian II: Mengobrol dengan Data Anda

Di bab sebelumnya, Anda mempelajari cara memproses data Anda dan membuat serta menyimpan penanaman (embeddings) dalam penyimpanan vektor (vector store). Dalam bab ini, Anda akan mempelajari cara mengambil penanaman dan bagian (chunks) dokumen yang paling relevan dengan efisien berdasarkan kueri pengguna. Ini memungkinkan Anda membangun petunjuk (prompt) yang berisi dokumen relevan sebagai konteks, meningkatkan akurasi keluaran akhir LLM.

Proses ini—yang melibatkan menanamkan kueri pengguna, mengambil dokumen serupa dari sumber data, dan kemudian meneruskannya sebagai konteks ke petunjuk yang dikirim ke LLM—secara formal dikenal sebagai [generasi yang diperkuat pengambilan (retrieval-augmented generation)](glossary01.xhtml#id545) (RAG).

RAG adalah komponen penting untuk membangun aplikasi LLM yang mendukung obrolan (chat-enabled) yang akurat, efisien, dan terkini. Dalam bab ini, Anda akan berkembang dari dasar ke strategi lanjutan untuk membangun sistem RAG yang efektif untuk berbagai sumber data (seperti penyimpanan vektor dan basis data) dan struktur data (terstruktur dan tidak terstruktur).

Tetapi pertama, mari kita definisikan RAG dan bahas manfaatnya.

## Memperkenalkan Generasi yang Diperkuat Pengambilan (Retrieval-Augmented Generation)

RAG [^1] adalah teknik yang digunakan untuk meningkatkan akurasi keluaran yang dihasilkan oleh LLM dengan menyediakan konteks dari sumber eksternal. Istilah ini awalnya diciptakan dalam makalah oleh peneliti Meta AI yang menemukan bahwa model yang diaktifkan RAG lebih faktual dan spesifik daripada model non-RAG.[^1]

Tanpa RAG, LLM hanya bergantung pada data yang dilatih sebelumnya, yang mungkin sudah kedaluwarsa. Misalnya, mari kita tanyakan ChatGPT sebuah pertanyaan tentang [peristiwa terkini](glossary01.xhtml#id548) dan lihat responsnya:

_Masukan_

```
Which country is the latest winner of the men's FIFA World Cup?
(Negara mana pemenang terbaru Piala Dunia FIFA putra?)
```

_Keluaran_

```
The most recent FIFA World Cup winner was France, who won the tournament in 2018.
(Pemenang Piala Dunia FIFA terbaru adalah Prancis, yang memenangkan turnamen pada 2018.)
```

Respons oleh LLM secara faktual salah dan kedaluwarsa. Pemenang terbaru pada saat publikasi buku ini adalah Argentina, yang memenangkan Piala Dunia pada 2022. Meskipun contoh pertanyaan ini mungkin sepele, halusinasi LLM dapat memiliki konsekuensi bencana jika jawabannya diandalkan untuk pemeriksaan fakta atau pengambilan keputusan penting.

Untuk memerangi masalah ini, kita perlu menyediakan LLM dengan informasi faktual dan terkini dari mana ia dapat merumuskan [keluaran yang akurat](glossary01.xhtml#id549). Melanjutkan dari contoh sebelumnya, mari kita buka halaman Wikipedia untuk [Piala Dunia FIFA](https://oreil.ly/LpLOV), salin paragraf pengantar, dan kemudian tambahkan sebagai [konteks](glossary01.xhtml#id550) ke petunjuk kita ke ChatGPT:

```
Which country is the latest winner of the men's FIFA World Cup?
(Negara mana pemenang terbaru Piala Dunia FIFA putra?)

See context below.
(Lihat konteks di bawah.)

The FIFA World Cup, often called the World Cup, is an international association
football competition among the senior men's national teams of the members of
the Fédération Internationale de Football Association (FIFA), the sport's
global governing body. The tournament has been held every four years since the
inaugural tournament in 1930, with the exception of 1942 and 1946 due to the
Second World War. The reigning champions are Argentina, who won their third
title at the 2022 tournament.
```

Perhatikan bahwa kalimat terakhir berisi konteks yang diperlukan yang dapat digunakan LLM untuk memberikan jawaban akurat. Berikut respons dari LLM:

```
The latest winner of the men's FIFA World Cup is Argentina, who won their third
title at the 2022 tournament.
(Pemenang terbaru Piala Dunia FIFA putra adalah Argentina, yang memenangkan gelar ketiga mereka di turnamen 2022.)
```

Karena konteks tambahan terkini yang disediakan, LLM mampu menghasilkan respons akurat terhadap petunjuk. Tetapi menyalin dan menempelkan informasi relevan sebagai konteks tidak praktis maupun dapat diskalakan untuk aplikasi AI produksi. Kita memerlukan sistem otomatis untuk mengambil informasi relevan berdasarkan kueri pengguna, menambahkannya sebagai konteks ke petunjuk, dan kemudian mengeksekusi permintaan generasi ke LLM.

### Mengambil Dokumen Relevan

Sebuah [sistem RAG](glossary01.xhtml#RAGrelevant03) untuk [aplikasi](glossary01.xhtml#Drelevant03) AI biasanya mengikuti [tiga tahap inti](glossary01.xhtml#id551):

- **Pengindeksan (Indexing)**
  [Tahap](glossary01.xhtml#id552) ini melibatkan pemrosesan awal sumber data eksternal dan menyimpan penanaman yang merepresentasikan data dalam penyimpanan vektor di mana mereka dapat dengan mudah diambil.

- **Pengambilan (Retrieval)**
  Tahap ini melibatkan mengambil penanaman dan data relevan yang disimpan dalam penyimpanan vektor berdasarkan kueri pengguna.

- **Generasi (Generation)**
  Tahap ini melibatkan sintesis petunjuk asli dengan dokumen relevan yang diambil sebagai satu petunjuk akhir yang dikirim ke model untuk prediksi.

Tiga tahap dasar terlihat seperti [Gambar 3-1](#ch03_figure_1_1736545666780536).

![Diagram dokumen Deskripsi dihasilkan otomatis](Images/lelc_0301.png "Tahap kunci RAG")

[Gambar 3-1. Tahap kunci RAG](#ch03_figure_1_1736545666780536)

Tahap pengindeksan dari proses ini dibahas secara ekstensif di [Bab 2](ch02.xhtml#ch02_rag_part_i_indexing_your_data_1736545662500927), di mana Anda mempelajari cara menggunakan pemuat dokumen (document loaders), pembagi teks (text splitters), penanaman (embeddings), dan penyimpanan vektor (vector stores).

Mari kita jalankan contoh dari awal lagi, dimulai dengan tahap pengindeksan:

_Python_ [^2]

```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# embed each chunk and insert it into the vector store
model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents, model, connection=connection)
```

_JavaScript_ [^3]

```javascript
import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { OpenAIEmbeddings } from "@langchain/openai"
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector"

// Load the document, split it into chunks
const loader = new TextLoader("./test.txt")
const raw_docs = await loader.load()
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})
const docs = await splitter.splitDocuments(docs)

// embed each chunk and insert it into the vector store
const model = new OpenAIEmbeddings()
const db = await PGVectorStore.fromDocuments(docs, model, {
  postgresConnectionOptions: {
    connectionString:
      "postgresql://langchain:langchain@localhost:6024/langchain",
  },
})
```

[Bab 2](ch02.xhtml#ch02_rag_part_i_indexing_your_data_1736545662500927) memiliki detail lebih lanjut tentang tahap pengindeksan.

Tahap pengindeksan sekarang selesai. Untuk mengeksekusi tahap pengambilan, kita perlu melakukan perhitungan pencarian kesamaan—seperti [kesamaan kosinus (cosine similarity)](glossary01.xhtml#id555)—antara kueri pengguna dan penanaman yang disimpan kita, sehingga bagian dari dokumen terindeks kita yang relevan diambil (lihat [Gambar 3-2](#ch03_figure_2_1736545666780585)).

![Screenshot 2024-02-12 at 1.36.56 PM.png](Images/lelc_0302.png "Contoh alur mengindeks dokumen bersama pengambilan dokumen relevan dari penyimpanan vektor; kotak Hierarchical Navigable Small World (HNSW) menggambarkan perhitungan kesamaan dokumen terhadap kueri pengguna")

[Gambar 3-2. Contoh alur mengindeks dokumen bersama pengambilan dokumen relevan dari penyimpanan vektor; kotak Hierarchical Navigable Small World (HNSW) menggambarkan perhitungan kesamaan dokumen terhadap kueri pengguna](#ch03_figure_2_1736545666780585)

[Gambar 3-2](#ch03_figure_2_1736545666780585) mengilustrasikan langkah-langkah dalam [proses pengambilan](glossary01.xhtml#id556):

1.  Konversi kueri pengguna menjadi penanaman.
2.  Hitung penanaman dalam penyimpanan vektor yang paling mirip dengan kueri pengguna.
3.  Ambil penanaman dokumen relevan dan bagian teks yang sesuai.

Kita dapat merepresentasikan langkah-langkah ini secara pemrograman menggunakan LangChain sebagai berikut:

_Python_ [^4]

```python
# create retriever
retriever = db.as_retriever()

# fetch relevant documents
docs = retriever.invoke("""Who are the key figures in the ancient greek
    history of philosophy?""")
```

_JavaScript_ [^5]

```javascript
// create retriever
const retriever = db.asRetriever()

// fetch relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
  greek history of philosophy?`)
```

Perhatikan bahwa kita menggunakan metode penyimpanan vektor yang belum pernah Anda lihat sebelumnya: `as_retriever`. Fungsi ini mengabstraksi logika menanamkan kueri pengguna dan perhitungan pencarian kesamaan yang mendasari yang dilakukan oleh penyimpanan vektor untuk mengambil dokumen relevan.

Ada juga [argumen k](glossary01.xhtml#id561), yang menentukan jumlah dokumen relevan yang akan diambil dari penyimpanan vektor. Misalnya:

_Python_

```python
# create retriever with k=2
retriever = db.as_retriever(search_kwargs={"k": 2})

# fetch the 2 most relevant documents
docs = retriever.invoke("""Who are the key figures in the ancient greek history
    of philosophy?""")
```

_JavaScript_

```javascript
// create retriever with k=2
const retriever = db.asRetriever({ k: 2 })

// fetch the 2 most relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
  greek history of philosophy?`)
```

Dalam contoh ini, argumen `k` ditentukan sebagai 2. Ini memberi tahu penyimpanan vektor untuk mengembalikan dua dokumen paling relevan berdasarkan kueri pengguna.

Mungkin tampak kontraintuitif untuk menggunakan nilai `k` rendah, tetapi mengambil lebih banyak dokumen tidak selalu lebih baik. Semakin banyak dokumen yang diambil, semakin lambat aplikasi Anda akan berkinerja, semakin besar petunjuk (dan biaya generasi terkait) akan menjadi, dan semakin besar kemungkinan mengambil bagian teks yang berisi informasi tidak relevan, yang akan menyebabkan LLM berhalusinasi.

Sekarang setelah kita menyelesaikan tahap pengambilan sistem RAG, mari kita lanjutkan ke tahap generasi akhir.

### Menghasilkan Prediksi LLM Menggunakan Dokumen Relevan

Setelah [kita](glossary01.xhtml#genstg03) mengambil dokumen relevan berdasarkan kueri pengguna, langkah terakhir adalah menambahkannya ke petunjuk asli sebagai konteks dan kemudian memanggil model untuk menghasilkan keluaran akhir ([Gambar 3-3](#ch03_figure_3_1736545666780616)).

![Diagram diagram Deskripsi dihasilkan otomatis](Images/lelc_0303.png "Contoh alur mendemonstrasikan pengindeksan dokumen, pengambilan dokumen relevan dari penyimpanan vektor, dan penyertaan dokumen yang diambil sebagai konteks dalam petunjuk LLM")

[Gambar 3-3. Contoh alur mendemonstrasikan pengindeksan dokumen, pengambilan dokumen relevan dari penyimpanan vektor, dan penyertaan dokumen yang diambil sebagai konteks dalam petunjuk LLM](#ch03_figure_3_1736545666780616)

Berikut contoh kode melanjutkan dari contoh kita sebelumnya:

_Python_ [^6]

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the question based only on
    the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = prompt | llm

# fetch relevant documents
docs = retriever.get_relevant_documents("""Who are the key figures in the
    ancient greek history of philosophy?""")

# run
chain.invoke({"context": docs,"question": """Who are the key figures in the
    ancient greek history of philosophy?"""})
```

_JavaScript_ [^7]

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"

const retriever = db.asRetriever()

const prompt = ChatPromptTemplate.fromTemplate(`Answer the question based only 
  on the following context:
{context}

Question: {question}
`)

const llm = new ChatOpenAI({ temperature: 0, modelName: "gpt-3.5-turbo" })

const chain = prompt.pipe(llm)

// fetch relevant documents
const docs = await retriever.invoke(`Who are the key figures in the ancient 
  greek history of philosophy?`)

await chain.invoke({
  context: docs,
  question: `Who are the key figures in the 
  ancient greek history of philosophy?`,
})
```

Perhatikan perubahan berikut:

- Kita menerapkan variabel `context` dan `question` dinamis ke petunjuk kita, yang memungkinkan kita mendefinisikan `ChatPromptTemplate` yang dapat digunakan model untuk menghasilkan respons.
- Kita mendefinisikan antarmuka `ChatOpenAI` untuk bertindak sebagai LLM kita. Suhu (temperature) diatur ke 0 untuk menghilangkan kreativitas dalam keluaran dari model.
- Kita membuat rantai (chain) untuk menyusun petunjuk dan LLM. Pengingat: operator `|` (atau metode `pipe` di JS) mengambil keluaran `prompt` dan menggunakannya sebagai masukan ke `llm`.
- Kita `invoke` rantai dengan meneruskan variabel `context` (dokumen relevan kita yang diambil) dan pertanyaan pengguna untuk menghasilkan keluaran akhir.

Kita dapat merangkum logika pengambilan ini dalam satu fungsi:

_Python_

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template("""Answer the question based only on
    the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
qa.invoke("Who are the key figures in the ancient greek history of philosophy?")
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { RunnableLambda } from "@langchain/core/runnables"

const retriever = db.asRetriever()

const prompt = ChatPromptTemplate.fromTemplate(`Answer the question based only 
  on the following context:
{context}

Question: {question}
`)

const llm = new ChatOpenAI({ temperature: 0, modelName: "gpt-3.5-turbo" })

const qa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input })
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Who are the key figures in the ancient greek history of 
  philosophy?`)
```

Perhatikan bagaimana kita sekarang memiliki fungsi runnable `qa` baru yang dapat dipanggil hanya dengan pertanyaan dan mengurus untuk pertama mengambil dokumen relevan untuk konteks, memformatnya ke dalam petunjuk, dan akhirnya menghasilkan jawaban. Dalam kode Python, dekorator `@chain` mengubah fungsi menjadi rantai runnable. Gagasan merangkum banyak langkah ke dalam satu fungsi ini akan menjadi kunci untuk membangun aplikasi menarik dengan LLM.

Anda juga dapat mengembalikan dokumen yang diambil untuk inspeksi lebih lanjut:

_Python_

```python
@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return {"answer": answer, "docs": docs}
```

_JavaScript_

```javascript
const qa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input })
  // generate answer
  const answer = await llm.invoke(formatted)
  return { answer, docs }
})
```

Selamat! Anda sekarang telah membangun sistem RAG dasar untuk menggerakkan aplikasi AI untuk penggunaan pribadi.

Namun, aplikasi AI siap produksi yang digunakan oleh banyak pengguna memerlukan [sistem RAG](glossary01.xhtml#id564) yang lebih canggih. Untuk membangun sistem RAG yang kokoh, kita perlu menjawab pertanyaan berikut secara efektif:

- Bagaimana kita menangani variabilitas dalam kualitas masukan pengguna?
- Bagaimana kita merutekan kueri untuk mengambil data relevan dari berbagai sumber data?
- Bagaimana kita mengubah bahasa alami ke bahasa kueri dari sumber data target?
- Bagaimana kita mengoptimalkan proses pengindeksan kita, yaitu penanaman, pembagian teks?

Selanjutnya kita akan membahas strategi berbasis penelitian terbaru untuk menjawab pertanyaan-pertanyaan ini dan membangun sistem RAG siap produksi. Strategi ini dapat diringkas dalam [Gambar 3-4](#ch03_figure_4_1736545666780656).

![Diagram proses pengembangan perangkat lunak Deskripsi dihasilkan otomatis](Images/lelc_0304.png "Strategi efektif untuk mengoptimalkan akurasi sistem RAG Anda")

[Gambar 3-4. Strategi efektif untuk mengoptimalkan akurasi sistem RAG Anda](#ch03_figure_4_1736545666780656)

> **Catatan**
> Semua blok kode di sisa bab ini menggunakan penyimpanan vektor yang kita siapkan di awal bab.

## Transformasi Kueri (Query Transformation)

Salah satu [dari](glossary01.xhtml#RAGqt03) masalah utama dengan sistem RAG dasar adalah bahwa itu terlalu bergantung pada kualitas kueri pengguna untuk menghasilkan keluaran yang akurat. Dalam pengaturan produksi, pengguna kemungkinan akan menyusun kueri mereka dengan cara yang tidak lengkap, ambigu, atau dikatakan buruk yang mengarah ke [halusinasi model](glossary01.xhtml#id570).

_[Transformasi kueri (query transformation)](glossary01.xhtml#id571)_ adalah subset strategi yang dirancang untuk memodifikasi masukan pengguna untuk menjawab pertanyaan masalah RAG pertama: Bagaimana kita menangani variabilitas dalam kualitas masukan pengguna? [Gambar 3-5](#ch03_figure_5_1736545666780682) mengilustrasikan rentang strategi transformasi kueri, mulai dari yang membuat masukan pengguna lebih atau kurang abstrak untuk menghasilkan keluaran LLM yang akurat. Bagian selanjutnya dimulai dengan strategi tengah.

![Diagram pertanyaan Deskripsi dihasilkan otomatis](Images/lelc_0305.png "Berbagai metode untuk mengubah kueri pengguna berdasarkan tingkat abstraksi")

[Gambar 3-5. Berbagai metode untuk mengubah kueri pengguna berdasarkan tingkat abstraksi](#ch03_figure_5_1736545666780682)

### Tulis Ulang-Ambil-Baca (Rewrite-Retrieve-Read)

[Strategi Tulis Ulang-Ambil-Baca (Rewrite-Retrieve-Read strategy)](glossary01.xhtml#RRRstrat03) yang diusulkan oleh tim Microsoft Research hanya meminta LLM untuk menulis ulang kueri pengguna sebelum melakukan pengambilan.[^8] Untuk menggambarkan, mari kembali ke rantai yang kita bangun di bagian sebelumnya, kali ini dipanggil dengan kueri pengguna yang dikatakan buruk:

_Python_

```python
@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.get_relevant_documents(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

qa.invoke("""Today I woke up and brushed my teeth, then I sat down to read the
    news. But then I forgot the food on the cooker. Who are some key figures in
    the ancient greek history of philosophy?""")
```

_JavaScript_

```javascript
const qa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retriever.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input })
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Today I woke up and brushed my teeth, then I sat down to read 
  the news. But then I forgot the food on the cooker. Who are some key figures 
  in the ancient greek history of philosophy?`)
```

_Keluaran_ (ingat: jika Anda menjalankan ulang, keluaran Anda mungkin berbeda dari ini):

```
Based on the given context, there is no information provided.
(Berdasarkan konteks yang diberikan, tidak ada informasi yang disediakan.)
```

Model gagal menjawab pertanyaan karena terganggu oleh informasi tidak relevan yang disediakan dalam kueri pengguna.

Sekarang mari kita terapkan petunjuk Tulis Ulang-Ambil-Baca:

_Python_ [^9]

```python
rewrite_prompt = ChatPromptTemplate.from_template("""Provide a better search
    query for web search engine to answer the given question, end the queries
    with '**'. Question: {x} Answer:""")

def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")

rewriter = rewrite_prompt | llm | parse_rewriter_output

@chain
def qa_rrr(input):
    # rewrite the query
    new_query = rewriter.invoke(input)
    # fetch relevant documents
    docs = retriever.get_relevant_documents(new_query)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
qa_rrr.invoke("""Today I woke up and brushed my teeth, then I sat down to read
    the news. But then I forgot the food on the cooker. Who are some key
    figures in the ancient greek history of philosophy?""")
```

_JavaScript_ [^10]

```javascript
const rewritePrompt = ChatPromptTemplate.fromTemplate(`Provide a better search 
  query for web search engine to answer the given question, end the queries 
  with '**'. Question: {question} Answer:`)

const rewriter = rewritePrompt.pipe(llm).pipe((message) => {
  return message.content.replaceAll('"', "").replaceAll("**")
})

const qa = RunnableLambda.from(async (input) => {
  const newQuery = await rewriter.invoke({ question: input })
  // fetch relevant documents
  const docs = await retriever.invoke(newQuery)
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input })
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await qa.invoke(`Today I woke up and brushed my teeth, then I sat down to read 
  the news. But then I forgot the food on the cooker. Who are some key 
  figures in the ancient greek history of philosophy?`)
```

_Keluaran:_

```
Based on the given context, some key figures in the ancient greek history of
philosophy include: Themistocles (an Athenian statesman), Pythagoras, and Plato.
(Berdasarkan konteks yang diberikan, beberapa tokoh kunci dalam sejarah filsafat Yunani kuno termasuk: Themistocles (seorang negarawan Athena), Pythagoras, dan Plato.)
```

Perhatikan bahwa kita telah meminta LLM untuk menulis ulang kueri awal pengguna yang terganggu menjadi yang jauh lebih jelas, dan kueri yang lebih fokus itulah yang diteruskan ke pengambil (retriever) untuk mengambil dokumen paling relevan. Catatan: teknik ini dapat digunakan dengan metode pengambilan apa pun, baik itu penyimpanan vektor seperti yang kita miliki di sini atau, misalnya, alat pencarian web. Kelemahan pendekatan ini adalah bahwa itu memperkenalkan latensi tambahan ke rantai Anda, karena sekarang kita perlu melakukan dua panggilan LLM secara berurutan.

### Pengambilan Multi-Kueri (Multi-Query Retrieval)

Sebuah [kueri tunggal pengguna](glossary01.xhtml#QTmulti03) dapat tidak cukup untuk menangkap cakupan penuh informasi yang diperlukan untuk menjawab kueri secara komprehensif. [Strategi pengambilan multi-kueri (multi-query retrieval strategy)](glossary01.xhtml#multistrat03) mengatasi masalah ini dengan menginstruksikan LLM untuk menghasilkan beberapa kueri berdasarkan kueri awal pengguna, mengeksekusi pengambilan paralel dari setiap kueri dari sumber data dan kemudian memasukkan hasil yang diambil sebagai konteks petunjuk untuk menghasilkan keluaran model akhir. [Gambar 3-6](#ch03_figure_6_1736545666780704) mengilustrasikan.

![Diagram diagram dokumen Deskripsi dihasilkan otomatis dengan kepercayaan sedang](Images/lelc_0306.png "Demonstrasi strategi pengambilan multi-kueri")

[Gambar 3-6. Demonstrasi strategi pengambilan multi-kueri](#ch03_figure_6_1736545666780704)

Strategi ini sangat berguna untuk kasus penggunaan di mana satu pertanyaan mungkin bergantung pada beberapa perspektif untuk memberikan jawaban yang komprehensif.

Berikut contoh kode pengambilan multi-kueri dalam aksi:

_Python_ [^11]

```python
from langchain.prompts import ChatPromptTemplate

perspectives_prompt = ChatPromptTemplate.from_template("""You are an AI language
    model assistant. Your task is to generate five different versions of the
    given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to
    help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by
    newlines. Original question: {question}""")

def parse_queries_output(message):
    return message.content.split('\n')

query_gen = perspectives_prompt | llm | parse_queries_output
```

_JavaScript_ [^12]

```javascript
const perspectivesPrompt = ChatPromptTemplate.fromTemplate(`You are an AI 
  language model assistant. Your task is to generate five different versions 
  of the given user question to retrieve relevant documents from a vector 
  database. By generating multiple perspectives on the user question, your 
  goal is to help the user overcome some of the limitations of the 
  distance-based similarity search. Provide these alternative questions 
  separated by newlines. Original question: {question}`)

const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split("\n")
})
```

Perhatikan bahwa templat petunjuk dirancang untuk menghasilkan variasi pertanyaan berdasarkan kueri awal pengguna.

Selanjutnya kita mengambil daftar kueri yang dihasilkan, mengambil dokumen paling relevan untuk masing-masing secara paralel, dan kemudian menggabungkan untuk mendapatkan gabungan unik dari semua dokumen relevan yang diambil:

_Python_

```python
def get_unique_union(document_lists):
    # Flatten list of lists, and dedupe them
    deduped_docs = {
        doc.page_content: doc
        for sublist in document_lists for doc in sublist
    }
    # return a flat list of unique docs
    return list(deduped_docs.values())

retrieval_chain = query_gen | retriever.batch | get_unique_union
```

_JavaScript_

```javascript
const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe((documentLists) => {
    const dedupedDocs = {}
    documentLists.flat().forEach((doc) => {
      dedupedDocs[doc.pageContent] = doc
    })
    return Object.values(dedupedDocs)
  })
```

Karena kita mengambil dokumen dari pengambil yang sama dengan beberapa kueri (terkait), kemungkinan setidaknya beberapa di antaranya berulang. Sebelum menggunakannya sebagai konteks untuk menjawab pertanyaan, kita perlu mendeduplikasinya, untuk berakhir dengan satu instance dari masing-masing. Di sini kita mendeduplikasi dokumen dengan menggunakan konten mereka (string) sebagai kunci dalam kamus (atau objek di JS), karena kamus hanya dapat berisi satu entri untuk setiap kunci. Setelah kita mengulangi semua dokumen, kita cukup mendapatkan semua nilai kamus, yang sekarang bebas dari duplikat.

Perhatikan penggunaan kita juga `.batch`, yang menjalankan semua kueri yang dihasilkan secara paralel dan mengembalikan daftar hasil—dalam hal ini, daftar daftar dokumen, yang kemudian kita ratakan dan deduplikasi seperti dijelaskan sebelumnya.

Langkah terakhir adalah membangun petunjuk, termasuk pertanyaan pengguna dan dokumen relevan yang diambil yang digabungkan, dan antarmuka model untuk menghasilkan prediksi:

_Python_

```python
prompt = ChatPromptTemplate.from_template("""Answer the following question based
    on this context:

{context}

Question: {question}
""")

@chain
def multi_query_qa(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

# run
multi_query_qa.invoke("""Who are some key figures in the ancient greek history
    of philosophy?""")
```

_JavaScript_

```javascript
const prompt = ChatPromptTemplate.fromTemplate(`Answer the following 
  question based on this context:

{context}

Question: {question}
`)

const multiQueryQa = RunnableLambda.from(async (input) => {
  // fetch relevant documents
  const docs = await retrievalChain.invoke(input)
  // format prompt
  const formatted = await prompt.invoke({ context: docs, question: input })
  // generate answer
  const answer = await llm.invoke(formatted)
  return answer
})

await multiQueryQa.invoke(`Who are some key figures in the ancient greek 
  history of philosophy?`)
```

Perhatikan bagaimana ini tidak berbeda jauh dari rantai QA kita sebelumnya, karena semua logika baru untuk pengambilan multi-kueri terkandung dalam `retrieval_chain`. Ini adalah kunci untuk memanfaatkan teknik-teknik ini dengan baik—mengimplementasikan setiap teknik sebagai rantai mandiri (dalam hal ini, `retrieval_chain`), yang memudahkan untuk mengadopsinya dan bahkan menggabungkannya.

### RAG-Fusion

[Strategi RAG-Fusion](glossary01.xhtml#QTragfusion03) berbagi kesamaan dengan strategi pengambilan multi-kueri, kecuali kita akan menerapkan langkah peringkat ulang (reranking) akhir ke semua dokumen yang diambil.[^13] Langkah peringkat ulang ini menggunakan algoritma _[fusi peringkat timbal balik (reciprocal rank fusion)](glossary01.xhtml#rrfalg03)_ (RRF), yang melibatkan menggabungkan peringkat dari hasil pencarian yang berbeda untuk menghasilkan peringkat tunggal yang disatukan. Dengan menggabungkan peringkat dari kueri yang berbeda, kita menarik dokumen paling relevan ke atas daftar akhir. RRF cocok untuk menggabungkan hasil dari kueri yang mungkin memiliki skala atau distribusi skor yang berbeda.

Mari kita demonstrasikan RAG-Fusion dalam kode. Pertama, kita buat petunjuk mirip dengan strategi pengambilan multi-kueri untuk menghasilkan daftar kueri berdasarkan kueri pengguna:

_Python_ [^14]

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful
    assistant that generates multiple search queries based on a single input
    query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):""")

def parse_queries_output(message):
    return message.content.split('\n')

llm = ChatOpenAI(temperature=0)

query_gen = prompt_rag_fusion | llm | parse_queries_output
```

_JavaScript_ [^15]

```javascript
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { ChatOpenAI } from "@langchain/openai"
import { RunnableLambda } from "@langchain/core/runnables"

const perspectivesPrompt = ChatPromptTemplate.fromTemplate(`You are a helpful 
  assistant that generates multiple search queries based on a single input 
  query. \n
  Generate multiple search queries related to: {question} \n
  Output (4 queries):`)

const queryGen = perspectivesPrompt.pipe(llm).pipe((message) => {
  return message.content.split("\n")
})
```

Setelah kita menghasilkan kueri kita, kita mengambil dokumen relevan untuk setiap kueri dan meneruskannya ke fungsi untuk [meringkat ulang](glossary01.xhtml#id583) (yaitu, _mengurutkan ulang_ sesuai relevansi) daftar akhir dokumen relevan.

Fungsi `reciprocal_rank_fusion` mengambil daftar hasil pencarian dari setiap kueri, jadi daftar daftar dokumen, di mana setiap daftar dalam dokumen diurutkan berdasarkan relevansinya dengan kueri itu. Algoritma RRF kemudian menghitung skor baru untuk setiap dokumen berdasarkan peringkat (atau posisi) mereka dalam daftar yang berbeda dan mengurutkannya untuk membuat daftar peringkat ulang akhir.

Setelah menghitung skor gabungan (fused), fungsi mengurutkan dokumen dalam urutan menurun dari skor ini untuk mendapatkan daftar peringkat ulang akhir, yang kemudian dikembalikan:

_Python_

```python
def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents
       and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list,
        # with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Use the document contents as the key for uniqueness
            doc_str = doc.page_content
            # If the document hasn't been seen yet,
            # - initialize score to 0
            # - save it for later
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            # Update the score of the document using the RRF formula:
            # 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    # to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d : fused_scores[d], reverse=True
    )
    # retrieve the corresponding doc for each doc_str
    return [
        documents[doc_str]
        for doc_str in reranked_doc_strs
    ]

retrieval_chain = generate_queries | retriever.batch | reciprocal_rank_fusion
```

_JavaScript_

```javascript
function reciprocalRankFusion(results, k = 60) {
  // Initialize a dictionary to hold fused scores for each document
  // Documents will be keyed by their contents to ensure uniqueness
  const fusedScores = {}
  const documents = {}

  results.forEach((docs) => {
    docs.forEach((doc, rank) => {
      // Use the document contents as the key for uniqueness
      const key = doc.pageContent
      // If the document hasn't been seen yet,
      // - initialize score to 0
      // - save it for later
      if (!(key in fusedScores)) {
        fusedScores[key] = 0
        documents[key] = 0
      }
      // Update the score of the document using the RRF formula:
      // 1 / (rank + k)
      fusedScores[key] += 1 / (rank + k)
    })
  })

  // Sort the documents based on their fused scores in descending order
  // to get the final reranked results
  const sorted = Object.entries(fusedScores).sort((a, b) => b[1] - a[1])
  // retrieve the corresponding doc for each key
  return sorted.map(([key]) => documents[key])
}

const retrievalChain = queryGen
  .pipe(retriever.batch.bind(retriever))
  .pipe(reciprocalRankFusion)
```

Perhatikan bahwa fungsi juga mengambil [parameter k](glossary01.xhtml#id585), yang menentukan seberapa banyak pengaruh dokumen dalam set hasil setiap kueri terhadap daftar akhir dokumen. Nilai yang lebih tinggi menunjukkan bahwa dokumen yang lebih rendah peringkatnya memiliki lebih banyak pengaruh.

Akhirnya, kita gabungkan rantai pengambilan baru kita (sekarang menggunakan RRF) dengan rantai penuh yang telah kita lihat sebelumnya:

[^1]: Patrick Lewis et al., ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://oreil.ly/Qzd2K), arXiv, 12 April 2021.
[^2]: Dari contoh [Python](glossary01.xhtml#id553) [pengindeksan data](glossary01.xhtml#id553)
[^3]: Dari contoh [JavaScript](glossary01.xhtml#id554) [pengindeksan data](glossary01.xhtml#id554)
[^4]: Dari [Python](glossary01.xhtml#id559) untuk [penanaman, mengambil relevan](glossary01.xhtml#id559)
[^5]: Dari [JavaScript](glossary01.xhtml#id560) untuk [penanaman, mengambil relevan](glossary01.xhtml#id560)
[^6]: Dari [Python](glossary01.xhtml#Pgenerate03) untuk [prediksi model, memanggil](glossary01.xhtml#Pgenerate03)
[^7]: Dari [JavaScript](glossary01.xhtml#JSllmpred03) untuk [prediksi model, memanggil](glossary01.xhtml#JSllmpred03)
[^8]: Xinbei Ma et al., ["Query Rewriting for Retrieval-Augmented Large Language Models"](https://oreil.ly/zyw5E), arXiv, 23 Oktober 2023. Riset ditugaskan oleh Microsoft Research Asia.
[^9]: Dari [Python](glossary01.xhtml#id573) untuk [strategi Tulis Ulang-Ambil-Baca](glossary01.xhtml#id573)
[^10]: Dari [JavaScript](glossary01.xhtml#id574) untuk [strategi Tulis Ulang-Ambil-Baca](glossary01.xhtml#id574)
[^11]: Dari [Python](glossary01.xhtml#Pmulti03) dalam [pengambilan multi-kueri](glossary01.xhtml#Pmulti03)
[^12]: Dari [JavaScript](glossary01.xhtml#JSmulti03) dalam [pengambilan multi-kueri](glossary01.xhtml#JSmulti03)
[^13]: Zackary Rackauckas, ["RAG-Fusion: A New Take on Retrieval-Augmented Generation"](https://oreil.ly/k7TTY), arXiv, 21 Februari 2024. Dari _International Journal on Natural Language Computing_, vol. 13, no. 1 (Februari 2024).
[^14]: Dari [Python](glossary01.xhtml#Pragfus03) untuk [strategi RAG-Fusion](glossary01.xhtml#Pragfus03)
[^15]: Dari [JavaScript](glossary01.xhtml#JSrag03) untuk [strategi RAG-Fusion](glossary01.xhtml#JSrag03)
