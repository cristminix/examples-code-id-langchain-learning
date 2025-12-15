# Bab 1. Dasar-dasar LLM dengan LangChain

[Pendahuluan](preface01.xhtml#pr01_preface_1736545679069216) telah memberikanmu gambaran tentang kekuatan pemberian petunjuk (prompt) pada LLM, di mana kita melihat langsung dampak dari berbagai teknik pemberian petunjuk terhadap keluaran dari LLM, terutama ketika digabungkan dengan bijak. Tantangan dalam membuat aplikasi LLM yang baik sebenarnya terletak pada cara membangun petunjuk yang dikirim ke model dan mengolah prediksi model untuk menghasilkan keluaran yang akurat (lihat [Gambar 1-1](#ch01_figure_1_1736545659763063)).

![Gambar 1-1. Tantangan dalam menjadikan LLM sebagai bagian yang berguna dari aplikasi Anda](Images/lelc_0101.png){width=600 height=105}

Jika Anda bisa menyelesaikan masalah ini, Anda sudah berada di jalur yang tepat untuk membangun aplikasi LLM, baik yang sederhana maupun kompleks. Dalam bab ini, Anda akan mempelajari lebih lanjut tentang bagaimana blok bangunan LangChain dipetakan ke konsep LLM dan bagaimana, ketika digabungkan secara efektif, blok-blok tersebut memungkinkan Anda membangun aplikasi LLM. Namun sebelumnya, kotak samping ["Mengapa LangChain?"](#ch01_why_langchain_1736545659776355) adalah pengantar singkat mengapa kami berpikir berguna untuk menggunakan LangChain dalam membangun aplikasi LLM.

## Mengapa LangChain?

Tentu saja Anda bisa membangun aplikasi LLM tanpa LangChain. Alternatif yang paling jelas adalah menggunakan kit pengembangan perangkat lunak (SDK)—paket yang mengekspos metode dari API HTTP mereka sebagai fungsi dalam bahasa pemrograman pilihan Anda—dari penyedia LLM yang pertama kali Anda coba (misalnya, OpenAI). Kami berpikir mempelajari LangChain akan membuahkan hasil dalam jangka pendek maupun jangka panjang karena faktor-faktor berikut:

**Pola umum yang sudah dibuat sebelumnya**
: LangChain hadir dengan implementasi referensi dari pola aplikasi LLM yang paling umum (kami menyebutkan beberapa di antaranya di [Pendahuluan](preface01.xhtml#pr01_preface_1736545679069216): rantai pemikiran (chain-of-thought), pemanggilan alat (tool calling), dan lainnya). Ini adalah cara tercepat untuk memulai dengan LLM dan seringkali semua yang Anda butuhkan. Kami menyarankan untuk memulai aplikasi baru apa pun dari pola-pola ini dan memeriksa apakah hasil yang langsung didapat sudah cukup baik untuk kasus penggunaan Anda. Jika tidak, lihat poin berikutnya untuk bagian lain dari pustaka LangChain.

**Blok bangunan yang dapat ditukar**
: Ini adalah komponen-komponen yang dapat dengan mudah diganti dengan alternatif. Setiap komponen (seperti LLM, model obrolan (chat model), pengurai keluaran (output parser), dan sebagainya—lebih lanjut tentang ini sebentar lagi) mengikuti spesifikasi bersama, yang membuat aplikasi Anda tahan masa depan (future-proof). Saat kemampuan baru dirilis oleh penyedia model dan saat kebutuhan Anda berubah, Anda dapat mengembangkan aplikasi tanpa menulis ulang setiap kali.

Sepanjang buku ini kami menggunakan komponen utama berikut dalam contoh kode:

- LLM/model obrolan: OpenAI
- Penanaman (embeddings): OpenAI
- Penyimpanan vektor (vector store): PGVector

Anda dapat menukar masing-masing komponen ini dengan alternatif apa pun yang tercantum di halaman-halaman berikut:

**Model obrolan**
: Lihat [dokumentasi LangChain](https://oreil.ly/8Qlnb). Jika Anda tidak ingin menggunakan OpenAI (API komersial), kami menyarankan [Anthropic](https://oreil.ly/XdGfD) sebagai alternatif komersial atau [Ollama](https://oreil.ly/eKy6-) sebagai alternatif sumber terbuka.

**Penanaman (embeddings)**
: Lihat [dokumentasi LangChain](https://oreil.ly/sKpfM). Jika Anda tidak ingin menggunakan OpenAI (API komersial), kami menyarankan [Cohere](https://oreil.ly/o1D0C) sebagai alternatif komersial atau [Ollama](https://oreil.ly/FarfL) sebagai alternatif sumber terbuka.

**Penyimpanan vektor (vector stores)**
: Lihat [dokumentasi LangChain](https://oreil.ly/q3RF1). Jika Anda tidak ingin menggunakan PGVector (ekstensi sumber terbuka untuk basis data SQL populer Postgres), kami menyarankan menggunakan [Weaviate](https://oreil.ly/XqlYa) (penyimpanan vektor khusus) atau [OpenSearch](https://oreil.ly/1s357) (fitur pencarian vektor yang merupakan bagian dari basis data pencarian populer).

Upaya ini melampaui, misalnya, semua LLM memiliki metode yang sama, dengan argumen dan nilai kembalian yang serupa. Mari kita lihat contoh model obrolan dan dua penyedia LLM populer, OpenAI dan Anthropic. Keduanya memiliki API obrolan yang menerima _pesan obrolan_ (didefinisikan secara longgar sebagai objek dengan string tipe dan string konten) dan mengembalikan pesan baru yang dihasilkan oleh model. Tetapi jika Anda mencoba menggunakan kedua model dalam percakapan yang sama, Anda akan segera menghadapi masalah, karena format pesan obrolan mereka sedikit tidak kompatibel. LangChain mengabstraksi perbedaan ini untuk memungkinkan pembuatan aplikasi yang benar-benar independen dari penyedia tertentu. Misalnya, dengan LangChain, percakapan chatbot di mana Anda menggunakan model OpenAI dan Anthropic dapat berjalan.

Terakhir, saat Anda membangun aplikasi LLM Anda dengan beberapa komponen ini, kami merasa berguna untuk memiliki kemampuan _orkestrasi_ dari LangChain:

- Semua komponen utama diinstrumentasi oleh sistem panggilan balik (callbacks) untuk kemampuan pengamatan (observability) (lebih lanjut tentang ini di [Bab 8](ch08.xhtml#ch08_patterns_to_make_the_most_of_llms_1736545674143600)).
- Semua komponen utama mengimplementasikan antarmuka yang sama (lebih lanjut tentang ini menjelang akhir bab ini).
- Aplikasi LLM yang berjalan lama dapat diinterupsi, dilanjutkan, atau diulang (lebih lanjut tentang ini di [Bab 6](ch06.xhtml#ch06_agent_architecture_1736545671750341)).

## Memulai dengan LangChain

Untuk mengikuti sisa bab ini, dan bab-bab selanjutnya, kami menyarankan untuk mengatur LangChain di komputer Anda terlebih dahulu.

Lihat petunjuk di [Pendahuluan](preface01.xhtml#pr01_preface_1736545679069216) mengenai pengaturan akun OpenAI dan selesaikan jika Anda belum melakukannya. Jika Anda lebih suka menggunakan penyedia LLM yang berbeda, lihat ["Mengapa LangChain?"](#ch01_why_langchain_1736545659776355) untuk alternatif.

Kemudian buka [halaman Kunci API](https://oreil.ly/BKrtV) di situs web OpenAI (setelah masuk ke akun OpenAI Anda), buat kunci API, dan simpan—Anda akan segera membutuhkannya.

> **Catatan**
> Dalam buku ini, kami akan menampilkan contoh kode dalam Python dan JavaScript (JS). LangChain menawarkan fungsionalitas yang sama dalam kedua bahasa, jadi pilih saja yang paling Anda kuasai dan ikuti cuplikan kode yang sesuai di seluruh buku (contoh kode untuk setiap bahasa setara).

Pertama, beberapa petunjuk pengaturan untuk pembaca yang menggunakan Python:

1.  Pastikan Anda telah menginstal Python. Lihat [petunjuk untuk sistem operasi Anda](https://oreil.ly/20K9l).
2.  Instal Jupyter jika Anda ingin menjalankan contoh dalam lingkungan notebook. Anda dapat melakukannya dengan menjalankan `pip install notebook` di terminal Anda.
3.  Instal pustaka LangChain dengan menjalankan perintah berikut di terminal Anda:

    ```
    pip install langchain langchain-openai langchain-community
    pip install langchain-text-splitters langchain-postgres
    ```

4.  Ambil kunci API OpenAI yang Anda buat di awal bagian ini dan buat tersedia di lingkungan terminal Anda. Anda dapat melakukannya dengan menjalankan perintah berikut:

    ```
    export OPENAI_API_KEY=kunci-anda
    ```

5.  Jangan lupa mengganti `kunci-anda` dengan kunci API yang Anda buat sebelumnya.
6.  Buka notebook Jupyter dengan menjalankan perintah ini:

    ```
    jupyter notebook
    ```

Sekarang Anda siap untuk mengikuti contoh kode Python.

Berikut adalah petunjuk untuk pembaca yang menggunakan JavaScript:

1.  Ambil kunci API OpenAI yang Anda buat di awal bagian ini dan buat tersedia di lingkungan terminal Anda. Anda dapat melakukannya dengan menjalankan perintah berikut:

    ```
    export OPENAI_API_KEY=kunci-anda
    ```

2.  Jangan lupa mengganti `kunci-anda` dengan kunci API yang Anda buat sebelumnya.
3.  Jika Anda ingin menjalankan contoh sebagai skrip Node.js, instal Node dengan mengikuti [petunjuk](https://oreil.ly/5gjiO).
4.  Instal pustaka LangChain dengan menjalankan perintah berikut di terminal Anda:

    ```
    npm install langchain @langchain/openai @langchain/community
    npm install @langchain/core pg
    ```

5.  Ambil setiap contoh, simpan sebagai file _.js_ dan jalankan dengan `node ./file.js`.

## Menggunakan LLM dalam LangChain

Sebagai ringkasan, LLM adalah mesin penggerak di balik sebagian besar aplikasi AI generatif. LangChain menyediakan dua antarmuka sederhana untuk berinteraksi dengan penyedia API LLM apa pun:

- Model obrolan (chat models)
- LLM

Antarmuka LLM hanya mengambil petunjuk string sebagai masukan, mengirimkan masukan ke penyedia model, dan kemudian mengembalikan prediksi model sebagai keluaran.

Mari impor pembungkus LLM OpenAI dari LangChain untuk `invoke` prediksi model menggunakan petunjuk sederhana:

_Python_

```python
from langchain_openai.llms import OpenAI

model = OpenAI(model="gpt-3.5-turbo")

model.invoke("The sky is")
```

_JavaScript_

```javascript
import { OpenAI } from "@langchain/openai"

const model = new OpenAI({ model: "gpt-3.5-turbo" })

await model.invoke("The sky is")
```

_Keluaran:_

```
Blue!
```

> **Tip**
> Perhatikan parameter `model` yang diteruskan ke `OpenAI`. Ini adalah parameter paling umum yang dikonfigurasi saat menggunakan LLM atau model obrolan, model dasar yang akan digunakan, karena sebagian besar penyedia menawarkan beberapa model dengan pertukaran kemampuan dan biaya yang berbeda (biasanya model yang lebih besar lebih mampu, tetapi juga lebih mahal dan lambat). Lihat [ikhtisar OpenAI](https://oreil.ly/dM886) tentang model yang mereka tawarkan.

> Parameter berguna lainnya untuk dikonfigurasi termasuk yang berikut, ditawarkan oleh sebagian besar penyedia.

> `temperature`
> : Ini mengontrol algoritma pengambilan sampel (sampling) yang digunakan untuk menghasilkan keluaran. Nilai yang lebih rendah menghasilkan keluaran yang lebih dapat diprediksi (misalnya, 0.1), sementara nilai yang lebih tinggi menghasilkan hasil yang lebih kreatif, atau tidak terduga (seperti 0.9). Tugas yang berbeda akan membutuhkan nilai yang berbeda untuk parameter ini. Misalnya, menghasilkan keluaran terstruktur biasanya mendapat manfaat dari suhu yang lebih rendah, sedangkan tugas penulisan kreatif lebih baik dengan nilai yang lebih tinggi:

> `max_tokens`
> : Ini membatasi ukuran (dan biaya) keluaran. Nilai yang lebih rendah dapat menyebabkan LLM berhenti menghasilkan keluaran sebelum mencapai akhir alami, sehingga mungkin tampak terpotong.

> Di luar ini, setiap penyedia mengekspos serangkaian parameter yang berbeda. Kami menyarankan untuk melihat dokumentasi untuk yang Anda pilih. Untuk contoh, lihat [platform OpenAI](https://oreil.ly/5O1RW).

Atau, antarmuka model obrolan memungkinkan percakapan bolak-balik antara pengguna dan model. Alasan mengapa ini adalah antarmuka terpisah adalah karena penyedia LLM populer seperti OpenAI membedakan pesan yang dikirim ke dan dari model menjadi peran _pengguna_, _asisten_, dan _sistem_ (di sini _peran_ menunjukkan jenis konten yang dikandung pesan):

**Peran sistem**
: Digunakan untuk instruksi yang harus digunakan model untuk menjawab pertanyaan pengguna

**Peran pengguna**
: Digunakan untuk kueri pengguna dan konten lain yang dihasilkan oleh pengguna

**Peran asisten**
: Digunakan untuk konten yang dihasilkan oleh model

Antarmuka model obrolan memudahkan untuk mengonfigurasi dan mengelola konversi dalam aplikasi chatbot AI Anda. Berikut contoh menggunakan model ChatOpenAI dari LangChain:

_Python_

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI()
prompt = [HumanMessage("What is the capital of France?")]

model.invoke(prompt)
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { HumanMessage } from "@langchain/core/messages"

const model = new ChatOpenAI()
const prompt = [new HumanMessage("What is the capital of France?")]

await model.invoke(prompt)
```

_Keluaran:_

```
AIMessage(content='The capital of France is Paris.')
```

Alih-alih satu string petunjuk, model obrolan menggunakan berbagai jenis antarmuka pesan obrolan yang terkait dengan setiap peran yang disebutkan sebelumnya. Ini termasuk yang berikut:

`HumanMessage`
: Pesan yang dikirim dari perspektif manusia, dengan peran pengguna

`AIMessage`
: Pesan yang dikirim dari perspektif AI yang sedang berinteraksi dengan manusia, dengan peran asisten

`SystemMessage`
: Pesan yang menetapkan instruksi yang harus diikuti AI, dengan peran sistem

`ChatMessage`
: Pesan yang memungkinkan pengaturan peran yang arbitrer

Mari sertakan instruksi `SystemMessage` dalam contoh kita:

_Python_

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()
system_msg = SystemMessage(
    '''You are a helpful assistant that responds to questions with three
        exclamation marks.'''
)
human_msg = HumanMessage('What is the capital of France?')

model.invoke([system_msg, human_msg])
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { HumanMessage, SystemMessage } from "@langchain/core/messages"

const model = new ChatOpenAI()
const prompt = [
  new SystemMessage(
    `You are a helpful assistant that responds to questions with three 
      exclamation marks.`
  ),
  new HumanMessage("What is the capital of France?"),
]

await model.invoke(prompt)
```

_Keluaran:_

```
AIMessage('Paris!!!')
```

Seperti yang Anda lihat, model mematuhi instruksi yang diberikan dalam `SystemMessage` meskipun tidak ada dalam pertanyaan pengguna. Ini memungkinkan Anda untuk mengonfigurasi aplikasi AI Anda sebelumnya agar merespons dengan cara yang relatif dapat diprediksi berdasarkan masukan pengguna.

## Membuat Petunjuk LLM Dapat Digunakan Kembali

Bagian sebelumnya menunjukkan bagaimana instruksi `prompt` secara signifikan memengaruhi keluaran model. Petunjuk membantu model memahami konteks dan menghasilkan jawaban yang relevan untuk kueri.

Berikut adalah contoh petunjuk yang detail:

```
Jawab pertanyaan berdasarkan konteks di bawah ini. Jika pertanyaan tidak dapat dijawab menggunakan informasi yang diberikan, jawab dengan "Saya tidak tahu".

Konteks: Kemajuan terbaru dalam NLP didorong oleh Model Bahasa Besar (Large Language Models/LLMs). Model-model ini mengungguli model yang lebih kecil dan telah menjadi sangat berharga bagi pengembang yang membuat aplikasi dengan kemampuan NLP. Pengembang dapat memanfaatkan model-model ini melalui pustaka `transformers` dari Hugging Face, atau dengan menggunakan penawaran dari OpenAI dan Cohere melalui pustaka `openai` dan `cohere` secara berturut-turut.

Pertanyaan: Penyedia model mana yang menawarkan LLM?

Jawaban:
```

```
Answer the question based on the context below. If the question cannot be
answered using the information provided, answer with "I don't know".

Context: The most recent advancements in NLP are being driven by Large Language
Models (LLMs). These models outperform their smaller counterparts and have
become invaluable for developers who are creating applications with NLP
capabilities. Developers can tap into these models through Hugging Face's
`transformers` library, or by utilizing OpenAI and Cohere's offerings through
the `openai` and `cohere` libraries, respectively.

Question: Which model providers offer LLMs?

Answer:
```

Meskipun petunjuk terlihat seperti string sederhana, tantangannya adalah mencari tahu apa yang harus dikandung teks tersebut dan bagaimana teks tersebut harus bervariasi berdasarkan masukan pengguna. Dalam contoh ini, nilai Konteks dan Pertanyaan dikodekan secara keras (hardcoded), tetapi bagaimana jika kita ingin meneruskan ini secara dinamis?

Untungnya, LangChain menyediakan antarmuka templat petunjuk yang memudahkan pembuatan petunjuk dengan masukan dinamis:

_Python_

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""Answer the question based on the
    context below. If the question cannot be answered using the information
    provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)
```

_JavaScript_

```javascript
import { PromptTemplate } from "@langchain/core/prompts"

const template = PromptTemplate.fromTemplate(`Answer the question based on the 
  context below. If the question cannot be answered using the information 
  provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `)
```

_Keluaran:_

```
StringPromptValue(text='Answer the question based on the context below. If the
    question cannot be answered using the information provided, answer with "I
    don\'t know".\n\nContext: The most recent advancements in NLP are being
    driven by Large Language Models (LLMs). These models outperform their
    smaller counterparts and have become invaluable for developers who are
    creating applications with NLP capabilities. Developers can tap into these
    models through Hugging Face\'s `transformers` library, or by utilizing
    OpenAI and Cohere\'s offerings through the `openai` and `cohere` libraries,
    respectively.\n\nQuestion: Which model providers offer LLMs?\n\nAnswer: ')
```

Contoh ini mengambil petunjuk statis dari blok sebelumnya dan membuatnya dinamis. `template` berisi struktur petunjuk akhir bersama dengan definisi di mana masukan dinamis akan disisipkan.

Dengan demikian, templat dapat digunakan sebagai resep untuk membangun beberapa petunjuk statis yang spesifik. Saat Anda memformat petunjuk dengan beberapa nilai spesifik—dalam hal ini, `context` dan `question`—Anda mendapatkan petunjuk statis yang siap untuk diteruskan ke LLM.

Seperti yang Anda lihat, argumen `question` diteruskan secara dinamis melalui fungsi `invoke`. Secara default, petunjuk LangChain mengikuti sintaks `f-string` Python untuk mendefinisikan parameter dinamis—kata apa pun yang diapit kurung kurawal, seperti `{question}`, adalah tempat penampung untuk nilai yang diteruskan saat runtime. Dalam contoh sebelumnya, `{question}` digantikan oleh `"Which model providers offer LLMs?"`

Mari lihat bagaimana kita akan memasukkan ini ke dalam model LLM OpenAI menggunakan LangChain:

_Python_

```python
from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate

# both `template` and `model` can be reused many times

template = PromptTemplate.from_template("""Answer the question based on the
    context below. If the question cannot be answered using the information
    provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)
```

_JavaScript_

```javascript
import { PromptTemplate } from "@langchain/core/prompts"
import { OpenAI } from "@langchain/openai"

const model = new OpenAI()
const template = PromptTemplate.fromTemplate(`Answer the question based on the  
  context below. If the question cannot be answered using the information 
  provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: `)
```

_Keluaran:_

```
Hugging Face's `transformers` library, OpenAI using the `openai` library, and
Cohere using the `cohere` library offer LLMs.
```

Jika Anda ingin membangun aplikasi obrolan AI, `ChatPromptTemplate` dapat digunakan sebagai gantinya untuk memberikan masukan dinamis berdasarkan peran pesan obrolan:

_Python_

```python
from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the
        question cannot be answered using the information provided, answer with
        "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])
```

_JavaScript_

```javascript
import { ChatPromptTemplate } from "@langchain/core/prompts"

const template = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Answer the question based on the context below. If the question 
    cannot be answered using the information provided, answer with "I 
    don\'t know".`,
  ],
  ["human", "Context: {context}"],
  ["human", "Question: {question}"],
])
```

_Keluaran:_

```
ChatPromptValue(messages=[SystemMessage(content='Answer the question based on
    the context below. If the question cannot be answered using the information
    provided, answer with "I don\'t know".'), HumanMessage(content="Context:
    The most recent advancements in NLP are being driven by Large Language
    Models (LLMs). These models outperform their smaller counterparts and have
    become invaluable for developers who are creating applications with NLP
    capabilities. Developers can tap into these models through Hugging Face\'s
    `transformers` library, or by utilizing OpenAI and Cohere\'s offerings
    through the `openai` and `cohere` libraries, respectively."), HumanMessage
    (content='Question: Which model providers offer LLMs?')])
```

Perhatikan bagaimana petunjuk berisi instruksi dalam `SystemMessage` dan dua contoh `HumanMessage` yang berisi variabel dinamis `context` dan `question`. Anda masih dapat memformat templat dengan cara yang sama dan mendapatkan kembali petunjuk statis yang dapat Anda berikan ke model bahasa besar untuk keluaran prediksi:

_Python_

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# both `template` and `model` can be reused many times

template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the
        question cannot be answered using the information provided, answer
        with "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOpenAI()
```

_JavaScript_

```javascript
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { ChatOpenAI } from "@langchain/openai"

const model = new ChatOpenAI()
const template = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Answer the question based on the context below. If the question 
    cannot be answered using the information provided, answer with "I 
    don\'t know".`,
  ],
  ["human", "Context: {context}"],
  ["human", "Question: {question}"],
])
```

_Keluaran:_

```
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

## Mendapatkan Format Spesifik dari LLM

Keluaran teks biasa berguna, tetapi mungkin ada kasus penggunaan di mana Anda memerlukan LLM untuk menghasilkan keluaran _terstruktur_—yaitu, keluaran dalam format yang dapat dibaca mesin, seperti JSON, XML, CSV, atau bahkan dalam bahasa pemrograman seperti Python atau JavaScript. Ini sangat berguna ketika Anda bermaksud untuk menyerahkan keluaran tersebut ke bagian kode lain, membuat LLM memainkan peran dalam aplikasi yang lebih besar.

### Keluaran JSON

Format paling umum yang dihasilkan dengan LLM adalah JSON. Keluaran JSON dapat (misalnya) dikirim melalui jaringan ke kode frontend Anda atau disimpan ke basis data.

Saat menghasilkan JSON, tugas pertama adalah mendefinisikan skema yang Anda ingin LLM hormati saat menghasilkan keluaran. Kemudian, Anda harus menyertakan skema itu dalam petunjuk, bersama dengan teks yang ingin Anda gunakan sebagai sumber. Mari kita lihat contoh:

_Python_

```python
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the
        answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

structured_llm.invoke("""What weighs more, a pound of bricks or a pound
    of feathers""")
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { z } from "zod"

const answerSchema = z.object({
  answer: z.string().describe("The answer to the user's question"),
  justification: z.string().describe(`Justification for the 
      answer`),
}).describe(`An answer to the user's question along with justification for 
    the answer.`)

const model = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
}).withStructuredOutput(answerSchema)
await model.invoke("What weighs more, a pound of bricks or a pound of feathers")
```

_Keluaran:_

```
{
  answer: "They weigh the same",
  justification: "Both a pound of bricks and a pound of feathers weigh one pound.
    The weight is the same, but the volu"... 42 more characters
}
```

Jadi, pertama-tama definisikan skema. Di Python, ini paling mudah dilakukan dengan Pydantic (pustaka yang digunakan untuk memvalidasi data terhadap skema). Di JS, ini paling mudah dilakukan dengan Zod (pustaka yang setara). Metode `with_structured_output` akan menggunakan skema itu untuk dua hal:

- Skema akan dikonversi ke objek `JSONSchema` (format JSON yang digunakan untuk menggambarkan bentuk [tipe, nama, deskripsi] data JSON), yang akan dikirim ke LLM. Untuk setiap LLM, LangChain memilih metode terbaik untuk melakukan ini, biasanya pemanggilan fungsi atau pemberian petunjuk.
- Skema juga akan digunakan untuk memvalidasi keluaran yang dikembalikan oleh LLM sebelum mengembalikannya; ini memastikan keluaran yang dihasilkan menghormati skema yang Anda berikan dengan tepat.

### Format yang Dapat Dibaca Mesin Lainnya dengan Pengurai Keluaran

Anda juga dapat menggunakan LLM atau model obrolan untuk menghasilkan keluaran dalam format lain, seperti CSV atau XML. Di sinilah pengurai keluaran (output parsers) berguna. _Pengurai keluaran_ adalah kelas yang membantu Anda menyusun respons model bahasa besar. Mereka memiliki dua fungsi:

**Menyediakan instruksi format**
: Pengurai keluaran dapat digunakan untuk menyuntikkan beberapa instruksi tambahan dalam petunjuk yang akan membantu membimbing LLM untuk mengeluarkan teks dalam format yang diketahui cara menguraikannya.

**Memvalidasi dan mengurai keluaran**
: Fungsi utamanya adalah mengambil keluaran tekstual dari LLM atau model obrolan dan merendernya ke format yang lebih terstruktur, seperti daftar, XML, atau format lainnya. Ini dapat mencakup menghapus informasi yang tidak relevan, memperbaiki keluaran yang tidak lengkap, dan memvalidasi nilai yang diurai.

Berikut contoh cara kerja pengurai keluaran:

_Python_

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
items = parser.invoke("apple, banana, cherry")
```

_JavaScript_

```javascript
import { CommaSeparatedListOutputParser } from "@langchain/core/output_parsers"

const parser = new CommaSeparatedListOutputParser()

await parser.invoke("apple, banana, cherry")
```

_Keluaran:_

```
['apple', 'banana', 'cherry']
```

LangChain menyediakan berbagai pengurai keluaran untuk berbagai kasus penggunaan, termasuk CSV, XML, dan lainnya. Kita akan melihat cara menggabungkan pengurai keluaran dengan model dan petunjuk di bagian selanjutnya.

## Merakit Banyak Bagian dari Aplikasi LLM

Komponen kunci yang telah Anda pelajari sejauh ini adalah blok bangunan penting dari kerangka kerja LangChain. Yang membawa kita ke pertanyaan kritis: Bagaimana Anda menggabungkannya secara efektif untuk membangun aplikasi LLM Anda?

### Menggunakan Antarmuka Runnable

Seperti yang mungkin Anda perhatikan, semua contoh kode yang digunakan sejauh ini menggunakan antarmuka serupa dan metode `invoke()` untuk menghasilkan keluaran dari model (atau templat petunjuk, atau pengurai keluaran). Semua komponen memiliki hal berikut:

- Ada antarmuka umum dengan metode-metode ini:
  - `invoke`: mengubah satu masukan menjadi satu keluaran
  - `batch`: mengubah banyak masukan menjadi banyak keluaran secara efisien
  - `stream`: mengalirkan keluaran dari satu masukan saat diproduksi
- Ada utilitas bawaan untuk percobaan ulang, fallback, skema, dan konfigurabilitas runtime.
- Di Python, masing-masing dari ketiga metode memiliki setara `asyncio`.

Dengan demikian, semua komponen berperilaku sama, dan antarmuka yang dipelajari untuk salah satunya berlaku untuk semua:

_Python_

```python
from langchain_openai.llms import ChatOpenAI

model = ChatOpenAI()

completion = model.invoke('Hi there!')
# Hi!

completions = model.batch(['Hi there!', 'Bye!'])
# ['Hi!', 'See you!']

for token in model.stream('Bye!'):
    print(token)
    # Good
    # bye
    # !
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"

const model = new ChatOpenAI()

const completion = await model.invoke("Hi there!")
// Hi!

const completions = await model.batch(["Hi there!", "Bye!"])
// ['Hi!', 'See you!']

for await (const token of await model.stream("Bye!")) {
  console.log(token)
  // Good
  // bye
  // !
}
```

Dalam contoh ini, Anda melihat bagaimana ketiga metode utama bekerja:

- `invoke()` mengambil satu masukan dan mengembalikan satu keluaran.
- `batch()` mengambil daftar masukan dan mengembalikan daftar keluaran.
- `stream()` mengambil satu masukan dan mengembalikan iterator dari bagian keluaran saat tersedia.

Dalam beberapa kasus, di mana komponen dasar tidak mendukung keluaran iteratif, akan ada satu bagian yang berisi semua keluaran.

Anda dapat menggabungkan komponen-komponen ini dengan dua cara:

**Imperatif**
: Panggil komponen Anda secara langsung, misalnya, dengan `model.invoke(...)`

**Deklaratif**
: Gunakan Bahasa Ekspresi LangChain (LCEL), seperti yang dibahas di bagian mendatang

[Tabel 1-1](#ch01_table_1_1736545659767905) merangkum perbedaan mereka, dan kita akan melihat masing-masing dalam aksi selanjutnya.

**Tabel 1-1. Perbedaan utama antara komposisi imperatif dan deklaratif.**

|                  | Imperatif                                                                  | Deklaratif |
| ---------------- | -------------------------------------------------------------------------- | ---------- |
| Sintaks          | Semua Python atau JavaScript                                               | LCEL       |
| Eksekusi paralel | Python: dengan thread atau korutin<br><br>JavaScript: dengan `Promise.all` | Otomatis   |
| Streaming        | Dengan kata kunci yield                                                    | Otomatis   |
| Eksekusi async   | Dengan fungsi async                                                        | Otomatis   |

### Komposisi Imperatif

_Komposisi imperatif_ hanyalah nama mewah untuk menulis kode yang biasa Anda tulis, menyusun komponen-komponen ini menjadi fungsi dan kelas. Berikut contoh menggabungkan petunjuk, model, dan pengurai keluaran:

_Python_

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

model = ChatOpenAI()

# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

# use it

chatbot.invoke({"question": "Which model providers offer LLMs?"})
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { RunnableLambda } from "@langchain/core/runnables"

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  ["human", "{question}"],
])

const model = new ChatOpenAI()

// combine them in a function
// RunnableLambda adds the same Runnable interface for any function you write

const chatbot = RunnableLambda.from(async (values) => {
  const prompt = await template.invoke(values)
  return await model.invoke(prompt)
})

// use it

await chatbot.invoke({
  question: "Which model providers offer LLMs?",
})
```

_Keluaran:_

```
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

Ini adalah contoh lengkap dari chatbot, menggunakan petunjuk dan model obrolan. Seperti yang Anda lihat, ini menggunakan sintaks Python yang familiar dan mendukung logika kustom apa pun yang mungkin ingin Anda tambahkan dalam fungsi itu.

Di sisi lain, jika Anda ingin mengaktifkan dukungan streaming atau async, Anda harus memodifikasi fungsi Anda untuk mendukungnya. Misalnya, dukungan streaming dapat ditambahkan sebagai berikut:

_Python_

```python
@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

for part in chatbot.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part)
```

_JavaScript_

```javascript
const chatbot = RunnableLambda.from(async function* (values) {
  const prompt = await template.invoke(values)
  for await (const token of await model.stream(prompt)) {
    yield token
  }
})

for await (const token of await chatbot.stream({
  question: "Which model providers offer LLMs?",
})) {
  console.log(token)
}
```

_Keluaran:_

```
AIMessageChunk(content="Hugging")
AIMessageChunk(content=" Face's")
AIMessageChunk(content=" `transformers`")
...
```

Jadi, baik di JS atau Python, Anda dapat mengaktifkan streaming untuk fungsi kustom Anda dengan menghasilkan nilai yang ingin Anda streaming dan kemudian memanggilnya dengan `stream`.

Untuk eksekusi asinkron, Anda akan menulis ulang fungsi Anda seperti ini:

_Python_

```python
@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})
# > AIMessage(content="""Hugging Face's `transformers` library, OpenAI using
    the `openai` library, and Cohere using the `cohere` library offer LLMs.""")
```

Ini hanya berlaku untuk Python, karena eksekusi asinkron adalah satu-satunya opsi di JavaScript.

### Komposisi Deklaratif

LCEL adalah _bahasa deklaratif_ untuk menyusun komponen LangChain. LangChain mengompilasi komposisi LCEL ke _rencana eksekusi yang dioptimalkan_, dengan dukungan paralelisasi, streaming, pelacakan, dan async otomatis.

Mari lihat contoh yang sama menggunakan LCEL:

_Python_

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('human', '{question}'),
])

model = ChatOpenAI()

# combine them with the | operator

chatbot = template | model

# use it

chatbot.invoke({"question": "Which model providers offer LLMs?"})
```

_JavaScript_

```javascript
import { ChatOpenAI } from "@langchain/openai"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { RunnableLambda } from "@langchain/core/runnables"

// the building blocks

const template = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant."],
  ["human", "{question}"],
])

const model = new ChatOpenAI()

// combine them in a function

const chatbot = template.pipe(model)

// use it

await chatbot.invoke({
  question: "Which model providers offer LLMs?",
})
```

_Keluaran:_

```
AIMessage(content="Hugging Face's `transformers` library, OpenAI using the
    `openai` library, and Cohere using the `cohere` library offer LLMs.")
```

Yang penting, baris terakhir sama antara kedua contoh—artinya, Anda menggunakan fungsi dan urutan LCEL dengan cara yang sama, dengan `invoke/stream/batch`. Dan dalam versi ini, Anda tidak perlu melakukan hal lain untuk menggunakan streaming:

_Python_

```python
chatbot = template | model

for part in chatbot.stream({
    "question": "Which model providers offer LLMs?"
}):
    print(part)
    # > AIMessageChunk(content="Hugging")
    # > AIMessageChunk(content=" Face's")
    # > AIMessageChunk(content=" `transformers`")
    # ...
```

_JavaScript_

```javascript
const chatbot = template.pipe(model)

for await (const token of await chatbot.stream({
  question: "Which model providers offer LLMs?",
})) {
  console.log(token)
}
```

Dan, hanya untuk Python, ini sama untuk menggunakan metode asinkron:

_Python_

```python
chatbot = template | model

await chatbot.ainvoke({
    "question": "Which model providers offer LLMs?"
})
```

## Ringkasan

Dalam bab ini, Anda telah mempelajari tentang blok bangunan dan komponen kunci yang diperlukan untuk membangun aplikasi LLM menggunakan LangChain. Aplikasi LLM pada dasarnya adalah rantai yang terdiri dari model bahasa besar untuk membuat prediksi, instruksi petunjuk untuk membimbing model menuju keluaran yang diinginkan, dan pengurai keluaran opsional untuk mengubah format keluaran model.

Semua komponen LangChain berbagi antarmuka yang sama dengan metode `invoke`, `stream`, dan `batch` untuk menangani berbagai masukan dan keluaran. Mereka dapat digabungkan dan dieksekusi secara imperatif dengan memanggilnya langsung atau secara deklaratif menggunakan LCEL.

Pendekatan imperatif berguna jika Anda bermaksud menulis banyak logika kustom, sedangkan pendekatan deklaratif berguna untuk sekadar merakit komponen yang ada dengan kustomisasi terbatas.

Di [Bab 2](ch02.xhtml#ch02_rag_part_i_indexing_your_data_1736545662500927), Anda akan mempelajari cara menyediakan data eksternal ke chatbot AI Anda sebagai _konteks_ sehingga Anda dapat membangun aplikasi LLM yang memungkinkan Anda "mengobrol" dengan data Anda.
