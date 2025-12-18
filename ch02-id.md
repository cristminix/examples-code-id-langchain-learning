> # Bab 2. RAG Bagian I: Mengindeks Data Anda

Di bab sebelumnya, Anda mempelajari tentang blok bangunan penting yang digunakan untuk membuat aplikasi LLM menggunakan LangChain. Anda juga membangun chatbot AI sederhana yang terdiri dari petunjuk yang dikirim ke model dan keluaran yang dihasilkan oleh model. Tetapi ada batasan besar pada chatbot sederhana ini.

Bagaimana jika kasus penggunaan Anda memerlukan pengetahuan yang tidak dilatih pada model? Misalnya, katakanlah Anda ingin menggunakan AI untuk mengajukan pertanyaan tentang suatu perusahaan, tetapi informasinya terkandung dalam PDF pribadi atau jenis dokumen lainnya. Meskipun kami telah melihat penyedia model memperkaya set data pelatihan mereka untuk mencakup semakin banyak informasi publik dunia (tidak peduli dalam format apa pun disimpan), dua batasan utama terus ada dalam korpus pengetahuan LLM:

**Data pribadi**
: Informasi yang tidak tersedia untuk umum, menurut definisi, tidak termasuk dalam data pelatihan LLM.

**Peristiwa terkini**
: Melatih LLM adalah proses yang mahal dan memakan waktu yang dapat berlangsung selama bertahun-tahun, dengan pengumpulan data menjadi salah satu langkah pertama. Ini menghasilkan apa yang disebut batas pengetahuan (knowledge cutoff), atau tanggal di luar mana LLM tidak memiliki pengetahuan tentang peristiwa dunia nyata; biasanya ini adalah tanggal set pelatihan diselesaikan. Ini bisa berkisar dari beberapa bulan hingga beberapa tahun ke belakang, tergantung pada model yang dimaksud.

Dalam kedua kasus, model kemungkinan besar akan berhalusinasi (menemukan informasi yang menyesatkan atau salah) dan merespons dengan informasi yang tidak akurat. Menyesuaikan petunjuk juga tidak akan menyelesaikan masalah karena bergantung pada pengetahuan model saat ini.

## Tujuan: Memilih Konteks yang Relevan untuk LLM

Jika satu-satunya data pribadi/terkini yang Anda perlukan untuk kasus penggunaan LLM Anda adalah satu hingga dua halaman teks, bab ini akan jauh lebih pendek: yang Anda perlukan untuk membuat informasi itu tersedia bagi LLM adalah dengan menyertakan seluruh teks itu dalam setiap petunjuk yang Anda kirim ke model.

Tantangan dalam membuat data tersedia untuk LLM pertama-tama adalah masalah kuantitas. Anda memiliki lebih banyak informasi daripada yang dapat muat dalam setiap petunjuk yang Anda kirim ke LLM. Subset kecil mana dari kumpulan teks besar Anda yang Anda sertakan setiap kali Anda memanggil model? Atau dengan kata lain, bagaimana Anda memilih (dengan bantuan model) teks mana yang paling relevan untuk menjawab setiap pertanyaan?

Dalam bab ini dan selanjutnya, Anda akan belajar bagaimana mengatasi tantangan ini dalam dua langkah:

1.  _Mengindeks_ dokumen Anda, yaitu, memprosesnya terlebih dahulu dengan cara di mana aplikasi Anda dapat dengan mudah menemukan yang paling relevan untuk setiap pertanyaan
2.  _Mengambil_ data eksternal ini dari indeks dan menggunakannya sebagai _konteks_ untuk LLM untuk menghasilkan keluaran yang akurat berdasarkan data Anda

Bab ini berfokus pada pengindeksan, langkah pertama, yang melibatkan pemrosesan awal dokumen Anda ke dalam format yang dapat dipahami dan dicari dengan LLM. Teknik ini disebut _generasi yang diperkuat pengambilan_ (retrieval-augmented generation/RAG). Tetapi sebelum kita mulai, mari bahas _mengapa_ dokumen Anda memerlukan pemrosesan awal.

Mari kita asumsikan Anda ingin menggunakan LLM untuk menganalisis kinerja dan risiko keuangan dalam [laporan tahunan Tesla 2022](https://oreil.ly/Bp51E), yang disimpan sebagai teks dalam format PDF. Tujuan Anda adalah dapat mengajukan pertanyaan seperti "Risiko utama apa yang dihadapi Tesla pada tahun 2022?" dan mendapatkan respons seperti manusia berdasarkan konteks dari bagian faktor risiko dokumen.

Memecahnya, ada empat langkah kunci (ditunjukkan pada [Gambar 2-1](#ch02_figure_1_1736545662484110)) yang perlu Anda ambil untuk mencapai tujuan ini:

1.  Ekstrak teks dari dokumen.
2.  Bagi teks menjadi bagian-bagian yang dapat dikelola (chunks).
3.  Ubah teks menjadi angka yang dapat dipahami komputer.
4.  Simpan representasi angka dari teks Anda di suatu tempat yang membuatnya mudah dan cepat untuk mengambil bagian dokumen Anda yang relevan untuk menjawab pertanyaan yang diberikan.

![Empat langkah kunci untuk memproses dokumen Anda untuk penggunaan LLM](Images/lelc_0201.png)

[Gambar 2-1](#ch02_figure_1_1736545662484110) mengilustrasikan alur pemrosesan awal dan transformasi dokumen Anda, sebuah proses yang dikenal sebagai ingesti (ingestion). _Ingesti_ hanyalah proses mengubah dokumen Anda menjadi angka yang dapat dipahami dan dianalisis komputer, dan menyimpannya dalam jenis basis data khusus untuk pengambilan yang efisien. Angka-angka ini secara formal dikenal sebagai _penanaman (embeddings)_, dan jenis basis data khusus ini dikenal sebagai _penyimpanan vektor (vector store)_. Mari kita lihat lebih dekat apa itu penanaman dan mengapa itu penting, dimulai dengan sesuatu yang lebih sederhana daripada penanaman bertenaga LLM.

## Penanaman (Embeddings): Mengubah Teks Menjadi Angka

_Penanaman (embedding)_ mengacu pada merepresentasikan teks sebagai urutan angka (yang panjang). Ini adalah representasi yang kehilangan informasi (lossy)—artinya, Anda tidak dapat memulihkan teks asli dari urutan angka ini, jadi biasanya Anda menyimpan teks asli dan representasi numerik ini.

Jadi, mengapa repot? Karena Anda mendapatkan fleksibilitas dan kekuatan yang datang dengan bekerja dengan angka: Anda dapat melakukan matematika pada kata-kata! Mari kita lihat mengapa itu menarik.

### Penanaman Sebelum LLM

Jauh sebelum LLM, ilmuwan komputer telah menggunakan penanaman—misalnya, untuk mengaktifkan kemampuan pencarian teks lengkap di situs web atau untuk mengklasifikasikan email sebagai spam. Mari kita lihat contoh:

1.  Ambil tiga kalimat ini:
    - What a sunny day. (Hari yang cerah.)
    - Such bright skies today. (Langit begitu cerah hari ini.)
    - I haven't seen a sunny day in weeks. (Saya belum melihat hari yang cerah selama berminggu-minggu.)
2.  Sebutkan semua kata unik di dalamnya: _what_, _a_, _sunny_, _day_, _such_, _bright_, dan seterusnya.
3.  Untuk setiap kalimat, pergi kata demi kata dan tetapkan angka 0 jika tidak ada, 1 jika digunakan sekali dalam kalimat, 2 jika ada dua kali, dan seterusnya.

[Tabel 2-1](#ch02_table_1_1736545662489691) menunjukkan hasilnya.

**Tabel 2-1. Penanaman kata untuk tiga kalimat**

| Kata      | What a sunny day. | Such bright skies today. | I haven't seen a sunny day in weeks. |
| --------- | ----------------- | ------------------------ | ------------------------------------ |
| _what_    | 1                 | 0                        | 0                                    |
| _a_       | 1                 | 0                        | 1                                    |
| _sunny_   | 1                 | 0                        | 1                                    |
| _day_     | 1                 | 0                        | 1                                    |
| _such_    | 0                 | 1                        | 0                                    |
| _bright_  | 0                 | 1                        | 0                                    |
| _skies_   | 0                 | 1                        | 0                                    |
| _today_   | 0                 | 1                        | 0                                    |
| _I_       | 0                 | 0                        | 1                                    |
| _haven't_ | 0                 | 0                        | 1                                    |
| _seen_    | 0                 | 0                        | 1                                    |
| _in_      | 0                 | 0                        | 1                                    |
| _weeks_   | 0                 | 0                        | 1                                    |

Dalam model ini, penanaman untuk _I haven't seen a sunny day in weeks_ adalah urutan angka _0 1 1 1 0 0 0 0 1 1 1 1 1_. Ini disebut model _kantong kata (bag-of-words)_, dan penanaman ini juga disebut _penanaman jarang (sparse embeddings)_ (atau vektor jarang—_vektor_ adalah kata lain untuk urutan angka), karena banyak angka akan menjadi 0. Sebagian besar kalimat bahasa Inggris hanya menggunakan subset yang sangat kecil dari semua kata bahasa Inggris yang ada.

Anda dapat berhasil menggunakan model ini untuk:

**Pencarian kata kunci**
: Anda dapat menemukan dokumen mana yang berisi kata atau kata-kata tertentu.

**Klasifikasi dokumen**
: Anda dapat menghitung penanaman untuk kumpulan contoh yang sebelumnya diberi label sebagai email spam atau bukan spam, merata-ratakannya, dan memperoleh frekuensi kata rata-rata untuk setiap kelas (spam atau bukan spam). Kemudian, setiap dokumen baru dibandingkan dengan rata-rata tersebut dan diklasifikasikan sesuai.

Batasan di sini adalah bahwa model tidak memiliki kesadaran akan makna, hanya kata-kata yang sebenarnya digunakan. Misalnya, penanaman untuk _sunny day_ dan _bright skies_ terlihat sangat berbeda. Bahkan mereka tidak memiliki kata yang sama, meskipun kita tahu mereka memiliki makna yang serupa. Atau, dalam masalah klasifikasi email, calon pengirim spam dapat menipu filter dengan mengganti "kata spam" umum dengan sinonimnya.

Di bagian selanjutnya, kita akan melihat bagaimana penanaman semantik (semantic embeddings) mengatasi batasan ini dengan menggunakan angka untuk mewakili makna teks, bukan kata-kata yang tepat yang ditemukan dalam teks.

### Penanaman Berbasis LLM

Kami akan melewatkan semua perkembangan ML yang ada di antaranya dan langsung melompat ke penanaman berbasis LLM. Ketahuilah saja bahwa ada evolusi bertahap dari metode sederhana yang diuraikan di bagian sebelumnya ke metode canggih yang dijelaskan dalam bagian ini.

Anda dapat menganggap model penanaman sebagai cabang dari proses pelatihan LLM. Jika Anda ingat dari [Pendahuluan](preface01.xhtml#pr01_preface_1736545679069216), proses pelatihan LLM (belajar dari sejumlah besar teks tertulis) memungkinkan LLM untuk menyelesaikan petunjuk (atau masukan) dengan kelanjutan (keluaran) yang paling tepat. Kemampuan ini berasal dari pemahaman tentang makna kata dan kalimat dalam konteks teks sekitarnya, dipelajari dari bagaimana kata-kata digunakan bersama dalam teks pelatihan. _Pemahaman_ tentang makna (atau semantik) dari petunjuk dapat diekstraksi sebagai representasi numerik (atau penanaman) dari teks masukan, dan dapat digunakan langsung untuk beberapa kasus penggunaan yang sangat menarik juga.

Dalam praktiknya, sebagian besar model penanaman dilatih untuk tujuan itu saja, mengikuti arsitektur dan proses pelatihan yang agak mirip dengan LLM, karena itu lebih efisien dan menghasilkan penanaman berkualitas lebih tinggi.^[1](#id478)]

_Model penanaman_ kemudian adalah algoritma yang mengambil sepotong teks dan mengeluarkan representasi numerik dari maknanya—secara teknis, daftar panjang angka titik-mengambang (desimal), biasanya di antara 100 dan 2.000 angka, atau _dimensi_. Ini juga disebut penanaman _padat (dense)_, sebagai lawan dari penanaman _jarang (sparse)_ dari bagian sebelumnya, karena di sini biasanya semua dimensi berbeda dari 0.

> **Tip**
> Model yang berbeda menghasilkan angka yang berbeda dan ukuran daftar yang berbeda. Semua ini spesifik untuk setiap model; artinya, bahkan jika ukuran daftarnya cocok, Anda tidak dapat membandingkan penanaman dari model yang berbeda. Menggabungkan penanaman dari model yang berbeda harus selalu dihindari.

### Penanaman Semantik Dijelaskan

Pertimbangkan tiga kata ini: _lion_ (singa), _pet_ (hewan peliharaan), dan _dog_ (anjing). Secara intuitif, pasangan mana dari kata-kata ini yang berbagi karakteristik serupa satu sama lain pada pandangan pertama? Jawaban yang jelas adalah _pet_ dan _dog_. Tetapi komputer tidak memiliki kemampuan untuk mengetuk intuisi ini atau pemahaman bernuansa tentang bahasa Inggris. Agar komputer dapat membedakan antara singa, hewan peliharaan, atau anjing, Anda harus dapat menerjemahkannya ke dalam bahasa komputer, yaitu angka*.*

[Gambar 2-2](#ch02_figure_2_1736545662484146) mengilustrasikan mengonversi setiap kata menjadi representasi angka hipotetis yang mempertahankan maknanya.

![Representasi semantik kata-kata](Images/lelc_0202.png)

[Gambar 2-2](#ch02_figure_2_1736545662484146) menunjukkan setiap kata bersama dengan penanaman semantik yang sesuai. Perhatikan bahwa angka-angka itu sendiri tidak memiliki makna tertentu, tetapi sebaliknya urutan angka untuk dua kata (atau kalimat) yang dekat dalam makna seharusnya _lebih dekat_ daripada kata-kata yang tidak terkait. Seperti yang Anda lihat, setiap angka adalah _nilai titik-mengambang (floating-point)_, dan masing-masing mewakili _dimensi_ semantik. Mari kita lihat apa yang kami maksud dengan _lebih dekat_:

Jika kita memplot vektor-vektor ini dalam ruang tiga dimensi, itu bisa terlihat seperti [Gambar 2-3](#ch02_figure_3_1736545662484170).

![Plot vektor kata dalam ruang multidimensi](Images/lelc_0203.png)

[Gambar 2-3](#ch02_figure_3_1736545662484170) menunjukkan vektor _pet_ dan _dog_ lebih dekat satu sama lain dalam jarak daripada plot _lion_. Kita juga dapat mengamati bahwa sudut antara setiap plot bervariasi tergantung pada seberapa mirip mereka. Misalnya, kata _pet_ dan _lion_ memiliki sudut yang lebih lebar di antara mereka daripada _pet_ dan _dog_, menunjukkan lebih banyak kesamaan yang dibagikan oleh pasangan kata yang terakhir. Semakin sempit sudut atau semakin pendek jarak antara dua vektor, semakin dekat kesamaannya.

Salah satu cara efektif untuk menghitung tingkat kesamaan antara dua vektor dalam ruang multidimensi disebut kesamaan kosinus (cosine similarity). _Kesamaan kosinus_ menghitung produk titik (dot product) vektor dan membaginya dengan produk magnitudo mereka untuk mengeluarkan angka antara –1 dan 1, di mana 0 berarti vektor tidak memiliki korelasi, –1 berarti mereka benar-benar tidak mirip, dan 1 berarti mereka benar-benar mirip. Jadi, dalam kasus tiga kata kita di sini, kesamaan kosinus antara _pet_ dan _dog_ bisa jadi 0,75, tetapi antara _pet_ dan _lion_ mungkin 0,1.

Kemampuan untuk mengonversi kalimat menjadi penanaman\* \*yang menangkap makna semantik dan kemudian melakukan perhitungan untuk menemukan kesamaan semantik antara kalimat yang berbeda memungkinkan kita untuk membuat LLM menemukan dokumen yang paling relevan untuk menjawab pertanyaan tentang badan teks besar seperti dokumen PDF Tesla kita. Sekarang setelah Anda memahami gambaran besarnya, mari kita tinjau kembali langkah pertama (pengindeksan) dari pemrosesan awal dokumen Anda.

### Penggunaan Lain untuk Penanaman

Urutan angka dan vektor ini memiliki sejumlah properti menarik:

- Seperti yang Anda pelajari sebelumnya, jika Anda menganggap vektor sebagai menggambarkan titik dalam ruang berdimensi tinggi, titik-titik yang lebih dekat bersama memiliki makna yang lebih mirip, jadi fungsi jarak dapat digunakan untuk mengukur kesamaan.
- Kelompok titik yang dekat bersama dapat dikatakan terkait; oleh karena itu, algoritma pengelompokan (clustering) dapat digunakan untuk mengidentifikasi topik (atau kelompok titik) dan mengklasifikasikan masukan baru ke salah satu topik tersebut.
- Jika Anda merata-ratakan beberapa penanaman, penanaman rata-rata dapat dikatakan mewakili makna keseluruhan dari kelompok itu; yaitu, Anda dapat menanamkan dokumen panjang (misalnya, buku ini) dengan:
  1.  Menanamkan setiap halaman secara terpisah
  2.  Mengambil rata-rata dari penanaman semua halaman sebagai penanaman buku
- Anda dapat "berjalan" di ruang "makna" dengan menggunakan operasi matematika dasar penambahan dan pengurangan: misalnya, operasi _king – man + woman = queen_. Jika Anda mengambil makna (atau penanaman semantik) dari _king_, mengurangi makna _man_, mungkin Anda sampai pada makna yang lebih abstrak dari _monarch_, pada titik itu, jika Anda menambahkan makna _woman_, Anda telah sampai dekat dengan makna (atau penanaman) kata _queen_.
- Ada model yang dapat menghasilkan penanaman untuk konten non-teks, misalnya, gambar, video, dan suara, selain teks. Ini memungkinkan, misalnya, menemukan gambar yang paling mirip atau relevan untuk kalimat tertentu.

Kami tidak akan menjelajahi semua atribut ini dalam buku ini, tetapi berguna untuk mengetahui mereka dapat digunakan untuk sejumlah [aplikasi](https://oreil.ly/PU2C8) seperti:

**Pencarian**
: Menemukan dokumen yang paling relevan untuk kueri baru

**Pengelompokan (clustering)**
: Diberikan badan dokumen, membaginya menjadi kelompok (misalnya, topik)

**Klasifikasi**
: Menetapkan dokumen baru ke kelompok atau label yang sebelumnya diidentifikasi (misalnya, topik)

**Rekomendasi**
: Diberikan dokumen, menampilkan dokumen serupa

**Mendeteksi anomali**
: Mengidentifikasi dokumen yang sangat berbeda dari yang sebelumnya dilihat

Kami harap ini memberi Anda beberapa intuisi bahwa penanaman cukup serbaguna dan dapat dimanfaatkan dengan baik dalam proyek masa depan Anda.

## Mengonversi Dokumen Anda Menjadi Teks

Seperti disebutkan di awal bab, langkah pertama dalam pemrosesan awal dokumen Anda adalah mengonversinya ke teks. Untuk mencapai ini, Anda perlu membangun logika untuk menguraikan dan mengekstrak dokumen dengan kehilangan kualitas minimal. Untungnya, LangChain menyediakan _pemuatan dokumen (document loaders)_ yang menangani logika penguraian dan memungkinkan Anda "memuat" data dari berbagai sumber ke dalam kelas `Document` yang terdiri dari teks dan metadata terkait.

Misalnya, pertimbangkan file _.txt_ sederhana. Anda cukup mengimpor kelas `TextLoader` LangChain untuk mengekstrak teks, seperti ini:

_Python_

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./test.txt")
loader.load()
```

_JavaScript_

```javascript
import { TextLoader } from "langchain/document_loaders/fs/text"

const loader = new TextLoader("./test.txt")

const docs = await loader.load()
```

_Keluaran:_

```
[Document(page_content='text content \n', metadata={'line_number': 0, 'source':
    './test.txt'})]
```

Blok kode sebelumnya mengasumsikan bahwa Anda memiliki file bernama `test.txt` di direktori Anda saat ini. Penggunaan semua pemuat dokumen LangChain mengikuti pola serupa:

1.  Mulailah dengan memilih pemuat untuk jenis dokumen Anda dari daftar panjang [integrasi](https://oreil.ly/iLJ33).
2.  Buat instance pemuat yang dimaksud, bersama dengan parameter apa pun untuk mengonfigurasinya, termasuk lokasi dokumen Anda (biasanya jalur sistem file atau alamat web).
3.  Muat dokumen dengan memanggil `load()`, yang mengembalikan daftar dokumen siap untuk diteruskan ke tahap berikutnya (lebih lanjut tentang itu segera).

Selain file _.txt_, LangChain menyediakan pemuat dokumen untuk jenis file populer lainnya termasuk _.csv_, _.json_, dan Markdown, bersama dengan integrasi dengan platform populer seperti Slack dan Notion.

Misalnya, Anda dapat menggunakan `WebBaseLoader` untuk memuat HTML dari URL web dan menguraikannya menjadi teks.

Instal paket beautifulsoup4:

```
pip install beautifulsoup4
```

_Python_

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com/")
loader.load()
```

_JavaScript_

```javascript
// install cheerio: npm install cheerio
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"

const loader = new CheerioWebBaseLoader("https://www.langchain.com/")

const docs = await loader.load()
```

Dalam kasus penggunaan PDF Tesla kami, kita dapat memanfaatkan `PDFLoader` LangChain untuk mengekstrak teks dari dokumen PDF:

_Python_

```python
# install the pdf parsing library
# pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./test.pdf")
pages = loader.load()
```

_JavaScript_

```javascript
// install the pdf parsing library: npm install pdf-parse

import { PDFLoader } from "langchain/document_loaders/fs/pdf"

const loader = new PDFLoader("./test.pdf")

const docs = await loader.load()
```

Teks telah diekstrak dari dokumen PDF dan disimpan dalam kelas `Document`. Tetapi ada masalah. Dokumen yang dimuat panjangnya lebih dari 100.000 karakter, sehingga tidak akan muat ke dalam jendela konteks (context window) dari sebagian besar LLM atau model penanaman. Untuk mengatasi batasan ini, kita perlu membagi `Document` menjadi bagian-bagian teks yang dapat dikelola yang nantinya dapat kita ubah menjadi penanaman dan cari secara semantik, membawa kita ke langkah kedua (pengambilan).

> **Tip**
> LLM dan model penanaman dirancang dengan batas keras pada ukuran token masukan dan keluaran yang dapat mereka tangani. Batas ini biasanya disebut _jendela konteks (context window)_, dan biasanya berlaku untuk kombinasi masukan dan keluaran; artinya, jika jendela konteks adalah 100 (kita akan berbicara tentang satuan sebentar lagi), dan masukan Anda berukuran 90, keluaran paling banyak dapat berukuran 10. Jendela konteks biasanya diukur dalam jumlah token, misalnya 8.192 token. Token, seperti disebutkan di [Pendahuluan](preface01.xhtml#pr01_preface_1736545679069216), adalah representasi teks sebagai angka, dengan setiap token biasanya mencakup antara tiga dan empat karakter teks bahasa Inggris.

## Membagi Teks Anda Menjadi Bagian (Chunks)

Pada pandangan pertama mungkin tampak mudah untuk membagi badan teks besar menjadi bagian-bagian, tetapi menjaga bagian teks yang terkait secara semantik (terkait berdasarkan makna) tetap bersama adalah proses yang kompleks. Untuk memudahkan membagi dokumen besar menjadi potongan teks kecil, tetapi masih bermakna, LangChain menyediakan `RecursiveCharacterTextSplitter`, yang melakukan hal berikut:

1.  Ambil daftar pemisah, sesuai urutan kepentingan. Secara default ini adalah:
    1.  Pemisah paragraf: `\n\n`
    2.  Pemisah baris: `\n`
    3.  Pemisah kata: karakter spasi
2.  Untuk menghormati ukuran bagian yang diberikan, misalnya, 1.000 karakter, mulailah dengan memisahkan paragraf.
3.  Untuk paragraf apa pun yang lebih panjang dari ukuran bagian yang diinginkan, bagi dengan pemisah berikutnya: baris. Lanjutkan sampai semua bagian lebih kecil dari panjang yang diinginkan, atau tidak ada pemisah tambahan untuk dicoba.
4.  Keluarkan setiap bagian sebagai `Document`, dengan metadata dari dokumen asli yang diteruskan dan informasi tambahan tentang posisi dalam dokumen asli.

Mari kita lihat contoh:

_Python_

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./test.txt") # or any other loader
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splitted_docs = splitter.split_documents(docs)
```

_JavaScript_

```javascript
import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"

const loader = new TextLoader("./test.txt") // or any other loader
const docs = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})

const splittedDocs = await splitter.splitDocuments(docs)
```

Dalam kode sebelumnya, dokumen yang dibuat oleh pemuat dokumen dibagi menjadi bagian-bagian masing-masing 1.000 karakter, dengan beberapa tumpang tindih antara bagian-bagian sebesar 200 karakter untuk mempertahankan beberapa konteks. Hasilnya juga adalah daftar dokumen, di mana setiap dokumen panjangnya hingga 1.000 karakter, dibagi sesuai pembagian alami teks tertulis—paragraf, baris baru dan akhirnya, kata-kata. Ini menggunakan struktur teks untuk menjaga setiap bagian tetap menjadi cuplikan teks yang konsisten dan dapat dibaca.

`RecursiveCharacterTextSplitter` juga dapat digunakan untuk membagi bahasa kode dan Markdown menjadi bagian semantik. Ini dilakukan dengan menggunakan kata kunci spesifik untuk setiap bahasa sebagai pemisah, yang memastikan, misalnya, tubuh setiap fungsi tetap dalam bagian yang sama, alih-alih dibagi antara beberapa bagian. Biasanya, karena bahasa pemrograman memiliki lebih banyak struktur daripada teks tertulis, ada lebih sedikit kebutuhan untuk menggunakan tumpang tindih antara bagian-bagian. LangChain berisi pemisah untuk sejumlah bahasa populer, seperti Python, JS, Markdown, HTML, dan banyak lagi. Berikut contohnya:

_Python_

```python
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
```

_JavaScript_

```javascript
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"

const PYTHON_CODE = `
def hello_world():
  print("Hello, World!")

# Call the function
hello_world()
`

const pythonSplitter = RecursiveCharacterTextSplitter.fromLanguage("python", {
  chunkSize: 50,
  chunkOverlap: 0,
})
const pythonDocs = await pythonSplitter.createDocuments([PYTHON_CODE])
```

_Keluaran:_

```
[Document(page_content='def hello_world():
    print("Hello, World!")'),
    Document(page_content='# Call the function
hello_world()')]
```

Perhatikan bagaimana kita masih menggunakan `RecursiveCharacterTextSplitter` seperti sebelumnya, tetapi sekarang kita membuat instance untuk bahasa tertentu, menggunakan metode `from_language`. Yang ini menerima nama bahasa, dan parameter biasa untuk ukuran bagian, dan sebagainya. Juga perhatikan kita sekarang menggunakan metode `create_documents`, yang menerima daftar string, daripada daftar dokumen yang kita miliki sebelumnya. Metode ini berguna ketika teks yang ingin Anda bagi tidak berasal dari pemuat dokumen, jadi Anda hanya memiliki string teks mentah.

Anda juga dapat menggunakan argumen kedua opsional ke `create_documents` untuk meneruskan daftar metadata untuk dikaitkan dengan setiap string teks. Daftar metadata ini harus memiliki panjang yang sama dengan daftar string dan akan digunakan untuk mengisi bidang metadata dari setiap `Document` yang dikembalikan.

Mari kita lihat contoh untuk teks Markdown, menggunakan argumen metadata juga:

_Python_

````python
markdown_text = """
# LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

```bash
pip install langchain
```
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text],
[{"source": "https://www.langchain.com"}])
````

_JavaScript_

```javascript
const markdownText = `
# LangChain

⚡ Building applications with LLMs through composability ⚡

## Quick Install

\`\`\`bash
pip install langchain
\`\`\`

As an open source project in a rapidly developing field, we are extremely
  open to contributions.
`

const mdSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
  chunkSize: 60,
  chunkOverlap: 0,
})
const mdDocs = await mdSplitter.createDocuments(
  [markdownText],
  [{ source: "https://www.langchain.com" }]
)
```

_Keluaran:_

````
[Document(page_content='# LangChain',
    metadata={"source": "https://www.langchain.com"}),
 Document(page_content='⚡ Building applications with LLMs through composability
    ⚡', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='## Quick Install

```bash',
    metadata={"source": "https://www.langchain.com"}),
 Document(page_content='pip install langchain',
    metadata={"source": "https://www.langchain.com"}),
 Document(page_content='```', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='As an open source project in a rapidly developing field,
    we', metadata={"source": "https://www.langchain.com"}),
 Document(page_content='are extremely open to contributions.',
    metadata={"source": "https://www.langchain.com"})]
````

Perhatikan dua hal:

- Teks dibagi sepanjang titik berhenti alami dalam dokumen Markdown; misalnya, heading masuk ke satu bagian, baris teks di bawahnya dalam bagian terpisah, dan seterusnya.
- Metadata yang kami berikan dalam argumen kedua dilampirkan pada setiap dokumen yang dihasilkan, yang memungkinkan Anda melacak, misalnya, dari mana dokumen berasal dan di mana Anda dapat pergi untuk melihat aslinya.

## Menghasilkan Penanaman Teks

LangChain juga memiliki kelas `Embeddings` yang dirancang untuk berinteraksi dengan model penanaman teks—termasuk OpenAI, Cohere, dan Hugging Face—dan menghasilkan representasi vektor dari teks. Kelas ini menyediakan dua metode: satu untuk menanamkan dokumen dan satu untuk menanamkan kueri. Yang pertama mengambil daftar string teks sebagai masukan, sedangkan yang kedua mengambil satu string teks.

Berikut contoh menanamkan dokumen menggunakan [model penanaman OpenAI](https://oreil.ly/9tnzQ):

_Python_

```python
from langchain_openai import OpenAIEmbeddings

model = OpenAIEmbeddings()

embeddings = model.embed_documents([
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])
```

_JavaScript_

```javascript
import { OpenAIEmbeddings } from "@langchain/openai"

const model = new OpenAIEmbeddings()

const embeddings = await model.embedDocuments([
  "Hi there!",
  "Oh, hello!",
  "What' s your name?",
  "My friends call me World",
  "Hello World!",
])
```

_Keluaran:_

```
[
  [
    -0.004845875, 0.004899438, -0.016358767, -0.024475135, -0.017341806,
      0.012571548, -0.019156644, 0.009036391, -0.010227379, -0.026945334,
      0.022861943, 0.010321903, -0.023479493, -0.0066544134, 0.007977734,
    0.0026371893, 0.025206111, -0.012048521, 0.012943339, 0.013094575,
    -0.010580265, -0.003509951, 0.004070787, 0.008639394, -0.020631202,
    ... 1511 more items
  ]
  [
      -0.009446913, -0.013253193, 0.013174579, 0.0057552797, -0.038993083,
      0.0077763423, -0.0260478, -0.0114384955, -0.0022683728, -0.016509168,
      0.041797023, 0.01787183, 0.00552271, -0.0049789557, 0.018146982,
      -0.01542166, 0.033752076, 0.006112323, 0.023872782, -0.016535373,
      -0.006623321, 0.016116094, -0.0061090477, -0.0044155475, -0.016627092,
    ... 1511 more items
  ]
  ... 3 more items
]
```

Perhatikan bahwa Anda dapat menanamkan beberapa dokumen pada waktu yang sama; Anda harus lebih memilih ini daripada menanamkannya satu per satu, karena akan lebih efisien (karena bagaimana model ini dibangun). Anda mendapatkan kembali daftar yang berisi beberapa daftar angka—setiap daftar dalam adalah vektor atau penanaman, seperti dijelaskan di bagian sebelumnya.

Sekarang mari kita lihat contoh ujung ke ujung menggunakan tiga kemampuan yang telah kita lihat sejauh ini:

- Pemuatan dokumen (document loaders), untuk mengonversi dokumen apa pun ke teks biasa
- Pembagi teks (text splitters), untuk membagi setiap dokumen besar menjadi banyak yang lebih kecil
- Model penanaman (embeddings models), untuk membuat representasi numerik dari makna setiap bagian

Berikut kodenya:

_Python_

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

## Load the document

loader = TextLoader("./test.txt")
doc = loader.load()

"""
[
    Document(page_content='Document loaders\n\nUse document loaders to load data
        from a source as `Document\'s. A `Document` is a piece of text
and
        associated metadata. For example, there are document loaders for
        loading a simple `.txt` file, for loading the text
        contents of any web
        page, or even for loading a transcript of a YouTube video.

Every
        document loader exposes two methods:
1. "Load": load documents from
        the configured source
2. "Load and split": load documents from the
        configured source and split them using the passed in text
        splitter

They optionally implement:

3. "Lazy load": load
        documents into memory lazily
', metadata={'source': 'test.txt'})
]
"""

## Split the document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(doc)

## Generate embeddings

embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)
"""
[[0.0053587136790156364,
 -0.0004999046213924885,
  0.038883671164512634,
 -0.003001077566295862,
 -0.00900818221271038, ...], ...]
"""
```

_JavaScript_

```javascript
import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { OpenAIEmbeddings } from "@langchain/openai"

// Load the document

const loader = new TextLoader("./test.txt")
const docs = await loader.load()

// Split the document

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})
const chunks = await splitter.splitDocuments(docs)

// Generate embeddings

const model = new OpenAIEmbeddings()
await embeddings.embedDocuments(chunks.map((c) => c.pageContent))
```

Setelah Anda menghasilkan penanaman dari dokumen Anda, langkah selanjutnya adalah menyimpannya dalam basis data khusus yang dikenal sebagai penyimpanan vektor.

## Menyimpan Penanaman dalam Penyimpanan Vektor

Lebih awal di bab ini, kita membahas perhitungan kesamaan kosinus untuk mengukur kesamaan antara vektor dalam ruang vektor. Penyimpanan vektor adalah basis data yang dirancang untuk menyimpan vektor dan melakukan perhitungan kompleks, seperti kesamaan kosinus, secara efisien dan cepat.

Berbeda dengan basis data tradisional yang mengkhususkan diri dalam menyimpan data terstruktur (seperti dokumen JSON atau data yang sesuai dengan skema basis data relasional), penyimpanan vektor menangani data tidak terstruktur, termasuk teks dan gambar. Seperti basis data tradisional, penyimpanan vektor mampu melakukan operasi buat, baca, perbarui, hapus (CRUD), dan pencarian.

Penyimpanan vektor membuka kunci berbagai kasus penggunaan, termasuk aplikasi yang dapat diskalakan yang memanfaatkan AI untuk menjawab pertanyaan tentang dokumen besar, seperti diilustrasikan dalam [Gambar 2-4](#ch02_figure_4_1736545662484192).

![Memuat, menanamkan, menyimpan, dan mengambil dokumen relevan dari penyimpanan vektor](Images/lelc_0204.png)

[Gambar 2-4](#ch02_figure_4_1736545662484192) mengilustrasikan bagaimana penanaman dokumen dimasukkan ke dalam penyimpanan vektor dan bagaimana nantinya, ketika kueri dikirim, penanaman serupa diambil dari penyimpanan vektor.

Saat ini, ada banyak penyedia penyimpanan vektor untuk dipilih, masing-masing mengkhususkan diri dalam kemampuan yang berbeda. Pilihan Anda harus bergantung pada persyaratan kritis aplikasi Anda, termasuk multitenansi, kemampuan penyaringan metadata, kinerja, biaya, dan skalabilitas.

Meskipun penyimpanan vektor adalah basis data khusus yang dibangun untuk mengelola data vektor, ada beberapa kerugian bekerja dengan mereka:

- Sebagian besar penyimpanan vektor relatif baru dan mungkin tidak bertahan ujian waktu.
- Mengelola dan mengoptimalkan penyimpanan vektor dapat menyajikan kurva belajar yang relatif curam.
- Mengelola basis data terpisah menambah kompleksitas pada aplikasi Anda dan dapat menguras sumber daya berharga.

Untungnya, kemampuan penyimpanan vektor baru-baru ini diperluas ke PostgreSQL (basis data relasional sumber terbuka populer) melalui ekstensi `pgvector`. Ini memungkinkan Anda menggunakan basis data yang sama yang sudah Anda kenal dan menggerakkan tabel transaksional Anda (misalnya tabel pengguna) serta tabel pencarian vektor Anda.

### Menyiapkan PGVector

Untuk menggunakan Postgres dan PGVector Anda perlu mengikuti beberapa langkah pengaturan:

1.  Pastikan Anda telah menginstal Docker di komputer Anda, mengikuti [petunjuk untuk sistem operasi Anda](https://oreil.ly/Gn28O).
2.  Jalankan perintah berikut di terminal Anda; itu akan meluncurkan instance Postgres di komputer Anda yang berjalan di port 6024:

    ```
    docker run \
        --name pgvector-container \
        -e POSTGRES_USER=langchain \
        -e POSTGRES_PASSWORD=langchain \
        -e POSTGRES_DB=langchain \
        -p 6024:5432 \
        -d pgvector/pgvector:pg16
    ```

    Buka dasbor docker kontainer dan Anda akan melihat status berjalan hijau di sebelah `pgvector-container`.

3.  Simpan string koneksi untuk digunakan dalam kode Anda; kita akan membutuhkannya nanti:

    ```
    postgresql+psycopg://langchain:langchain@localhost:6024/langchain
    ```

### Bekerja dengan Penyimpanan Vektor

Melanjutkan dari tempat kita tinggalkan di bagian sebelumnya tentang penanaman, sekarang mari kita lihat contoh memuat, membagi, menanamkan, dan menyimpan dokumen di PGVector:

_Python_

```python
# first, pip install langchain-postgres
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# embed each chunk and insert it into the vector store
embeddings_model = OpenAIEmbeddings()
connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(documents, embeddings_model, connection=connection)
```

_JavaScript_

```javascript
import { TextLoader } from "langchain/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { OpenAIEmbeddings } from "@langchain/openai"
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector"
import { v4 as uuidv4 } from "uuid"

// Load the document, split it into chunks
const loader = new TextLoader("./test.txt")
const raw_docs = await loader.load()
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})
const docs = await splitter.splitDocuments(docs)

// embed each chunk and insert it into the vector store
const embeddings_model = new OpenAIEmbeddings()
const db = await PGVectorStore.fromDocuments(docs, embeddings_model, {
  postgresConnectionOptions: {
    connectionString:
      "postgresql://langchain:langchain@localhost:6024/langchain",
  },
})
```

Perhatikan bagaimana kita menggunakan kembali kode dari bagian sebelumnya untuk pertama memuat dokumen dengan pemuat dan kemudian membaginya menjadi bagian-bagian yang lebih kecil. Kemudian, kita membuat instance model penanaman yang ingin kita gunakan—dalam kasus ini, milik OpenAI. Perhatikan bahwa Anda dapat menggunakan model penanaman lain yang didukung LangChain di sini.

Selanjutnya, kita memiliki baris kode baru, yang membuat penyimpanan vektor diberikan dokumen, model penanaman, dan string koneksi. Ini akan melakukan beberapa hal:

- Membangun koneksi ke instance Postgres yang berjalan di komputer Anda (lihat ["Menyiapkan PGVector"](#ch02_getting_set_up_with_pgvector_1736545662501989).)
- Menjalankan pengaturan apa pun yang diperlukan, seperti membuat tabel untuk menampung dokumen dan vektor Anda, jika ini adalah pertama kalinya Anda menjalankannya.
- Membuat penanaman untuk setiap dokumen yang Anda berikan, menggunakan model yang Anda pilih.
- Menyimpan penanaman, metadata dokumen, dan konten teks dokumen di Postgres, siap untuk dicari.

Mari kita lihat bagaimana cara mencari dokumen:

_Python_

```python
db.similarity_search("query", k=4)
```

_JavaScript_

```javascript
await pgvectorStore.similaritySearch("query", 4)
```

Metode ini akan menemukan dokumen yang paling relevan (yang sebelumnya Anda indeks), dengan mengikuti proses ini:

- Kueri pencarian—dalam kasus ini, kata `query`—akan dikirim ke model penanaman untuk mengambil penanamannya.
- Kemudian, itu akan menjalankan kueri di Postgres untuk menemukan N (dalam hal ini 4) penanaman yang sebelumnya disimpan yang paling mirip dengan kueri Anda.
- Akhirnya, itu akan mengambil konten teks dan metadata yang terkait dengan masing-masing penanaman tersebut.
- Model sekarang dapat mengembalikan daftar `Document` yang diurutkan berdasarkan seberapa mirip mereka dengan kueri—yang paling mirip pertama, yang kedua paling mirip setelahnya, dan seterusnya.

Anda juga dapat menambahkan lebih banyak dokumen ke basis data yang ada. Mari kita lihat contoh:

_Python_

```python
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)
```

_JavaScript_

```javascript
const ids = [uuidv4(), uuidv4()]

await db.addDocuments(
  [
    {
      pageContent: "there are cats in the pond",
      metadata: { location: "pond", topic: "animals" },
    },
    {
      pageContent: "ducks are also found in the pond",
      metadata: { location: "pond", topic: "animals" },
    },
  ],
  { ids }
)
```

Metode `add_documents` yang kami gunakan di sini akan mengikuti proses yang mirip dengan `fromDocuments`:

- Membuat penanaman untuk setiap dokumen yang Anda berikan, menggunakan model yang Anda pilih.
- Menyimpan penanaman, metadata dokumen, dan konten teks dokumen di Postgres, siap untuk dicari.

Dalam contoh ini, kami menggunakan argumen opsional `ids` untuk menetapkan pengidentifikasi ke setiap dokumen, yang memungkinkan kami memperbarui atau menghapusnya nanti.

Berikut contoh operasi hapus:

_Python_

```python
db.delete(ids=[1])
```

_JavaScript_

```javascript
await db.delete({ ids: [ids[1]] })
```

Ini menghapus dokumen kedua yang dimasukkan dengan menggunakan Pengidentifikasi Unik Universal (UUID) miliknya. Sekarang mari kita lihat bagaimana melakukan ini dengan cara yang lebih sistematis.

## Melacak Perubahan pada Dokumen Anda

Salah satu tantangan utama dengan bekerja dengan penyimpanan vektor adalah bekerja dengan data yang secara teratur berubah, karena perubahan berarti mengindeks ulang. Dan mengindeks ulang dapat menyebabkan perhitungan ulang penanaman yang mahal dan duplikasi konten yang sudah ada sebelumnya.

Untungnya, LangChain menyediakan API pengindeksan untuk memudahkan menjaga dokumen Anda tetap sinkron dengan penyimpanan vektor Anda. API memanfaatkan kelas (`RecordManager`) untuk melacak penulisan dokumen ke dalam penyimpanan vektor. Saat mengindeks konten, hash dihitung untuk setiap dokumen dan informasi berikut disimpan di `RecordManager`:

- Hash dokumen (hash dari konten halaman dan metadata)
- Waktu penulisan
- ID sumber (setiap dokumen harus menyertakan informasi dalam metadatanya untuk menentukan sumber akhir dokumen ini).

Selain itu, API pengindeksan menyediakan mode pembersihan untuk membantu Anda memutuskan bagaimana menghapus dokumen yang ada di penyimpanan vektor. Misalnya, Jika Anda telah membuat perubahan pada cara dokumen diproses sebelum penyisipan atau jika dokumen sumber telah berubah, Anda mungkin ingin menghapus semua dokumen yang ada yang berasal dari sumber yang sama dengan dokumen baru yang sedang diindeks. Jika beberapa dokumen sumber telah dihapus, Anda akan ingin menghapus semua dokumen yang ada di penyimpanan vektor dan menggantinya dengan dokumen yang diindeks ulang.

Mode-mode tersebut adalah sebagai berikut:

- Mode `None` tidak melakukan pembersihan otomatis, memungkinkan pengguna untuk melakukan pembersihan konten lama secara manual.
- Mode `Incremental` dan `full` menghapus versi sebelumnya dari konten jika konten dari dokumen sumber atau dokumen turunan telah berubah.
- Mode `Full` akan tambahan menghapus dokumen apa pun yang tidak termasuk dalam dokumen yang saat ini diindeks.

Berikut contoh penggunaan API pengindeksan dengan basis data Postgres yang disiapkan sebagai manajer rekaman:

_Python_

```python
from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
)

# Create the schema if it doesn't exist
record_manager.create_schema()

# Create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={
        "id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={
        "id": 2, "source": "ducks.txt"}),
]

# Index the documents
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # prevent duplicate documents
    source_id_key="source",  # use the source field as the source_id
)

print("Index attempt 1:", index_1)

# second time you attempt to index, it will not add the documents again
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 2:", index_2)

# If we mutate a document, the new version will be written and all old
# versions sharing the same source will be deleted.

docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("Index attempt 3:", index_3)
```

_JavaScript_

```javascript
/** 
1. Ensure docker is installed and running (https://docs.docker.com/get-docker/)
2. Run the following command to start the postgres container:
	   
docker run \
  --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 \
  -d pgvector/pgvector:pg16
3. Use the connection string below for the postgres container
*/

import { PostgresRecordManager } from "@langchain/community/indexes/postgres"
import { index } from "langchain/indexes"
import { OpenAIEmbeddings } from "@langchain/openai"
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector"
import { v4 as uuidv4 } from "uuid"

const tableName = "test_langchain"
const connectionString =
  "postgresql://langchain:langchain@localhost:6024/langchain"
// Load the document, split it into chunks

const config = {
  postgresConnectionOptions: {
    connectionString,
  },
  tableName: tableName,
  columns: {
    idColumnName: "id",
    vectorColumnName: "vector",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
}

const vectorStore = await PGVectorStore.initialize(
  new OpenAIEmbeddings(),
  config
)

// Create a new record manager
const recordManagerConfig = {
  postgresConnectionOptions: {
    connectionString,
  },
  tableName: "upsertion_records",
}
const recordManager = new PostgresRecordManager(
  "test_namespace",
  recordManagerConfig
)

// Create the schema if it doesn't exist
await recordManager.createSchema()

const docs = [
  {
    pageContent: "there are cats in the pond",
    metadata: { id: uuidv4(), source: "cats.txt" },
  },
  {
    pageContent: "ducks are also found in the pond",
    metadata: { id: uuidv4(), source: "ducks.txt" },
  },
]

// the first attempt will index both documents
const index_attempt_1 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    // prevent duplicate documents by id from being indexed
    cleanup: "incremental",
    // the key in the metadata that will be used to identify the document
    sourceIdKey: "source",
  },
})

console.log(index_attempt_1)

// the second attempt will skip indexing because the identical documents
// already exist
const index_attempt_2 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    cleanup: "incremental",
    sourceIdKey: "source",
  },
})

console.log(index_attempt_2)

// If we mutate a document, the new version will be written and all old
// versions sharing the same source will be deleted.
docs[0].pageContent = "I modified the first document content"
const index_attempt_3 = await index({
  docsSource: docs,
  recordManager,
  vectorStore,
  options: {
    cleanup: "incremental",
    sourceIdKey: "source",
  },
})

console.log(index_attempt_3)
```

Pertama, Anda membuat manajer rekaman, yang melacak dokumen mana yang telah diindeks sebelumnya. Kemudian Anda menggunakan fungsi `index` untuk menyinkronkan penyimpanan vektor Anda dengan daftar dokumen baru. Dalam contoh ini, kami menggunakan mode inkremental, sehingga dokumen apa pun yang memiliki ID yang sama dengan yang sebelumnya akan diganti dengan versi baru.

## Optimasi Pengindeksan

Tahap pengindeksan RAG dasar melibatkan pembagian teks naif (naive text splitting) dan penanaman bagian dari dokumen tertentu. Namun, pendekatan dasar ini mengarah pada hasil pengambilan yang tidak konsisten dan kejadian halusinasi yang relatif tinggi, terutama ketika sumber data berisi gambar dan tabel.

Ada berbagai strategi untuk meningkatkan akurasi dan kinerja tahap pengindeksan. Kami akan membahas tiga di antaranya di bagian selanjutnya: MultiVectorRetriever, RAPTOR, dan ColBERT.

### MultiVectorRetriever

Dokumen yang berisi campuran teks dan tabel tidak dapat hanya dibagi berdasarkan teks menjadi bagian-bagian dan ditanamkan sebagai konteks: seluruh tabel dapat dengan mudah hilang. Untuk mengatasi masalah ini, kita dapat memisahkan dokumen yang ingin kita gunakan untuk sintesis jawaban, dari referensi yang ingin kita gunakan untuk pengambil (retriever). [Gambar 2-5](#ch02_figure_5_1736545662484212) mengilustrasikan caranya.

![Mengindeks banyak representasi dari satu dokumen](Images/lelc_0205.png)

Misalnya, dalam kasus dokumen yang berisi tabel, kita dapat pertama menghasilkan dan menanamkan ringkasan elemen tabel, memastikan setiap ringkasan berisi referensi `id` ke tabel mentah penuh. Selanjutnya, kita menyimpan tabel mentah yang direferensikan dalam penyimpanan dokumen (docstore) terpisah. Akhirnya, ketika kueri pengguna mengambil ringkasan tabel, kita meneruskan seluruh tabel mentah yang direferensikan sebagai konteks ke petunjuk akhir yang dikirim ke LLM untuk sintesis jawaban. Pendekatan ini memungkinkan kita untuk memberikan model dengan konteks penuh informasi yang diperlukan untuk menjawab pertanyaan.

Berikut contohnya. Pertama, mari kita gunakan LLM untuk menghasilkan ringkasan dokumen:

_Python_

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "summaries"
embeddings_model = OpenAIEmbeddings()
# Load the document
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()

print("length of loaded docs: ", len(docs[0].page_content))
# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# The rest of your code remains the same, starting from:
prompt_text = "Summarize the following document:\n\n{doc}"

prompt = ChatPromptTemplate.from_template(prompt_text)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
summarize_chain = {
    "doc": lambda x: x.page_content} | prompt | llm | StrOutputParser()

# batch the chain across the chunks
summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})
```

Selanjutnya, mari kita definisikan penyimpanan vektor dan penyimpanan dokumen untuk menyimpan ringkasan mentah dan penanamannya:

_Python_

```python
# The vectorstore to use to index the child chunks
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# indexing the summaries in our vector store, whilst retaining the original
# documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Changed from summaries to chunks since we need same length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# Each summary is linked to the original document by the doc_id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add the document summaries to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)

# Store the original documents in the document store, linked to their summaries
# via doc_ids
# This allows us to first search summaries efficiently, then fetch the full
# docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))

# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search(
    "chapter on philosophy", k=2)
```

Akhirnya, mari kita ambil dokumen konteks penuh yang relevan berdasarkan kueri:

_Python_

```python
# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("chapter on philosophy")

```

Berikut implementasi lengkap dalam JavaScript:

_JavaScript_

```javascript
import * as uuid from "uuid"
import { MultiVectorRetriever } from "langchain/retrievers/multi_vector"
import { OpenAIEmbeddings } from "@langchain/openai"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { InMemoryStore } from "@langchain/core/stores"
import { TextLoader } from "langchain/document_loaders/fs/text"
import { Document } from "@langchain/core/documents"
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector"
import { ChatOpenAI } from "@langchain/openai"
import { PromptTemplate } from "@langchain/core/prompts"
import { RunnableSequence } from "@langchain/core/runnables"
import { StringOutputParser } from "@langchain/core/output_parsers"

const connectionString =
  "postgresql://langchain:langchain@localhost:6024/langchain"
const collectionName = "summaries"

const textLoader = new TextLoader("./test.txt")
const parentDocuments = await textLoader.load()
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 10000,
  chunkOverlap: 20,
})
const docs = await splitter.splitDocuments(parentDocuments)

const prompt = PromptTemplate.fromTemplate(
  `Summarize the following document:\n\n{doc}`
)

const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo" })

const chain = RunnableSequence.from([
  { doc: (doc) => doc.pageContent },
  prompt,
  llm,
  new StringOutputParser(),
])

// batch summarization chain across the chunks
const summaries = await chain.batch(docs, {
  maxConcurrency: 5,
})

const idKey = "doc_id"
const docIds = docs.map((_) => uuid.v4())
// create summary docs with metadata linking to the original docs
const summaryDocs = summaries.map((summary, i) => {
  const summaryDoc = new Document({
    pageContent: summary,
    metadata: {
      [idKey]: docIds[i],
    },
  })
  return summaryDoc
})

// The byteStore to use to store the original chunks
const byteStore = new InMemoryStore()

// vector store for the summaries
const vectorStore = await PGVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings(),
  {
    postgresConnectionOptions: {
      connectionString,
    },
  }
)

const retriever = new MultiVectorRetriever({
  vectorstore: vectorStore,
  byteStore,
  idKey,
})

const keyValuePairs = docs.map((originalDoc, i) => [docIds[i], originalDoc])

// Use the retriever to add the original chunks to the document store
await retriever.docstore.mset(keyValuePairs)

// Vectorstore alone retrieves the small chunks
const vectorstoreResult = await retriever.vectorstore.similaritySearch(
  "chapter on philosophy",
  2
)
console.log(`summary: ${vectorstoreResult[0].pageContent}`)
console.log(
  `summary retrieved length: ${vectorstoreResult[0].pageContent.length}`
)

// Retriever returns larger chunk result
const retrieverResult = await retriever.invoke("chapter on philosophy")
console.log(
  `multi-vector retrieved chunk length: ${retrieverResult[0].pageContent.length}`
)
```

### RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Sistem RAG perlu menangani pertanyaan tingkat bawah yang merujuk pada fakta spesifik yang ditemukan dalam satu dokumen atau pertanyaan tingkat tinggi yang menyaring ide-ide yang menjangkau banyak dokumen. Menangani kedua jenis pertanyaan dapat menjadi tantangan dengan pengambilan k-tetangga terdekat (k-NN) tipikal di atas bagian dokumen.

_Recursive abstractive processing for tree-organized retrieval_ (RAPTOR) adalah strategi efektif yang melibatkan membuat ringkasan dokumen yang menangkap konsep tingkat tinggi, menanamkan dan mengelompokkan dokumen-dokumen tersebut, dan kemudian [meringkas setiap kelompok](https://oreil.ly/VdIpJ).^[2](#id538)] Ini dilakukan secara rekursif, menghasilkan pohon ringkasan dengan konsep yang semakin tinggi tingkatnya. Ringkasan dan dokumen awal kemudian diindeks bersama, memberikan cakupan di seluruh pertanyaan pengguna tingkat bawah hingga tinggi. [Gambar 2-6](#ch02_figure_6_1736545662484232) mengilustrasikan.

![Meringkas dokumen secara rekursif](Images/lelc_0206.png)

### ColBERT: Mengoptimalkan Penanaman

Salah satu tantangan menggunakan model penanaman selama tahap pengindeksan adalah bahwa mereka mengompres teks menjadi representasi panjang-tetap (vektor) yang menangkap konten semantik dari dokumen. Meskipun kompresi ini berguna untuk pengambilan, menanamkan konten yang tidak relevan atau berlebihan dapat menyebabkan halusinasi dalam keluaran LLM akhir.

Salah satu solusi untuk masalah ini adalah melakukan hal berikut:

1.  Hasilkan penanaman kontekstual untuk setiap token dalam dokumen dan kueri.
2.  Hitung dan skor kesamaan antara setiap token kueri dan semua token dokumen.
3.  Jumlahkan skor kesamaan maksimum dari setiap penanaman kueri ke salah satu penanaman dokumen untuk mendapatkan skor untuk setiap dokumen.

Ini menghasilkan pendekatan penanaman yang granular dan efektif untuk pengambilan yang lebih baik. Untungnya, model penanaman yang disebut ColBERT\*\* \*\*menwujudkan solusi untuk masalah ini.^[3](#id542)]

Berikut cara kita dapat memanfaatkan ColBERT untuk penanaman optimal dari data kita:

_Python_

```python
# RAGatouille is a library that makes it simple to use ColBERT
# pip install -U ragatouille

from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

import requests

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")

## Create an index
RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)

#query
results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
results

#utilize langchain retriever
retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")
```

Dengan menggunakan ColBERT, Anda dapat meningkatkan relevansi dokumen yang diambil yang digunakan sebagai konteks oleh LLM.

## Ringkasan

Dalam bab ini, Anda telah mempelajari cara mempersiapkan dan memproses dokumen Anda untuk aplikasi LLM menggunakan berbagai modul LangChain. Pemuatan dokumen (document loaders) memungkinkan Anda mengekstrak teks dari sumber data Anda, pembagi teks (text splitters) membantu Anda membagi dokumen Anda menjadi bagian-bagian yang serupa secara semantik, dan model penanaman (embeddings models) mengonversi teks Anda menjadi representasi vektor dari maknanya.

Secara terpisah, penyimpanan vektor (vector stores) memungkinkan Anda melakukan operasi CRUD pada penanaman ini bersama dengan perhitungan kompleks untuk menghitung bagian teks yang serupa secara semantik. Akhirnya, strategi optimasi pengindeksan memungkinkan aplikasi AI Anda meningkatkan kualitas penanaman dan melakukan pengambilan dokumen yang akurat yang berisi data semiterstruktur termasuk tabel.

Di [Bab 3](ch03.xhtml#ch03_rag_part_ii_chatting_with_your_data_1736545666793580), Anda akan mempelajari cara mengambil bagian dokumen yang paling mirip dari penyimpanan vektor Anda berdasarkan kueri Anda, memberikannya sebagai konteks\* \*yang dapat dilihat model, dan kemudian menghasilkan keluaran yang akurat.

---

1.  Arvind Neelakantan et al., ["Text and Code Embeddings by Contrastive Pre-Training"](https://oreil.ly/YOVmh), arXiv, 21 Januari 2022.
2.  Parth Sarthi et al., ["RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"](https://oreil.ly/hS4NB), arXiv, 31 Januari 2024. Makalah diterbitkan di ICLR 2024.
3.  Keshav Santhanam et al., ["ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"](https://oreil.ly/9spW2), arXiv, 2 Desember 2021.
