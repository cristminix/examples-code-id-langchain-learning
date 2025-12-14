# Kata Pengantar

Pada tanggal 30 November 2022, perusahaan OpenAI yang berbasis di San Francisco [merilis ChatGPT](https://oreil.ly/uAnsr) ke publik—bot AI viral yang bisa membuat konten, menjawab pertanyaan, dan menyelesaikan masalah seperti manusia. Dalam dua bulan setelah peluncurannya, ChatGPT menarik lebih dari [100 juta pengguna aktif bulanan](https://oreil.ly/ATsLe), menjadi tingkat adopsi tercepat untuk aplikasi teknologi konsumen baru (sejauh ini). ChatGPT adalah pengalaman chatbot yang ditenagai oleh versi model bahasa besar (large language model/LLM) keluarga GPT‑3.5 OpenAI yang telah disetel untuk instruksi dan dialog. Kita akan segera membahas definisi konsep-konsep ini.

> Catatan
> Membangun aplikasi LLM dengan atau tanpa LangChain membutuhkan penggunaan LLM. Dalam buku ini kita akan menggunakan [API OpenAI](https://oreil.ly/-YYoR) sebagai penyedia LLM yang kita pakai dalam contoh kode (harga tercantum di platformnya). Salah satu keuntungan bekerja dengan LangChain adalah kamu bisa mengikuti semua contoh ini menggunakan OpenAI atau penyedia LLM komersial atau sumber terbuka alternatif.

Tiga bulan kemudian, OpenAI [merilis API ChatGPT](https://oreil.ly/DwU7R), memberi pengembang akses ke kemampuan obrolan dan ucapan-ke-teks. Ini memicu tak terhitung banyaknya aplikasi baru dan perkembangan teknis di bawah payung longgar yang disebut _AI generatif_.

Sebelum kita mendefinisikan AI generatif dan LLM, mari kita sentuh konsep _pembelajaran mesin_ (machine learning/ML). Beberapa _algoritma_ komputer (bayangkan resep yang bisa diulang untuk mencapai tugas yang sudah ditentukan, seperti mengurutkan setumpuk kartu) ditulis langsung oleh insinyur perangkat lunak. Algoritma komputer lainnya justru _dipelajari_ dari sejumlah besar contoh pelatihan—pekerjaan insinyur perangkat lunak bergeser dari menulis algoritma itu sendiri ke menulis logika pelatihan yang menciptakan algoritma. Banyak perhatian di bidang ML diarahkan untuk mengembangkan algoritma guna memprediksi berbagai hal, dari cuaca besok hingga rute pengiriman paling efisien untuk pengemudi Amazon.

Dengan munculnya LLM dan model generatif lain (seperti model difusi untuk menghasilkan gambar, yang tidak kita bahas dalam buku ini), teknik ML yang sama kini diterapkan pada masalah menghasilkan konten baru, misalnya paragraf teks atau gambar baru, yang sekaligus unik dan diilhami oleh contoh-contoh dalam data pelatihan. LLM khususnya adalah model generatif yang dikhususkan untuk menghasilkan teks.

LLM memiliki dua perbedaan lain dari algoritma ML sebelumnya:

- Mereka dilatih dengan data yang jauh lebih banyak; melatih salah satu model ini dari awal akan sangat mahal.
- Mereka lebih serbaguna.

Model pembangkit teks yang sama bisa digunakan untuk ringkasan, terjemahan, klasifikasi, dan sebagainya, sedangkan model ML sebelumnya biasanya dilatih dan digunakan untuk tugas spesifik.

Kedua perbedaan ini menyebabkan pekerjaan insinyur perangkat lunak bergeser lagi, dengan semakin banyak waktu yang dikhususkan untuk mencari cara membuat LLM bekerja untuk kasus penggunaan mereka. Dan itulah yang dimaksud dengan LangChain.

Pada akhir 2023, muncul LLM pesaing, termasuk Claude dari Anthropic dan Bard (kemudian berganti nama menjadi Gemini) dari Google, yang memberikan akses yang lebih luas ke kemampuan baru ini. Selanjutnya, ribuan startup sukses dan perusahaan besar telah memasukkan API AI generatif untuk membangun aplikasi untuk berbagai kasus penggunaan, mulai dari chatbot dukungan pelanggan hingga menulis dan mendebug kode.

Pada tanggal 22 Oktober 2022, Harrison Chase [mempublikasikan commit pertama](https://oreil.ly/mCdYZ) di GitHub untuk pustaka sumber terbuka LangChain. LangChain berawal dari kesadaran bahwa aplikasi LLM yang paling menarik perlu menggunakan LLM bersama dengan ["sumber komputasi atau pengetahuan lain"](https://oreil.ly/uXiPi). Misalnya, kamu bisa mencoba meminta LLM menghasilkan jawaban untuk pertanyaan ini:

```
Berapa banyak bola yang tersisa setelah membagi 1.234 bola secara merata kepada 123 orang?
```

Kamu mungkin akan kecewa dengan kemampuan matematikanya. Namun, jika kamu memasangkannya dengan fungsi kalkulator, kamu bisa menginstruksikan LLM untuk mengubah kata pertanyaan menjadi masukan yang bisa ditangani kalkulator:

```
1.234 % 123
```

Lalu kamu bisa meneruskannya ke fungsi kalkulator dan mendapatkan jawaban akurat untuk pertanyaan awalmu. LangChain adalah pustaka pertama (dan, pada saat penulisan, yang terbesar) yang menyediakan blok bangunan seperti itu serta peralatan untuk menggabungkannya dengan andal ke dalam aplikasi yang lebih besar. Sebelum membahas apa yang diperlukan untuk membangun aplikasi yang menarik dengan alat-alat baru ini, mari kita lebih mengenal LLM dan LangChain.

## Pengenalan Singkat tentang LLM

Dengan bahasa yang sederhana, LLM adalah algoritma terlatih yang menerima masukan teks dan memprediksi serta menghasilkan keluaran teks seperti manusia. Pada dasarnya, mereka berperilaku seperti fitur penyelesaian otomatis (autocomplete) yang dikenal di banyak ponsel pintar, tetapi diambil ke tingkat ekstrem.

Mari kita urai istilah _model bahasa besar_:

- _Besar_ mengacu pada ukuran model ini dalam hal data pelatihan dan parameter yang digunakan selama proses pembelajaran. Misalnya, model GPT‑3 OpenAI mengandung 175 miliar _parameter_, yang dipelajari dari pelatihan pada 45 terabita data teks.^[Tom B. Brown dkk., ["Language Models Are Few-Shot Learners"](https://oreil.ly/1qoM6), arXiv, 22 Juli 2020.] _Parameter_ dalam model jaringan saraf terdiri dari angka-angka yang mengendalikan keluaran setiap _neuron_ dan bobot relatif koneksinya dengan neuron tetangga. (Neuron mana yang terhubung ke neuron lain bervariasi untuk setiap arsitektur jaringan saraf dan berada di luar cakupan buku ini.)
- _Model bahasa_ mengacu pada algoritma komputer yang dilatih untuk menerima teks tertulis (dalam bahasa Inggris atau bahasa lain) dan menghasilkan keluaran juga sebagai teks tertulis (dalam bahasa yang sama atau berbeda). Ini adalah _jaringan saraf_, sejenis model ML yang menyerupai konsepsi bergaya otak manusia, dengan keluaran akhir dihasilkan dari gabungan keluaran individu banyak fungsi matematika sederhana, disebut _neuron_, dan interkoneksinya. Jika banyak neuron ini diatur dengan cara tertentu, dengan proses pelatihan dan data pelatihan yang tepat, ini menghasilkan model yang mampu menafsirkan makna kata dan kalimat individual, yang memungkinkan kita menggunakannya untuk menghasilkan teks tertulis yang masuk akal dan dapat dibaca.

Karena dominasi bahasa Inggris dalam data pelatihan, sebagian besar model lebih baik dalam bahasa Inggris daripada bahasa lain dengan jumlah penutur yang lebih sedikit. Dengan "lebih baik" maksudnya lebih mudah membuat mereka menghasilkan keluaran yang diinginkan dalam bahasa Inggris. Ada LLM yang dirancang untuk keluaran multibahasa, seperti [BLOOM](https://oreil.ly/Nq7w0), yang menggunakan proporsi lebih besar data pelatihan dalam bahasa lain. Menariknya, perbedaan kinerja antar bahasa tidak sebesar yang diperkirakan, bahkan dalam LLM yang dilatih pada korpus pelatihan yang didominasi bahasa Inggris. Peneliti menemukan bahwa LLM mampu mentransfer sebagian pemahaman semantiknya ke bahasa lain.^[Xiang Zhang dkk., ["Don't Trust ChatGPT When Your Question Is Not in English: A Study of Multilingual Abilities and Types of LLMs"](https://oreil.ly/u5Cy1), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 6–10 Desember 2023.]

Secara gabungan, _model bahasa besar_ adalah contoh model bahasa tujuan umum yang besar yang dilatih pada sejumlah besar teks. Dengan kata lain, model-model ini telah belajar dari pola dalam kumpulan data teks yang besar—buku, artikel, forum, dan sumber tersedia publik lainnya—untuk melakukan tugas terkait teks umum. Tugas-tugas ini mencakup pembangkitan teks, ringkasan, terjemahan, klasifikasi, dan lainnya.

Katakanlah kita menginstruksikan LLM untuk melengkapi kalimat berikut:

```
Ibu kota Inggris adalah _______.
```

LLM akan mengambil teks masukan itu dan memprediksi jawaban keluaran yang benar sebagai `London`. Ini terlihat seperti sihir, tetapi sebenarnya bukan. Di balik layar, LLM memperkirakan probabilitas urutan kata berdasarkan urutan kata sebelumnya.

> Tips
> Secara teknis, model membuat prediksi berdasarkan token, bukan kata. Sebuah _token_ mewakili unit teks atomik. Token dapat mewakili karakter individu, kata, subkata, atau bahkan unit linguistik yang lebih besar, tergantung pada pendekatan tokenisasi yang digunakan. Misalnya, menggunakan tokenizer GPT‑3.5 (disebut `cl100k`), frasa _good morning dearest friend_ akan terdiri dari [lima token](https://oreil.ly/dU83b) (menggunakan `_` untuk menunjukkan karakter spasi):

> `Good`
> : Dengan ID token `19045`

> `_morning`
> : Dengan ID token `6693`

> `_de`
> : Dengan ID token `409`

> `arest`
> : Dengan ID token `15795`

> `_friend`
> : Dengan ID token `4333`

> Biasanya tokenizer dilatih dengan tujuan mengodekan kata-kata paling umum menjadi satu token, misalnya, kata _morning_ dikodekan sebagai token `6693`. Kata yang kurang umum, atau kata dalam bahasa lain (biasanya tokenizer dilatih pada teks bahasa Inggris), memerlukan beberapa token untuk mengodekannya. Contohnya, kata _dearest_ dikodekan sebagai token `409, 15795`. Satu token rata-rata mencakup empat karakter teks untuk teks bahasa Inggris umum, atau kira-kira tiga perempat kata.

Mesin penggerak di balik daya prediktif LLM dikenal sebagai _arsitektur jaringan saraf transformer_.^[Untuk informasi lebih lanjut, lihat Ashish Vaswani dkk., ["Attention Is All You Need "](https://oreil.ly/Frtul), arXiv, 12 Juni 2017.] Arsitektur transformer memungkinkan model menangani urutan data, seperti kalimat atau baris kode, dan membuat prediksi tentang kata(‑kata) berikutnya yang paling mungkin dalam urutan tersebut. Transformer dirancang untuk memahami konteks setiap kata dalam sebuah kalimat dengan mempertimbangkannya dalam hubungannya dengan setiap kata lain. Hal ini memungkinkan model membangun pemahaman komprehensif tentang makna sebuah kalimat, paragraf, dan seterusnya (dengan kata lain, urutan kata) sebagai makna gabungan bagian-bagiannya dalam hubungan satu sama lain.

Jadi, ketika model melihat urutan kata _ibu kota Inggris adalah_, ia membuat prediksi berdasarkan contoh serupa yang dilihatnya selama pelatihan. Dalam korpus pelatihan model, kata _England_ (atau token yang mewakilinya) sering muncul dalam kalimat di tempat yang mirip dengan kata seperti _France_, _United States_, _China_. Kata _capital_ akan muncul dalam data pelatihan di banyak kalimat yang juga mengandung kata seperti _England_, _France_, dan _US_, serta kata seperti _London_, _Paris_, _Washington_. Pengulangan ini selama pelatihan model menghasilkan kemampuan untuk memprediksi dengan benar bahwa kata berikutnya dalam urutan seharusnya _London_.

Instruksi dan teks masukan yang kamu berikan ke model disebut _prompt_. Pemberian prompt dapat berdampak signifikan pada kualitas keluaran dari LLM. Ada beberapa praktik terbaik untuk _desain prompt_ atau _rekayasa prompt_, termasuk memberikan instruksi yang jelas dan ringkas dengan contoh kontekstual, yang akan kita bahas nanti dalam buku ini. Sebelum kita lebih jauh membahas prompting, mari kita lihat beberapa jenis LLM yang tersedia untuk kamu gunakan.

Jenis dasar, dari mana semua jenis lain diturunkan, umumnya dikenal sebagai _LLM yang telah dilatih sebelumnya_ (pretrained LLM): ia telah dilatih pada jumlah teks yang sangat besar (ditemukan di internet, buku, surat kabar, kode, transkrip video, dan sebagainya) dengan cara diawasi‑sendiri (self‑supervised). Artinya—tidak seperti dalam ML yang diawasi (supervised), di mana sebelum pelatihan peneliti perlu menyusun kumpulan data pasangan _masukan_ ke _keluaran yang diharapkan_—untuk LLM pasangan‑pasangan itu disimpulkan dari data pelatihan. Faktanya, satu‑satunya cara yang layak untuk menggunakan kumpulan data yang begitu besar adalah dengan menyusun pasangan‑pasangan itu dari data pelatihan secara otomatis. Dua teknik untuk melakukan ini melibatkan meminta model melakukan hal berikut:

Memprediksi kata berikutnya
: Menghapus kata terakhir dari setiap kalimat dalam data pelatihan, dan itu menghasilkan pasangan _masukan_ dan _keluaran yang diharapkan_, misalnya \_Ibu kota Inggris adalah \_\_\__ dan \_London_.

Memprediksi kata yang hilang
: Serupa, jika kamu mengambil setiap kalimat dan menghilangkan satu kata dari tengah, kamu sekarang memiliki pasangan masukan dan keluaran yang diharapkan lainnya, misalnya _The \_\_\_ of England is London_ dan _capital_.

Model‑model ini cukup sulit digunakan apa adanya, mereka mengharuskan kamu memancing respons dengan awalan yang sesuai. Misalnya, jika kamu ingin tahu ibu kota Inggris, kamu mungkin mendapatkan respons dengan memprompt model dengan _Ibu kota Inggris adalah_, tetapi tidak dengan _Apa ibu kota Inggris?_ yang lebih alami.

### LLM yang Disetel untuk Instruksi

[Para peneliti](https://oreil.ly/lP6hr) telah membuat LLM yang telah dilatih sebelumnya lebih mudah digunakan dengan pelatihan lebih lanjut (pelatihan tambahan yang diterapkan di atas pelatihan panjang dan mahal yang dijelaskan di bagian sebelumnya), juga dikenal sebagai _menyetel‑halus_ (fine‑tuning) mereka pada hal berikut:

Kumpulan data khusus tugas
: Ini adalah kumpulan data pasangan pertanyaan/jawaban yang disusun manual oleh peneliti, memberikan contoh respons yang diinginkan untuk pertanyaan umum yang mungkin diprompt oleh pengguna akhir. Misalnya, kumpulan data mungkin berisi pasangan berikut: *Q: Apa ibu kota Inggris? A: Ibu kota Inggris adalah London. *Berbeda dengan kumpulan data pelatihan awal, ini disusun manual, sehingga ukurannya jauh lebih kecil:

Pembelajaran penguatan dari umpan balik manusia (reinforcement learning from human feedback/RLHF)
: Melalui penggunaan [metode RLHF](https://oreil.ly/lrlAK), kumpulan data yang disusun manual itu diperkaya dengan umpan balik pengguna yang diterima pada keluaran yang dihasilkan model. Misalnya, pengguna A lebih menyukai _Ibu kota Inggris adalah London_ daripada _London adalah ibu kota Inggris_ sebagai jawaban untuk pertanyaan sebelumnya.

Penyetelan instruksi telah menjadi kunci untuk memperluas jumlah orang yang dapat membangun aplikasi dengan LLM, karena mereka sekarang dapat diprompt dengan _instruksi_, sering kali dalam bentuk pertanyaan seperti, _Apa ibu kota Inggris?_, dibandingkan dengan _Ibu kota Inggris adalah_.

### LLM yang Disetel untuk Dialog

Model yang disesuaikan untuk tujuan dialog atau obrolan adalah [penyempurnaan lebih lanjut](https://oreil.ly/1DxW6) dari LLM yang disetel untuk instruksi. Penyedia LLM berbeda menggunakan teknik berbeda, jadi ini belum tentu benar untuk semua _model obrolan_ (chat model), tetapi biasanya ini dilakukan melalui:

Kumpulan data dialog
: Kumpulan data _penyetelan‑halus_ yang disusun manual diperluas untuk mencakup lebih banyak contoh interaksi dialog banyak‑giliran, yaitu urutan pasangan prompt‑balasan.

Format obrolan
: Format masukan dan keluaran model diberikan lapisan struktur di atas teks bebas, yang membagi teks menjadi bagian‑bagian yang terkait dengan peran (dan opsional metadata lain seperti nama). Biasanya, peran yang tersedia adalah _sistem_ (untuk instruksi dan pengaturan tugas), _pengguna_ (tugas atau pertanyaan sebenarnya), dan _asisten_ (untuk keluaran model). Metode ini berevolusi dari [teknik rekayasa prompt awal](https://oreil.ly/dINx0) dan memudahkan menyesuaikan keluaran model sekaligus membuat model lebih sulit membingungkan masukan pengguna dengan instruksi. Membingungkan masukan pengguna dengan instruksi sebelumnya juga dikenal sebagai _jailbreaking_, yang dapat, misalnya, menyebabkan prompt yang dibuat dengan hati‑hati, mungkin termasuk rahasia dagang, terbuka ke pengguna akhir.

### LLM yang Disetel‑Halus

LLM yang disetel‑halus dibuat dengan mengambil LLM dasar dan melatihnya lebih lanjut pada kumpulan data milik untuk tugas spesifik. Secara teknis, LLM yang disetel untuk instruksi dan LLM yang disetel untuk dialog adalah LLM yang disetel‑halus, tetapi istilah "LLM yang disetel‑halus" biasanya digunakan untuk menggambarkan LLM yang disetel oleh pengembang untuk tugas spesifik mereka. Misalnya, model dapat disetel‑halus untuk mengekstrak sentimen, faktor risiko, dan angka keuangan kunci dari laporan tahunan perusahaan publik secara akurat. Biasanya, model yang disetel‑halus memiliki kinerja yang lebih baik pada tugas yang dipilih dengan mengorbankan hilangnya keumuman. Artinya, mereka menjadi kurang mampu menjawab kueri pada tugas yang tidak terkait.

Sepanjang sisa buku ini, ketika kita menggunakan istilah _LLM_, yang kita maksud adalah LLM yang disetel untuk instruksi, dan untuk _model obrolan_ yang kita maksud adalah LLM yang diinstruksikan‑dialog, seperti yang didefinisikan sebelumnya di bagian ini. Ini harus menjadi andalanmu ketika menggunakan LLM—alat pertama yang kamu raih ketika memulai aplikasi LLM baru.

Sekarang mari kita bahas secara singkat beberapa teknik prompting LLM umum sebelum menyelami LangChain.

## Pengenalan Singkat tentang Prompting

Seperti yang kita singgung sebelumnya, tugas utama insinyur perangkat lunak yang bekerja dengan LLM bukanlah melatih LLM, atau bahkan menyetel‑halusnya (biasanya), melainkan mengambil LLM yang sudah ada dan mencari cara membuatnya menyelesaikan tugas yang kamu butuhkan untuk aplikasimu. Ada penyedia komersial LLM, seperti OpenAI, Anthropic, dan Google, serta LLM sumber terbuka ([Llama](https://oreil.ly/ld3Fu), [Gemma](https://oreil.ly/RGKfi), dan lainnya), yang dirilis gratis untuk dibangun oleh orang lain. Menyesuaikan LLM yang ada untuk tugasmu disebut _rekayasa prompt_.

Banyak teknik prompting telah dikembangkan dalam dua tahun terakhir, dan secara luas, buku ini adalah tentang cara melakukan rekayasa prompt dengan LangChain—bagaimana menggunakan LangChain untuk membuat LLM melakukan apa yang kamu bayangkan. Tetapi sebelum kita masuk ke LangChain secara tepat, ada baiknya membahas beberapa teknik ini terlebih dahulu (dan kami mohon maaf sebelumnya jika [teknik prompting favoritmu](https://oreil.ly/8uGK_) tidak tercantum di sini; terlalu banyak untuk dibahas).

Untuk mengikuti bagian ini, kami sarankan menyalin prompt‑prompt ini ke OpenAI Playground untuk mencobanya sendiri:

1. Buat akun untuk API OpenAI di [http://platform.openai.com](http://platform.openai.com), yang akan memungkinkanmu menggunakan LLM OpenAI secara terprogram, yaitu menggunakan API dari kode Python atau JavaScript‑mu. Itu juga akan memberi kamu akses ke OpenAI Playground, di mana kamu bisa bereksperimen dengan prompt dari peramban web.
2. Jika perlu, tambahkan detail pembayaran untuk akun OpenAI barumu. OpenAI adalah penyedia LLM komersial dan mengenakan biaya untuk setiap kali kamu menggunakan model mereka melalui API OpenAI atau melalui Playground. Kamu dapat menemukan harga terbaru di [situs web mereka](https://oreil.ly/MiKRD). Dalam dua tahun terakhir, harga untuk menggunakan model OpenAI telah turun secara signifikan seiring diperkenalkannya kemampuan dan optimisasi baru.
3. Pergi ke [OpenAI Playground](https://oreil.ly/rxiAG) dan kamu siap mencoba prompt‑prompt berikut untuk dirimu sendiri. Kita akan menggunakan API OpenAI di seluruh buku ini.
4. Setelah masuk ke Playground, kamu akan melihat panel prasetel di sisi kanan layar, termasuk model pilihanmu. Jika kamu melihat lebih jauh ke bawah panel, kamu akan melihat Temperature di bawah judul "Model configuration". Geser tombol Temperature dari tengah ke kiri hingga angkanya menunjukkan 0.00. Intinya, temperature mengendalikan keacakan keluaran LLM. Semakin rendah temperature, keluaran model semakin deterministik.

Sekarang mari kita lanjutkan ke prompt‑promptnya!

### Prompting Zero‑Shot

Teknik prompting pertama dan paling langsung terdiri dari sekadar menginstruksikan LLM untuk melakukan tugas yang diinginkan:

```
Berapa usia presiden ke‑30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Ini biasanya yang harus kamu coba pertama kali, dan biasanya akan berhasil untuk pertanyaan sederhana, terutama ketika jawabannya kemungkinan ada dalam beberapa data pelatihan. Jika kita memprompt `gpt‑3.5‑turbo` OpenAI dengan prompt sebelumnya, yang berikut dikembalikan:

```
Presiden ke‑30 Amerika Serikat, Calvin Coolidge, berusia 48 tahun ketika ibu mertuanya meninggal pada tahun 1926.
```

> Catatan
> Kamu mungkin mendapatkan hasil yang berbeda dari yang kita dapatkan. Ada unsur keacakan dalam cara LLM menghasilkan respons, dan OpenAI mungkin telah memperbarui model pada saat kamu mencobanya.

Meskipun model benar mengidentifikasi presiden ke‑30, jawabannya tidak sepenuhnya benar. Seringkali kamu harus mengulangi prompt dan respons untuk mendapatkan sistem yang andal. Bagian‑bagian selanjutnya memperluas caranya.

### Rantai‑Pemikiran (Chain‑of‑Thought)

Iterasi yang sangat berguna adalah dengan lebih lanjut menginstruksikan model untuk [meluangkan waktu untuk _berpikir_](https://oreil.ly/8xWcs). Teknik ini telah terbukti meningkatkan kinerja pada berbagai tugas. Menariknya, makalah terbaru^[Ryan Liu dkk. ["Mind Your Step (by Step): Chain‑of‑Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse"](https://oreil.ly/UHFp9), arXiv, 8 November 2024.] menyarankan bahwa teknik ini juga dapat mengurangi kinerja LLM pada tugas‑tugas di mana berpikir mengurangi kinerja manusia. Disebut prompting rantai‑pemikiran (chain‑of‑thought/CoT), ini biasanya dilakukan dengan menambahkan instruksi di awal prompt agar LLM menggambarkan bagaimana ia bisa sampai pada jawaban:

```
Pikirkan langkah demi langkah.

Berapa usia presiden ke‑30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan sekarang mari kita lihat apa yang dihasilkan model yang sama untuk prompt baru ini:

```
1. Pertama, cari tahu siapa presiden ke‑30 Amerika Serikat. Ini bisa dilakukan dengan meneliti daftar presiden AS atau dengan mencari "presiden ke‑30 Amerika Serikat" secara daring. Kamu akan menemukan bahwa presiden ke‑30 adalah Calvin Coolidge.

2. Selanjutnya, cari tahu tahun kapan Calvin Coolidge menjadi presiden. Ini juga bisa ditemukan dengan meneliti daring atau di buku sejarah. Kamu akan menemukan bahwa ia adalah presiden dari 1923‑1929.

3. Sekarang, kamu perlu menentukan tahun kapan ibu mertua Calvin Coolidge meninggal. Ini bisa dilakukan dengan meneliti kehidupan dan kematian istrinya, Grace Goodhue Coolidge. Kamu akan menemukan bahwa ia hidup dari 1879‑1957.

4. Hitung usia ibu mertua Calvin Coolidge ketika ia meninggal. Ia berusia 78 tahun.

5. Tentukan usia Calvin Coolidge ketika ibu mertuanya meninggal. Kurangi tahun kelahirannya, 1872, dari tahun kematian ibu mertuanya, 1957. Ini berarti Calvin Coolidge berusia 85 tahun ketika ibu mertuanya meninggal.
```

Perhatikan bagaimana satu‑satunya perubahan dalam prompt adalah penambahan `Pikirkan langkah demi langkah`, dan betapa besarnya perbedaan yang dibuatnya pada keluaran model. Sekarang model mengeluarkan urutan langkah yang bisa kamu lalui ketika mencoba menjawab pertanyaan ini. Sayangnya, meskipun langkah‑langkahnya sangat masuk akal, keluaran akhir masih salah, karena beberapa fakta salah, yaitu ia mencantumkan tahun kelahiran dan kematian yang salah untuk ibu mertua Presiden Coolidge. Mari kita lihat bagaimana memperbaikinya.

### Pembangkitan‑Diperkaya‑Pengambilan (Retrieval‑Augmented Generation)

_Pembangkitan‑diperkaya‑pengambilan_ (retrieval‑augmented generation/RAG) terdiri dari menemukan potongan‑potongan teks yang relevan, juga dikenal sebagai _konteks_, seperti fakta yang akan kamu temukan di ensiklopedia dan menyertakan konteks itu dalam prompt. Teknik RAG dapat (dan dalam aplikasi nyata seharusnya) digabungkan dengan CoT, tetapi untuk kesederhanaan kita akan menggunakan teknik ini satu per satu di sini. Berikut prompt yang menyertakan RAG:

```
Konteks:

- Calvin Coolidge (lahir John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; 4 Juli 1872 –
5 Januari 1933) adalah pengacara dan politikus Amerika yang menjabat sebagai
presiden ke‑30 Amerika Serikat dari 1923 hingga 1929.

- Grace Anna Coolidge (née Goodhue; 3 Januari 1879 – 8 Juli 1957) adalah
istri presiden ke‑30 Amerika Serikat, Calvin Coolidge.

- Grace Anna Goodhue lahir pada 3 Januari 1879, di Burlington, Vermont,
anak tunggal Andrew Issachar Goodhue dan Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Tanggal lahir: 26 April 1849 ; Tempat lahir:
Burlington, Kabupaten Chittenden, VT, Amerika Serikat ; Kematian: 24 Oktober 1929.

Berapa usia presiden ke‑30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan keluaran dari model:

```
Presiden ke‑30 Amerika Serikat, Calvin Coolidge, berusia 54 tahun ketika ibu mertuanya, Lemira A. Goodhue, meninggal pada 24 Oktober 1929.
```

Sekarang kita jauh lebih dekat dengan jawaban yang benar, tetapi seperti yang kita singgung sebelumnya, LLM tidak terlalu hebat dalam matematika langsung. Dalam kasus ini, hasil akhir 54 tahun meleset 3. Mari kita lihat bagaimana kita bisa memperbaikinya.

### Pemanggilan Alat (Tool Calling)

Teknik _pemanggilan alat_ (tool calling) terdiri dari menambahkan di awal prompt daftar fungsi eksternal yang dapat digunakan LLM, beserta deskripsi tentang apa yang bagus untuk masing‑masing dan instruksi tentang bagaimana memberi sinyal dalam keluaran bahwa ia _ingin_ menggunakan satu (atau lebih) dari fungsi‑fungsi ini. Akhirnya, kamu—pengembang aplikasi—harus mengurai keluaran dan memanggil fungsi yang sesuai. Inilah salah satu cara melakukannya:

```
Alat:

- calculator: Alat ini menerima ekspresi matematika dan mengembalikan hasilnya.

- search: Alat ini menerima kueri mesin pencari dan mengembalikan hasil pencarian pertama.

Jika kamu ingin menggunakan alat untuk sampai pada jawaban, keluarkan daftar alat dan masukan dalam format CSV, dengan baris header `tool,input`.

Berapa usia presiden ke‑30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan ini keluaran yang mungkin kamu dapatkan:

```
tool,input

calculator,2023-1892

search,"Berapa usia Calvin Coolidge ketika ibu mertuanya meninggal?"
```

Meskipun LLM benar mengikuti instruksi format keluaran, alat dan masukan yang dipilih bukan yang paling tepat untuk pertanyaan ini. Ini menyentuh salah satu hal terpenting yang harus diingat saat memprompt LLM: _setiap teknik prompting paling berguna ketika digunakan bersama (beberapa) yang lain_. Misalnya, di sini kita bisa memperbaikinya dengan menggabungkan pemanggilan alat, rantai‑pemikiran, dan RAG ke dalam prompt yang menggunakan ketiganya. Mari kita lihat seperti apa itu:

```
Konteks:

- Calvin Coolidge (lahir John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; 4 Juli 1872 –
5 Januari 1933) adalah pengacara dan politikus Amerika yang menjabat sebagai presiden ke‑30
Amerika Serikat dari 1923 hingga 1929.

- Grace Anna Coolidge (née Goodhue; 3 Januari 1879 – 8 Juli 1957) adalah istri
presiden ke‑30 Amerika Serikat, Calvin Coolidge.

- Grace Anna Goodhue lahir pada 3 Januari 1879, di Burlington, Vermont,
anak tunggal Andrew Issachar Goodhue dan Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Tanggal lahir: 26 April 1849 ; Tempat lahir:
Burlington, Kabupaten Chittenden, VT, Amerika Serikat ; Kematian: 24 Oktober 1929.

Alat:

- calculator: Alat ini menerima ekspresi matematika dan mengembalikan hasilnya.

Jika kamu ingin menggunakan alat untuk sampai pada jawaban, keluarkan daftar alat dan
masukan dalam format CSV, dengan baris header `tool,input`.

Pikirkan langkah demi langkah.

Berapa usia presiden ke‑30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan dengan prompt ini, mungkin setelah beberapa kali mencoba, kita mungkin mendapatkan keluaran ini:

```
tool,input

calculator,1929 - 1872
```

Jika kita mengurai keluaran CSV itu, dan memiliki fungsi kalkulator menjalankan operasi `1929 - 1872`, kita akhirnya mendapatkan jawaban yang benar: 57 tahun.

Seperti contoh sebelumnya, dengan menggabungkan RAG dengan rantai‑pemikiran dan pemanggilan alat, kamu dapat mengambil data yang paling relevan untuk membumi (ground) keluaran model, kemudian membimbingnya langkah demi langkah untuk memastikannya menggunakan konteks itu secara efektif.

### Prompting Beberapa‑Contoh (Few‑Shot)

Akhirnya, kita sampai pada teknik prompting lain yang sangat berguna: _prompting beberapa‑contoh_ (few‑shot prompting). Ini terdiri dari memberikan LLM contoh pertanyaan lain dan jawaban yang benar, yang memungkinkan LLM _belajar_ cara melakukan tugas baru tanpa melalui pelatihan atau penyetelan‑halus tambahan. Dibandingkan dengan penyetelan‑halus, prompting beberapa‑contoh lebih fleksibel—kamu bisa melakukannya langsung pada saat kueri—tetapi kurang kuat, dan kamu mungkin mencapai kinerja lebih baik dengan penyetelan‑halus. Namun demikian, kamu biasanya harus selalu mencoba prompting beberapa‑contoh sebelum penyetelan‑halus:

Prompting beberapa‑contoh statis
: Versi paling dasar dari prompting beberapa‑contoh adalah menyusun daftar tetap sejumlah kecil contoh yang kamu sertakan dalam prompt.

Prompting beberapa‑contoh dinamis
: Jika kamu menyusun kumpulan data banyak contoh, kamu bisa memilih contoh yang paling relevan untuk setiap kueri baru.

Bagian selanjutnya membahas penggunaan LangChain untuk membangun aplikasi menggunakan LLM dan teknik‑teknik prompting ini.

## LangChain dan Mengapa Itu Penting

LangChain adalah salah satu pustaka sumber terbuka paling awal yang menyediakan blok bangunan LLM dan prompting serta peralatan untuk menggabungkannya dengan andal ke dalam aplikasi yang lebih besar. Pada saat penulisan, LangChain telah mengumpulkan lebih dari [28 juta unduhan bulanan](https://oreil.ly/8OKbf), [99.000 bintang GitHub](https://oreil.ly/bF5pc), dan komunitas pengembang terbesar dalam AI generatif ([lebih dari 72.000 orang](https://oreil.ly/PNWL3)). Ini telah memungkinkan insinyur perangkat lunak yang tidak memiliki latar belakang ML untuk memanfaatkan kekuatan LLM guna membangun berbagai aplikasi, mulai dari chatbot AI hingga agen AI yang dapat bernalar dan mengambil tindakan secara bertanggung jawab.

LangChain dibangun di atas ide yang ditekankan di bagian sebelumnya: bahwa teknik prompting paling berguna ketika digunakan bersama. Untuk mempermudahnya, LangChain menyediakan _abstraksi_ sederhana untuk setiap teknik prompting utama. Dengan abstraksi yang kami maksud adalah fungsi dan kelas Python dan JavaScript yang merangkum ide‑ide teknik tersebut menjadi pembungkus yang mudah digunakan. Abstraksi‑abstraksi ini dirancang untuk bekerja dengan baik bersama‑sama dan untuk digabungkan menjadi aplikasi LLM yang lebih besar.

Pertama‑tama, LangChain menyediakan integrasi dengan penyedia LLM utama, baik komersial ([OpenAI](https://oreil.ly/TTLXA), [Anthropic](https://oreil.ly/O4UXw), [Google](https://oreil.ly/12g3Z), dan lainnya) maupun sumber terbuka ([Llama](https://oreil.ly/5WAVi), [Gemma](https://oreil.ly/-40Ne), dan lainnya). Integrasi‑integrasi ini berbagi antarmuka yang sama, membuatnya sangat mudah untuk mencoba LLM baru saat diumumkan dan membantumu menghindari terkunci ke satu penyedia. Kita akan menggunakan ini di Bab 1.

LangChain juga menyediakan abstraksi _template prompt_, yang memungkinkanmu menggunakan kembali prompt lebih dari sekali, memisahkan teks statis dalam prompt dari pengganti yang akan berbeda setiap kali kamu mengirimkannya ke LLM untuk mendapatkan penyelesaian yang dihasilkan. Kita akan membahas ini juga di Bab 1. Prompt LangChain juga dapat disimpan di LangChain Hub untuk dibagikan dengan rekan tim.

LangChain berisi banyak integrasi dengan layanan pihak ketiga (seperti Google Sheets, Wolfram Alpha, Zapier, hanya untuk menyebut beberapa) yang diekspos sebagai _alat_, yang merupakan antarmuka standar untuk fungsi yang digunakan dalam teknik pemanggilan alat.

Untuk RAG, LangChain menyediakan integrasi dengan model _penyematan_ (embedding) utama (model bahasa yang dirancang untuk mengeluarkan representasi numerik, _penyematan_, dari makna sebuah kalimat, paragraf, dan seterusnya), _penyimpan vektor_ (basis data yang dikhususkan untuk menyimpan penyematan), dan _indeks vektor_ (basis data reguler dengan kemampuan penyimpanan vektor). Kamu akan belajar lebih banyak tentang ini di Bab 2 dan 3.

Untuk CoT, LangChain (melalui pustaka LangGraph) menyediakan abstraksi agen yang menggabungkan penalaran rantai‑pemikiran dan pemanggilan alat, pertama kali dipopulerkan oleh [makalah ReAct](https://oreil.ly/27BIC). Ini memungkinkan membangun aplikasi LLM yang melakukan hal berikut:

1. Menalar tentang langkah‑langkah yang harus diambil.
2. Menerjemahkan langkah‑langkah itu menjadi panggilan alat eksternal.
3. Menerima keluaran dari panggilan alat itu.
4. Mengulangi hingga tugas selesai.

Kita membahas ini di Bab 5 hingga 8.

Untuk kasus penggunaan chatbot, menjadi berguna untuk melacak interaksi sebelumnya dan menggunakannya saat menghasilkan respons untuk interaksi mendatang. Ini disebut _memori_, dan Bab 4 membahas penggunaannya dalam LangChain.

Akhirnya, LangChain menyediakan alat untuk menyusun blok‑blok bangunan ini menjadi aplikasi yang kohesif. Bab 1 hingga 6 membahas lebih lanjut tentang ini.

Di luar pustaka ini, LangChain menyediakan [LangSmith](https://oreil.ly/geRgx)—platform untuk membantu men‑debug, menguji, menerapkan, dan memantau alur kerja AI—dan LangGraph Platform—platform untuk menerapkan dan menskalakan agen LangGraph. Kita membahas ini di Bab 9 dan 10.

## Apa yang Diharapkan dari Buku Ini

Dengan buku ini, kami berharap dapat menyampaikan kegembiraan dan kemungkinan menambahkan LLM ke dalam perkakas rekayasa perangkat lunakmu.

Kami terjun ke pemrograman karena kami suka membangun sesuatu, sampai di akhir proyek, melihat produk akhir dan menyadari ada sesuatu yang baru di luar sana, dan kami membangunnya. Pemrograman dengan LLM sangat menarik bagi kami karena memperluas hal‑hal yang bisa kami bangun, membuat hal‑hal yang sebelumnya sulit menjadi mudah (misalnya, mengekstrak angka relevan dari teks panjang) dan hal‑hal yang sebelumnya tidak mungkin menjadi mungkin—coba membangun asisten otomatis setahun yang lalu dan kamu akan berakhir dengan _neraka pohon telepon_ yang kita semua kenal dan cintai dari menelepon nomor dukungan pelanggan.

Sekarang dengan LLM dan LangChain, kamu benar‑benar bisa membangun asisten yang menyenangkan (atau banyak aplikasi lain) yang mengobrol denganmu dan memahami niatmu sampai tingkat yang sangat masuk akal. Perbedaannya seperti siang dan malam! Jika itu terdengar mengasyikkan bagimu (seperti bagi kami) maka kamu sudah berada di tempat yang tepat.

Dalam Kata Pengantar ini, kami telah memberimu penyegaran tentang apa yang membuat LLM bekerja dan mengapa itu memberimu kekuatan super "pembangun‑hal". Memiliki model ML yang sangat besar ini yang memahami bahasa dan dapat mengeluarkan jawaban tertulis dalam bahasa Inggris percakapan (atau bahasa lain) memberimu alat pembangkit bahasa yang _dapat diprogram_ (melalui rekayasa prompt) dan serbaguna. Di akhir buku, kami harap kamu akan melihat betapa kuatnya itu.

Kita akan mulai dengan chatbot AI yang disesuaikan oleh, sebagian besar, instruksi bahasa Inggris sederhana. Itu sendiri seharusnya membuka mata: kamu sekarang dapat "memprogram" sebagian perilaku aplikasimu tanpa kode.

Kemudian muncul kemampuan berikutnya: memberi chatbot‑mu akses ke dokumen‑dokumenmu sendiri, yang mengubahnya dari asisten generik menjadi asisten yang berpengetahuan tentang bidang pengetahuan manusia mana pun yang bisa kamu temukan perpustakaan teks tertulisnya. Ini akan memungkinkanmu membuat chatbot menjawab pertanyaan atau merangkum dokumen yang kamu tulis, misalnya.

Setelah itu, kita akan membuat chatbot mengingat percakapan sebelumnya. Ini akan meningkatkannya dalam dua cara: Akan terasa lebih alami untuk mengobrol dengan chatbot yang mengingat apa yang sebelumnya kamu bicarakan, dan seiring waktu chatbot dapat dipersonalisasi sesuai preferensi masing‑masing penggunanya.

Selanjutnya, kita akan menggunakan teknik rantai‑pemikiran dan pemanggilan alat untuk memberi chatbot kemampuan merencanakan dan bertindak berdasarkan rencana itu, secara berulang. Ini akan memungkinkannya bekerja menuju permintaan yang lebih rumit, seperti menulis laporan penelitian tentang subjek pilihanmu.

Saat kamu menggunakan chatbot‑mu untuk tugas yang lebih rumit, kamu akan merasa perlu memberinya alat untuk berkolaborasi denganmu. Ini mencakup memberi kamu kemampuan untuk menginterupsi atau mengizinkan tindakan sebelum dilakukan, serta memberi chatbot kemampuan untuk meminta informasi atau klarifikasi lebih lanjut sebelum bertindak.

Akhirnya, kami akan menunjukkan cara menerapkan chatbot‑mu ke produksi dan membahas apa yang perlu kamu pertimbangkan sebelum dan setelah mengambil langkah itu, termasuk latensi, keandalan, dan keamanan. Lalu kami akan menunjukkan cara memantau chatbot‑mu di produksi dan terus meningkatkannya saat digunakan.

Sepanjang jalan, kami akan mengajarkanmu seluk‑beluk setiap teknik ini, sehingga ketika kamu menyelesaikan buku, kamu akan benar‑benar menambahkan alat baru (atau dua) ke perkakas rekayasa perangkat lunakmu.

## Konvensi yang Digunakan dalam Buku Ini

Konvensi tipografi berikut digunakan dalam buku ini:

_Miring_
: Menunjukkan istilah baru, URL, alamat email, nama file, dan ekstensi file.

`Lebar tetap`
: Digunakan untuk daftar program, serta dalam paragraf untuk merujuk elemen program seperti nama variabel atau fungsi, basis data, tipe data, variabel lingkungan, pernyataan, dan kata kunci.

> Tips
> Elemen ini menandakan tip atau saran.

> Catatan
> Elemen ini menandakan catatan umum.

## Penggunaan Contoh Kode

Materi tambahan (contoh kode, latihan, dll.) tersedia untuk diunduh di [https://oreil.ly/supp‑LearningLangChain](https://oreil.ly/supp‑LearningLangChain).

Jika kamu memiliki pertanyaan teknis atau masalah menggunakan contoh kode, silakan kirim email ke [support@oreilly.com](mailto:support@oreilly.com).

Buku ini ada untuk membantu menyelesaikan pekerjaanmu. Secara umum, jika contoh kode disertakan dengan buku ini, kamu boleh menggunakannya dalam program dan dokumentasimu. Kamu tidak perlu menghubungi kami untuk izin kecuali kamu mereproduksi sebagian besar kode. Misalnya, menulis program yang menggunakan beberapa potongan kode dari buku ini tidak memerlukan izin. Menjual atau mendistribusikan contoh dari buku O'Reilly memerlukan izin. Menjawab pertanyaan dengan mengutip buku ini dan mengutip contoh kode tidak memerlukan izin. Menggabungkan sejumlah besar contoh kode dari buku ini ke dalam dokumentasi produkmu memerlukan izin.

Kami menghargai, tetapi umumnya tidak memerlukan, atribusi. Atribusi biasanya mencakup judul, penulis, penerbit, dan ISBN. Contoh: "Learning LangChain oleh Mayo Oshin dan Nuno Campos (O'Reilly). Hak cipta 2025 Olumayowa 'Mayo' Olufemi Oshin, 978‑1‑098‑16728‑8."

Jika kamu merasa penggunaan contoh kode‑mu berada di luar penggunaan wajar atau izin yang diberikan di atas, jangan ragu untuk menghubungi kami di [permissions@oreilly.com](mailto:permissions@oreilly.com).

## Pembelajaran Daring O'Reilly

> Catatan
> Selama lebih dari 40 tahun, [O'Reilly Media](https://oreilly.com) telah menyediakan pelatihan teknologi dan bisnis, pengetahuan, dan wawasan untuk membantu perusahaan sukses.

Jaringan unik pakar dan inovator kami berbagi pengetahuan dan keahlian mereka melalui buku, artikel, dan platform pembelajaran daring kami. Platform pembelajaran daring O'Reilly memberi kamu akses sesuai permintaan ke kursus pelatihan langsung, jalur pembelajaran mendalam, lingkungan pengodean interaktif, dan kumpulan besar teks dan video dari O'Reilly dan 200+ penerbit lainnya. Untuk informasi lebih lanjut, kunjungi [https://oreilly.com](https://oreilly.com).

## Cara Menghubungi Kami

Silakan alamatkan komentar dan pertanyaan tentang buku ini ke penerbit:

- O'Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800‑889‑8969 (di Amerika Serikat atau Kanada)
- 707‑827‑7019 (internasional atau lokal)
- 707‑829‑0104 (faks)
- [support@oreilly.com](mailto:support@oreilly.com)
- [https://oreilly.com/about/contact.html](https://oreilly.com/about/contact.html)

Kami memiliki halaman web untuk buku ini, di mana kami mencantumkan erratum, contoh, dan informasi tambahan apa pun. Kamu dapat mengakses halaman ini di [https://oreil.ly/learning‑langchain](https://oreil.ly/learning‑langchain).

Untuk berita dan informasi tentang buku dan kursus kami, kunjungi [https://oreilly.com](https://oreilly.com).

Temukan kami di LinkedIn: [https://linkedin.com/company/oreilly‑media](https://linkedin.com/company/oreilly‑media).

Tonton kami di YouTube: [https://youtube.com/oreillymedia](https://youtube.com/oreillymedia).

## Ucapan Terima Kasih

Kami ingin mengucapkan terima kasih dan penghargaan kepada para peninjau—Rajat Kant Goel, Douglas Bailley, Tom Taulli, Gourav Bais, dan Jacob Lee—atas umpan balik teknis berharga dalam meningkatkan buku ini.
