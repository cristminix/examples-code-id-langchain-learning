# Kata Pengantar

Pada tanggal 30 November 2022, perusahaan yang berbasis di San Francisco, OpenAI, [merilis ChatGPT secara publik](https://oreil.ly/uAnsr)—robot obrolan AI yang viral dan bisa bikin konten, jawab pertanyaan, dan selesaikan masalah seperti manusia. Dalam dua bulan peluncurannya, ChatGPT menarik lebih dari [100 juta pengguna aktif bulanan](https://oreil.ly/ATsLe), tingkat adopsi tercepat untuk aplikasi teknologi konsumen baru (sampai sekarang). ChatGPT adalah pengalaman robot obrolan yang ditenagai oleh versi yang disesuaikan untuk instruksi dan dialog dari keluarga model bahasa besar (LLM) GPT-3.5 milik OpenAI. Kami akan segera membahas definisi konsep-konsep ini.

> Catatan
> Membangun aplikasi LLM dengan atau tanpa LangChain memerlukan penggunaan LLM. Dalam buku ini kami akan menggunakan [API OpenAI](https://oreil.ly/-YYoR) sebagai penyedia LLM yang kami gunakan dalam contoh kode (harga tercantum di platformnya). Salah satu manfaat bekerja dengan LangChain adalah Anda bisa mengikuti semua contoh ini menggunakan OpenAI atau penyedia LLM komersial atau sumber terbuka lainnya.

Tiga bulan kemudian, OpenAI [merilis API ChatGPT](https://oreil.ly/DwU7R), memberikan akses kepada pengembang ke kemampuan obrolan dan ucapan ke teks. Ini memulai jumlah tak terhitung aplikasi baru dan pengembangan teknis di bawah istilah payung longgar _AI buatan_.

Sebelum kita mendefinisikan AI buatan dan LLM, mari kita sentuh konsep _pembelajaran mesin_ (ML). Beberapa _algoritma_ komputer (bayangkan resep yang bisa diulang untuk mencapai tugas yang sudah ditentukan, seperti mengocok setumpuk kartu) ditulis langsung oleh insinyur perangkat lunak. Algoritma komputer lainnya justru _dipelajari_ dari jumlah contoh latihan yang sangat besar—pekerjaan insinyur perangkat lunak berganti dari menulis algoritma itu sendiri menjadi menulis logika latihan yang menciptakan algoritma. Banyak perhatian di bidang ML ditujukan untuk mengembangkan algoritma untuk memprediksi berbagai hal, dari cuaca besok sampai rute pengiriman paling efisien untuk pengemudi Amazon.

Dengan kemunculan LLM dan model buatan lainnya (seperti model difusi untuk membuat gambar, yang tidak kami bahas dalam buku ini), teknik ML yang sama sekarang diterapkan pada masalah membuat konten baru, seperti paragraf teks atau gambar baru, yang pada saat yang sama unik dan didasarkan pada contoh dalam data latihan. LLM khususnya adalah model buatan yang dikhususkan untuk membuat teks.

LLM memiliki dua perbedaan lain dari algoritma ML sebelumnya:

- Mereka dilatih dengan jumlah data yang jauh lebih besar; melatih salah satu model ini dari awal akan sangat mahal.
- Mereka lebih serbaguna.

Model pembuatan teks yang sama bisa digunakan untuk ringkasan, terjemahan, klasifikasi, dan lain-lain, sedangkan model ML sebelumnya biasanya dilatih dan digunakan untuk tugas tertentu.

Dua perbedaan ini bekerja sama untuk membuat pekerjaan insinyur perangkat lunak berganti lagi, dengan jumlah waktu yang semakin banyak dikhususkan untuk mencari tahu cara membuat LLM bekerja untuk kasus penggunaan mereka. Dan itulah yang LangChain semua tentang.

Pada akhir 2023, LLM pesaing muncul, termasuk Claude dari Anthropic dan Bard dari Google (kemudian diubah nama menjadi Gemini), memberikan akses yang lebih luas lagi ke kemampuan baru ini. Dan selanjutnya, ribuan startup sukses dan perusahaan besar telah memasukkan API AI buatan untuk membangun aplikasi untuk berbagai kasus penggunaan, mulai dari robot obrolan dukungan pelanggan sampai menulis dan mencari kesalahan kode.

Pada tanggal 22 Oktober 2022, Harrison Chase [menerbitkan komit pertama](https://oreil.ly/mCdYZ) di GitHub untuk perpustakaan sumber terbuka LangChain. LangChain dimulai dari kesadaran bahwa aplikasi LLM paling menarik perlu menggunakan LLM bersama dengan ["sumber komputasi atau pengetahuan lainnya"](https://oreil.ly/uXiPi). Misalnya, Anda bisa coba membuat LLM menghasilkan jawaban untuk pertanyaan ini:

```
Berapa banyak bola yang tersisa setelah membagi 1.234 bola secara merata di antara 123 orang?
```

Anda mungkin akan kecewa dengan kemampuan matematikanya. Namun, jika Anda memasangkannya dengan fungsi kalkulator, Anda bisa malah menginstruksikan LLM untuk mengubah kata-kata pertanyaan menjadi input yang bisa ditangani kalkulator:

```
1.234 % 123
```

Kemudian Anda bisa memberikannya ke fungsi kalkulator dan dapatkan jawaban yang akurat untuk pertanyaan asli Anda. LangChain adalah perpustakaan pertama (dan, saat penulisan, terbesar) yang menyediakan blok bangunan seperti itu dan alat untuk menggabungkannya secara andal ke dalam aplikasi yang lebih besar. Sebelum membahas apa yang diperlukan untuk membangun aplikasi yang menarik dengan alat baru ini, mari kita kenali lebih baik LLM dan LangChain.

## Pengantar Singkat tentang LLM

Dalam istilah awam, LLM adalah algoritma yang dilatih yang menerima input teks dan memprediksi serta menghasilkan output teks yang mirip manusia. Pada dasarnya, mereka berperilaku seperti fitur pelengkapan otomatis yang ditemukan di banyak ponsel pintar, tetapi dibawa ke tingkat yang ekstrem.

Mari kita uraikan istilah _model bahasa besar_:

- _Besar_ mengacu pada ukuran model-model ini dalam hal data latihan dan parameter yang digunakan selama proses pembelajaran. Misalnya, model GPT-3 milik OpenAI berisi 175 miliar _parameter_, yang dipelajari dari latihan pada 45 terabita data teks.^[Tom B. Brown dkk., ["Model Bahasa Adalah Pembelajar Sedikit Contoh"](https://oreil.ly/1qoM6), arXiv, 22 Juli 2020.] _Parameter_ dalam model jaringan saraf terdiri dari angka-angka yang mengontrol output setiap _neuron_ dan bobot relatif koneksi dengan neuron tetangganya. (Neuron mana yang terhubung dengan neuron lainnya bervariasi untuk setiap arsitektur jaringan saraf dan di luar cakupan buku ini.)
- _Model bahasa_ mengacu pada algoritma komputer yang dilatih untuk menerima teks tertulis (dalam bahasa Inggris atau bahasa lain) dan menghasilkan output juga sebagai teks tertulis (dalam bahasa yang sama atau berbeda). Ini adalah _jaringan saraf_, jenis model ML yang menyerupai konsep bergaya dari otak manusia, dengan output akhir resulting dari kombinasi output individual dari banyak fungsi matematika sederhana, yang disebut _neuron_, dan interkoneksi mereka. Jika banyak neuron ini diatur dengan cara tertentu, dengan proses latihan yang tepat dan data latihan yang tepat, ini menghasilkan model yang mampu menafsirkan makna kata-kata dan kalimat individual, yang membuatnya mungkin untuk menggunakannya untuk menghasilkan teks tertulis yang masuk akal dan bisa dibaca.

Karena prevalensi bahasa Inggris dalam data latihan, sebagian besar model lebih baik dalam bahasa Inggris daripada bahasa lain dengan jumlah penutur yang lebih sedikit. Dengan "lebih baik" kami maksud lebih mudah membuat mereka menghasilkan output yang diinginkan dalam bahasa Inggris. Ada LLM yang dirancang untuk output multibahasa, seperti [BLOOM](https://oreil.ly/Nq7w0), yang menggunakan proporsi data latihan yang lebih besar dalam bahasa lain. Menariknya, perbedaan kinerja antar bahasa tidak sebesar yang mungkin diharapkan, bahkan dalam LLM yang dilatih pada korpus latihan yang didominasi bahasa Inggris. Peneliti telah menemukan bahwa LLM mampu mentransfer beberapa pemahaman semantik mereka ke bahasa lain.^[Xiang Zhang dkk., ["Jangan Percaya ChatGPT Ketika Pertanyaan Anda Tidak dalam Bahasa Inggris: Studi tentang Kemampuan Multibahasa dan Jenis LLM"](https://oreil.ly/u5Cy1), Prosiding Konferensi 2023 tentang Metode Empiris dalam Pemrosesan Bahasa Alami, 6–10 Desember 2023.]

Secara keseluruhan, _model bahasa besar_ adalah contoh model bahasa umum yang besar yang dilatih pada jumlah teks yang sangat besar. Dengan kata lain, model-model ini telah belajar dari pola dalam kumpulan data teks yang besar—buku, artikel, forum, dan sumber publik lainnya—untuk melakukan tugas terkait teks umum. Tugas-tugas ini termasuk pembuatan teks, ringkasan, terjemahan, klasifikasi, dan lainnya.

Misalkan kita menginstruksikan LLM untuk melengkapi kalimat berikut:

```
Ibukota Inggris adalah _______.
```

LLM akan mengambil input teks itu dan memprediksi jawaban output yang benar sebagai `London`. Ini terlihat seperti sihir, tetapi tidak. Di balik layar, LLM memperkirakan probabilitas urutan kata(-kata) yang diberikan urutan kata sebelumnya.

> Tip
> Secara teknis, model membuat prediksi berdasarkan token, bukan kata. _Token_ mewakili unit teks atomik. Token bisa mewakili karakter individual, kata, subkata, atau unit linguistik yang lebih besar, tergantung pada pendekatan tokenisasi spesifik yang digunakan. Misalnya, menggunakan tokenisasi GPT-3.5 (disebut `cl100k`), frasa _selamat pagi sahabat terkasih_ akan terdiri dari [lima token](https://oreil.ly/dU83b) (menggunakan `_` untuk menunjukkan karakter spasi):

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

> Biasanya tokenisator dilatih dengan tujuan memiliki kata-kata paling umum yang dikodekan menjadi satu token, misalnya, kata _morning_ dikodekan sebagai token `6693`. Kata-kata yang kurang umum, atau kata dalam bahasa lain (biasanya tokenisator dilatih pada teks bahasa Inggris), memerlukan beberapa token untuk mengodekannya. Misalnya, kata _dearest_ dikodekan sebagai token `409, 15795`. Satu token mencakup rata-rata empat karakter teks untuk teks bahasa Inggris umum, atau kira-kira tiga perempat kata.

Mesin pendorong di balik kekuatan prediktif LLM dikenal sebagai _arsitektur jaringan saraf transformator_.^[Untuk informasi lebih lanjut, lihat Ashish Vaswani dkk., ["Perhatian Adalah Semua yang Anda Butuhkan"](https://oreil.ly/Frtul), arXiv, 12 Juni 2017.] Arsitektur transformator memungkinkan model untuk menangani urutan data, seperti kalimat atau baris kode, dan membuat prediksi tentang kata(-kata) yang paling mungkin berikutnya dalam urutan. Transformator dirancang untuk memahami konteks setiap kata dalam kalimat dengan mempertimbangkannya dalam hubungan dengan setiap kata lainnya. Ini memungkinkan model untuk membangun pemahaman komprehensif tentang makna kalimat, paragraf, dan lain-lain (dengan kata lain, urutan kata) sebagai makna gabungan bagiannya dalam hubungan satu sama lain.

Jadi, ketika model melihat urutan kata* ibukota Inggris adalah*, model membuat prediksi berdasarkan contoh serupa yang dilihatnya selama latihan. Dalam korpus latihan model, kata _Inggris_ (atau token yang mewakilinya) akan sering muncul dalam kalimat di tempat serupa dengan kata-kata seperti _Prancis_, _Amerika Serikat_, _Cina_. Kata _ibukota_ akan muncul dalam data latihan dalam banyak kalimat yang juga mengandung kata-kata seperti _Inggris_, _Prancis_, dan _AS_, dan kata-kata seperti _London_, _Paris_, _Washington_. Pengulangan ini selama latihan model menghasilkan kemampuan untuk memprediksi dengan benar bahwa kata berikutnya dalam urutan seharusnya _London_.

Instruksi dan teks input yang Anda berikan kepada model disebut _prompt_. Prompting bisa memiliki dampak signifikan pada kualitas output dari LLM. Ada beberapa praktik terbaik untuk _desain prompt_ atau _rekayasa prompt_, termasuk memberikan instruksi yang jelas dan ringkas dengan contoh kontekstual, yang kami bahas nanti dalam buku ini. Sebelum kita masuk lebih jauh ke prompting, mari kita lihat beberapa jenis LLM berbeda yang tersedia untuk Anda gunakan.

Jenis dasar, yang menjadi dasar semua yang lain, umumnya dikenal sebagai _LLM pralatih_: model ini telah dilatih pada jumlah teks yang sangat besar (ditemukan di internet dan dalam buku, surat kabar, kode, transkrip video, dan lain-lain) dengan cara terawasi sendiri. Ini berarti bahwa—tidak seperti dalam ML terawasi, di mana sebelum latihan peneliti perlu mengumpulkan kumpulan data pasangan*input* ke _output yang diharapkan_—untuk LLM pasangan-pasangan itu disimpulkan dari data latihan. Faktanya, satu-satunya cara yang layak untuk menggunakan kumpulan data yang sangat besar adalah mengumpulkan pasangan-pasangan itu dari data latihan secara otomatis. Dua teknik untuk melakukan ini melibatkan membuat model melakukan hal berikut:

Prediksi kata berikutnya
: Hapus kata terakhir dari setiap kalimat dalam data latihan, dan itu menghasilkan pasangan _input_ dan _output yang diharapkan_, seperti \_Ibukota Inggris adalah \_\_\__ dan \_London_.

Prediksi kata yang hilang
: Demikian pula, jika Anda mengambil setiap kalimat dan menghilangkan kata dari tengah, Anda sekarang memiliki pasangan input dan output yang diharapkan lainnya, seperti _\_\_\_ Inggris adalah London_ dan _ibukota_.

Model-model ini cukup sulit digunakan apa adanya, mereka memerlukan Anda untuk memulai respons dengan awalan yang sesuai. Misalnya, jika Anda ingin tahu ibukota Inggris, Anda mungkin mendapat respons dengan memprompt model dengan _Ibukota Inggris adalah_, tetapi tidak dengan _Berapa ibukota Inggris?_ yang lebih alami.

### LLM yang Disesuaikan Instruksi

[Peneliti](https://oreil.ly/lP6hr) telah membuat LLM pralatih lebih mudah digunakan dengan latihan lebih lanjut (latihan tambahan yang diterapkan di atas latihan panjang dan mahal yang dijelaskan di bagian sebelumnya), juga dikenal sebagai _fine-tuning_ mereka pada hal berikut:

Kumpulan data tugas-spesifik
: Ini adalah kumpulan data pasangan pertanyaan/jawaban yang dirakit manual oleh peneliti, memberikan contoh respons yang diinginkan untuk pertanyaan umum yang mungkin diprompt pengguna akhir ke model. Misalnya, kumpulan data mungkin berisi pasangan berikut: *T: Berapa ibukota Inggris? J: Ibukota Inggris adalah London. *Tidak seperti kumpulan data pralatihan, ini dirakit manual, jadi secara necessity jauh lebih kecil:

Pembelajaran penguatan dari umpan balik manusia (RLHF)
: Melalui penggunaan [metode RLHF](https://oreil.ly/lrlAK), kumpulan data yang dirakit manual itu diperkuat dengan umpan balik pengguna yang diterima pada output yang dihasilkan model. Misalnya, pengguna A lebih memilih _Ibukota Inggris adalah London_ daripada _London adalah ibukota Inggris_ sebagai jawaban untuk pertanyaan sebelumnya.

Penyesuaian instruksi telah menjadi kunci untuk memperluas jumlah orang yang bisa membangun aplikasi dengan LLM, karena mereka sekarang bisa diprompt dengan _instruksi_, sering dalam bentuk pertanyaan seperti, _Berapa ibukota Inggris?_, sebagai lawan dari _Ibukota Inggris adalah_.

### LLM yang Disesuaikan Dialog

Model yang disesuaikan untuk tujuan dialog atau obrolan adalah [peningkatan lebih lanjut](https://oreil.ly/1DxW6) dari LLM yang disesuaikan instruksi. Penyedia LLM yang berbeda menggunakan teknik yang berbeda, jadi ini tidak selalu benar untuk semua _model obrolan_, tetapi biasanya ini dilakukan melalui hal berikut:

Kumpulan data dialog
: Kumpulan data _fine-tuning_ yang dirakit manual diperluas untuk menyertakan lebih banyak contoh interaksi dialog multi-giliran, yaitu urutan pasangan prompt-respons.

Format obrolan
: Format input dan output model diberikan lapisan struktur di atas teks bebas, yang membagi teks menjadi bagian-bagian yang terkait dengan peran (dan opsional metadata lain seperti nama). Biasanya, peran yang tersedia adalah _sistem_ (untuk instruksi dan kerangka tugas), _pengguna_ (tugas atau pertanyaan aktual), dan _asisten_ (untuk output model). Metode ini berevolusi dari [teknik rekayasa prompt awal](https://oreil.ly/dINx0) dan membuatnya lebih mudah untuk menyesuaikan output model sambil membuatnya lebih sulit bagi model untuk membingitkan input pengguna dengan instruksi. Membingitkan input pengguna dengan instruksi sebelumnya juga dikenal sebagai _jailbreaking_, yang bisa, misalnya, mengarah ke prompt yang dibuat dengan hati-hati, mungkin termasuk rahasia dagang, terbuka untuk pengguna akhir.

### LLM yang Disesuaikan Halus

LLM yang disesuaikan halus dibuat dengan mengambil LLM dasar dan melatihnya lebih lanjut pada kumpulan data milik untuk tugas tertentu. Secara teknis, LLM yang disesuaikan instruksi dan dialog adalah LLM yang disesuaikan halus, tetapi istilah "LLM yang disesuaikan halus" biasanya digunakan untuk menggambarkan LLM yang disesuaikan oleh pengembang untuk tugas spesifik mereka. Misalnya, model bisa disesuaikan halus untuk mengekstrak dengan akurat sentimen, faktor risiko, dan angka keuangan kunci dari laporan tahunan perusahaan publik. Biasanya, model yang disesuaikan halus memiliki kinerja yang lebih baik pada tugas yang dipilih dengan mengorbankan kehilangan keumuman. Artinya, mereka menjadi kurang mampu menjawab pertanyaan pada tugas yang tidak terkait.

Sepanjang sisa buku ini, ketika kami menggunakan istilah* LLM*, kami maksud LLM yang disesuaikan instruksi, dan untuk _model obrolan_ kami maksud LLM yang diinstruksikan dialog, seperti didefinisikan sebelumnya dalam bagian ini. Ini seharusnya menjadi kuda kerja Anda saat menggunakan LLM—alat pertama yang Anda ambil saat memulai aplikasi LLM baru.

Sekarang mari kita bahas singkat beberapa teknik prompting LLM umum sebelum masuk ke LangChain.

## Pengantar Singkat tentang Prompting

Seperti yang kami sentuh sebelumnya, tugas utama insinyur perangkat lunak yang bekerja dengan LLM bukan untuk melatih LLM, atau bahkan menyesuaikan halus satu (biasanya), tetapi untuk mengambil LLM yang ada dan mencari tahu cara membuatnya melakukan tugas yang Anda butuhkan untuk aplikasi Anda. Ada penyedia komersial LLM, seperti OpenAI, Anthropic, dan Google, serta LLM sumber terbuka ([Llama](https://oreil.ly/ld3Fu), [Gemma](https://oreil.ly/RGKfi), dan lainnya), yang dirilis gratis untuk orang lain bangun di atasnya. Menyesuaikan LLM yang ada untuk tugas Anda disebut _rekayasa prompt_.

Banyak teknik prompting telah dikembangkan dalam dua tahun terakhir, dan dalam arti luas, ini adalah buku tentang cara melakukan rekayasa prompt dengan LangChain—cara menggunakan LangChain untuk membuat LLM melakukan yang Anda pikirkan. Tapi sebelum kita masuk ke LangChain yang sebenarnya, membantu untuk membahas beberapa teknik ini terlebih dahulu (dan kami mohon maaf sebelumnya jika [teknik prompting](https://oreil.ly/8uGK_) favorit Anda tidak tercantum di sini; ada terlalu banyak untuk dibahas).

Untuk mengikuti bagian ini kami sarankan menyalin prompt-prompt ini ke Playground OpenAI untuk mencobanya sendiri:

1. Buat akun untuk API OpenAI di [http://platform.openai.com](http://platform.openai.com), yang akan memungkinkan Anda menggunakan LLM OpenAI secara terprogram, yaitu, menggunakan API dari kode Python atau JavaScript Anda. Ini juga akan memberi Anda akses ke Playground OpenAI, di mana Anda bisa bereksperimen dengan prompt dari browser web Anda.
2. Jika perlu, tambahkan detail pembayaran untuk akun OpenAI baru Anda. OpenAI adalah penyedia komersial LLM dan membebankan biaya untuk setiap kali Anda menggunakan model mereka melalui API OpenAI atau melalui Playground. Anda bisa menemukan harga terbaru di [situs web mereka](https://oreil.ly/MiKRD). Selama dua tahun terakhir, harga untuk menggunakan model OpenAI telah turun secara signifikan saat kemampuan dan optimasi baru diperkenalkan.
3. Kunjungi [Playground OpenAI](https://oreil.ly/rxiAG) dan Anda siap untuk mencoba prompt-prompt berikut sendiri. Kami akan menggunakan API OpenAI sepanjang buku ini.
4. Setelah Anda menavigasi ke Playground, Anda akan melihat panel preset di sisi kanan layar, termasuk model pilihan Anda. Jika Anda melihat lebih ke bawah panel, Anda akan melihat Suhu di bawah judul "Konfigurasi model". Pindahkan toggle Suhu dari tengah ke kiri sampai angka menunjukkan 0.00. Pada dasarnya, suhu mengontrol keacakan output LLM. Semakin rendah suhu, semakin deterministik output model.

Sekarang ke prompt-promptnya!

### Prompting Tanpa Contoh

Teknik prompting pertama dan paling lurus ke depan terdiri dari menginstruksikan LLM untuk melakukan tugas yang diinginkan:

```
Berapa usia presiden ke-30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Ini biasanya yang harus Anda coba dulu, dan biasanya akan berhasil untuk pertanyaan sederhana, terutama ketika jawabannya kemungkinan ada di beberapa data latihan. Jika kami prompt `gpt-3.5-turbo` OpenAI dengan prompt sebelumnya, berikut yang dikembalikan:

```
Presiden ke-30 Amerika Serikat, Calvin Coolidge, berusia 48 tahun ketika
ibu mertuanya meninggal pada tahun 1926.
```

> Catatan
> Anda mungkin mendapat hasil berbeda dari yang kami dapatkan. Ada elemen keacakan dalam cara LLM menghasilkan respons, dan OpenAI mungkin telah memperbarui model pada saat Anda mencobanya.

Meskipun model berhasil mengidentifikasi presiden ke-30 dengan benar, jawabannya tidak sepenuhnya benar. Seringkali Anda harus mengulang pada prompt dan respons untuk mendapatkan sistem yang andal. Bagian berikut memperluas caranya.

### Rantai Pikiran

Iterasi yang sangat berguna adalah menginstruksikan model lebih lanjut untuk [meluangkan waktu untuk _berpikir_](https://oreil.ly/8xWcs). Teknik ini telah ditemukan meningkatkan kinerja pada berbagai tugas. Menariknya, makalah yang lebih baru^[Ryan Liu dkk., ["Hati-hati Langkah demi Langkah: Rantai Pikiran Bisa Mengurangi Kinerja pada Tugas di mana Berpikir Membuat Manusia Lebih Buruk"](https://oreil.ly/UHFp9), arXiv, 8 November 2024.] menyarankan bahwa teknik ini juga bisa mengurangi kinerja LLM pada tugas di mana berpikir mengurangi kinerja manusia. Disebut prompting rantai-pikiran (CoT), ini biasanya dilakukan dengan menambahkan prompt dengan instruksi untuk LLM untuk menggambarkan bagaimana model bisa sampai ke jawaban:

```
Berpikir langkah demi langkah.

Berapa usia presiden ke-30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan sekarang mari kita lihat apa yang dihasilkan model yang sama untuk prompt baru ini:

```
1. Pertama, cari tahu siapa presiden ke-30 Amerika Serikat. Ini bisa
dilakukan dengan meneliti daftar presiden AS atau dengan mencari "presiden ke-30
Amerika Serikat" online. Anda akan menemukan bahwa presiden ke-30 adalah Calvin
Coolidge.

2. Selanjutnya, cari tahu tahun ketika Calvin Coolidge menjadi presiden. Ini juga
bisa ditemukan dengan meneliti online atau dalam buku sejarah. Anda akan menemukan bahwa dia
menjadi presiden dari 1923-1929.

3. Sekarang, Anda perlu menentukan tahun ketika ibu mertua Calvin Coolidge meninggal.
Ini bisa dilakukan dengan meneliti kehidupan dan kematian istrinya, Grace Goodhue
Coolidge. Anda akan menemukan bahwa dia hidup dari 1879-1957.

4. Hitung usia ibu mertua Calvin Coolidge ketika dia meninggal. Dia berusia
78 tahun.

5. Tentukan usia Calvin Coolidge ketika ibu mertuanya meninggal. Kurangi
tahun kelahirannya, 1872, dari tahun ibu mertuanya meninggal, 1957. Ini
berarti bahwa Calvin Coolidge berusia 85 tahun ketika ibu mertuanya meninggal.
```

Perhatikan bagaimana satu-satunya perubahan dalam prompt adalah penambahan `Berpikir langkah demi langkah`, dan betapa besar perbedaan yang dibuatnya pada output model. Sekarang model menghasilkan urutan langkah yang bisa Anda lalui saat mencoba menjawab pertanyaan ini. Sayangnya, sementara langkah-langkahnya sangat masuk akal, output akhir masih salah, mengingat model mendapat beberapa fakta salah, yaitu model mencantumkan tahun kelahiran dan kematian yang salah untuk ibu mertua Presiden Coolidge. Mari kita lihat cara memperbaikinya.

### Generasi yang Diperkuat Pengambilan

_Generasi yang diperkuat pengambilan_ (RAG) terdiri dari menemukan potongan teks yang relevan, juga dikenal sebagai _konteks_, seperti fakta yang akan Anda temukan di ensiklopedia dan memasukkan konteks itu ke dalam prompt. Teknik RAG bisa (dan seharusnya dalam aplikasi nyata) digabungkan dengan CoT, tetapi untuk kesederhanaan kami akan menggunakan teknik ini satu per satu di sini. Berikut prompt yang mencakup RAG:

```
Konteks:

- Calvin Coolidge (lahir John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; 4 Juli 1872 –
5 Januari 1933) adalah pengacara dan politikus Amerika yang menjabat sebagai
presiden ke-30 Amerika Serikat dari 1923 hingga 1929.

- Grace Anna Coolidge (née Goodhue; 3 Januari 1879 – 8 Juli 1957) adalah
istri presiden ke-30 Amerika Serikat, Calvin Coolidge.

- Grace Anna Goodhue lahir pada 3 Januari 1879, di Burlington, Vermont,
anak tunggal Andrew Issachar Goodhue dan Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Tanggal lahir: 26 April 1849 ; Tempat lahir:
Burlington, Kabupaten Chittenden, VT, Amerika Serikat ; Kematian: 24 Oktober 1929.

Berapa usia presiden ke-30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan output dari model:

```
Presiden ke-30 Amerika Serikat, Calvin Coolidge, berusia 54 tahun ketika
ibu mertuanya, Lemira A. Goodhue, meninggal pada 24 Oktober 1929.
```

Sekarang kita jauh lebih dekat ke jawaban yang benar, tetapi seperti yang kami sentuh sebelumnya, LLM tidak hebat dalam matematika langsung. Dalam kasus ini, hasil akhir 54 tahun salah 3. Mari kita lihat cara memperbaikinya.

### Pemanggilan Alat

Teknik _pemanggilan alat_ terdiri dari menambahkan prompt dengan daftar fungsi eksternal yang bisa digunakan LLM, bersama dengan deskripsi tentang apa saja yang baik untuk setiapnya dan instruksi tentang cara memberi sinyal dalam output bahwa model _ingin_ menggunakan satu (atau lebih) fungsi ini. Terakhir, Anda—pengembang aplikasi—sebaiknya menguraikan output dan memanggil fungsi yang sesuai. Berikut satu cara untuk melakukan ini:

```
Alat:

- kalkulator: Alat ini menerima ekspresi matematika dan mengembalikan hasilnya.

- pencarian: Alat ini menerima kueri mesin pencari dan mengembalikan hasil pencarian
pertama.

Jika Anda ingin menggunakan alat untuk sampai ke jawaban, keluarkan daftar alat dan
input dalam format CSV, dengan baris header `tool,input`.

Berapa usia presiden ke-30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan ini adalah output yang mungkin Anda dapatkan:

```
tool,input

kalkulator,2023-1892

pencarian,"Berapa usia Calvin Coolidge ketika ibu mertuanya meninggal?"
```

Sementara LLM mengikuti instruksi format output dengan benar, alat dan input yang dipilih tidak yang paling sesuai untuk pertanyaan ini. Ini menyentuh salah satu hal terpenting yang perlu diingat saat prompting LLM: _setiap teknik prompting paling berguna saat digunakan dalam kombinasi dengan (beberapa) yang lain_. Misalnya, di sini kami bisa memperbaikinya dengan menggabungkan pemanggilan alat, rantai-pikiran, dan RAG ke dalam prompt yang menggunakan ketiganya. Mari kita lihat seperti apa itu:

```
Konteks:

- Calvin Coolidge (lahir John Calvin Coolidge Jr.; /ˈkuːlɪdʒ/; 4 Juli 1872 –
5 Januari 1933) adalah pengacara dan politikus Amerika yang menjabat sebagai presiden ke-30
Amerika Serikat dari 1923 hingga 1929.

- Grace Anna Coolidge (née Goodhue; 3 Januari 1879 – 8 Juli 1957) adalah istri
presiden ke-30 Amerika Serikat, Calvin Coolidge.

- Grace Anna Goodhue lahir pada 3 Januari 1879, di Burlington, Vermont,
anak tunggal Andrew Issachar Goodhue dan Lemira Barrett Goodhue.

- Lemira A. Goodhue (Barrett) ; Tanggal lahir: 26 April 1849 ; Tempat lahir:
Burlington, Kabupaten Chittenden, VT, Amerika Serikat ; Kematian: 24 Oktober 1929.

Alat:

- kalkulator: Alat ini menerima ekspresi matematika dan mengembalikan hasilnya.

Jika Anda ingin menggunakan alat untuk sampai ke jawaban, keluarkan daftar alat dan
input dalam format CSV, dengan baris header `tool,input`.

Berpikir langkah demi langkah.

Berapa usia presiden ke-30 Amerika Serikat ketika ibu mertuanya meninggal?
```

Dan dengan prompt ini, mungkin setelah beberapa kali coba, kita mungkin mendapat output ini:

```
tool,input

kalkulator,1929 - 1872
```

Jika kita menguraikan output CSV itu, dan memiliki fungsi kalkulator menjalankan operasi `1929 - 1872`, kita akhirnya mendapat jawaban yang benar: 57 tahun.

Sesuai contoh sebelumnya, dengan menggabungkan RAG dengan rantai-pikiran dan pemanggilan alat, Anda bisa mengambil data paling relevan untuk mendasarkan output model Anda, lalu memandunya langkah demi langkah untuk memastikan model menggunakan konteks itu secara efektif.

### Prompting Beberapa Contoh

Terakhir, kita sampai ke teknik prompting lain yang sangat berguna: _prompting beberapa contoh_. Ini terdiri dari memberikan LLM contoh pertanyaan lain dan jawaban yang benar, yang memungkinkan LLM untuk _belajar_ cara melakukan tugas baru tanpa melalui latihan atau penyesuaian halus tambahan. Ketika dibandingkan dengan penyesuaian halus, prompting beberapa contoh lebih fleksibel—Anda bisa melakukannya langsung saat waktu kueri—tetapi kurang kuat, dan Anda mungkin mencapai kinerja lebih baik dengan penyesuaian halus. Itu dikatakan, Anda biasanya harus selalu mencoba prompting beberapa contoh sebelum penyesuaian halus:

Prompting beberapa contoh statis
: Versi paling dasar dari prompting beberapa contoh adalah merakit daftar yang sudah ditentukan dari jumlah kecil contoh yang Anda sertakan dalam prompt.

Prompting beberapa contoh dinamis
: Jika Anda merakit kumpulan data dari banyak contoh, Anda bisa malah memilih contoh paling relevan untuk setiap kueri baru.

Bagian berikut membahas penggunaan LangChain untuk membangun aplikasi menggunakan LLM dan teknik-teknik prompting ini.

## LangChain dan Mengapa Ini Penting

LangChain adalah salah satu perpustakaan sumber terbuka paling awal yang menyediakan blok bangunan LLM dan prompting dan alat untuk menggabungkannya secara andal ke dalam aplikasi yang lebih besar. Saat penulisan, LangChain telah mengumpulkan lebih dari [28 juta unduhan bulanan](https://oreil.ly/8OKbf), [99.000 bintang GitHub](https://oreil.ly/bF5pc), dan komunitas pengembang terbesar dalam AI buatan ([lebih dari 72.000](https://oreil.ly/PNWL3)). Ini telah memungkinkan insinyur perangkat lunak yang tidak memiliki latar belakang ML untuk memanfaatkan kekuatan LLM untuk membangun berbagai aplikasi, mulai dari robot obrolan AI hingga agen AI yang bisa bernalar dan mengambil tindakan secara bertanggung jawab.

LangChain dibangun di atas ide yang ditekankan di bagian sebelumnya: bahwa teknik prompting paling berguna saat digunakan bersama. Untuk membuatnya lebih mudah, LangChain menyediakan _abstraksi_ sederhana untuk setiap teknik prompting utama. Dengan abstraksi kami maksud fungsi dan kelas Python dan JavaScript yang membungkus ide-ide teknik itu ke dalam pembungkus yang mudah digunakan. Abstraksi ini dirancang untuk bekerja dengan baik bersama dan untuk digabungkan ke dalam aplikasi LLM yang lebih besar.

Pertama-tama, LangChain menyediakan integrasi dengan penyedia LLM utama, baik komersial ([OpenAI](https://oreil.ly/TTLXA), [Anthropic](https://oreil.ly/O4UXw), [Google](https://oreil.ly/12g3Z), dan lainnya) maupun sumber terbuka ([Llama](https://oreil.ly/5WAVi), [Gemma](https://oreil.ly/-40Ne), dan lainnya). Integrasi ini berbagi antarmuka umum, membuatnya sangat mudah untuk mencoba LLM baru saat diumumkan dan memungkinkan Anda menghindari terkunci pada satu penyedia. Kami akan menggunakan ini di Bab 1.

LangChain juga menyediakan abstraksi _template prompt_, yang memungkinkan Anda menggunakan kembali prompt lebih dari sekali, memisahkan teks statis dalam prompt dari placeholder yang akan berbeda untuk setiap kali Anda mengirimkannya ke LLM untuk mendapat penyelesaian yang dihasilkan. Kami akan membahas lebih banyak tentang ini juga di Bab 1. Prompt LangChain juga bisa disimpan di LangChain Hub untuk berbagi dengan rekan tim.

LangChain berisi banyak integrasi dengan layanan pihak ketiga (seperti Google Sheets, Wolfram Alpha, Zapier, hanya untuk menyebutkan beberapa) yang diekspos sebagai _alat_, yang merupakan antarmuka standar untuk fungsi yang digunakan dalam teknik pemanggilan alat.

Untuk RAG, LangChain menyediakan integrasi dengan _model penyematan_ utama (model bahasa yang dirancang untuk menghasilkan representasi numerik, _penyematan_, dari makna kalimat, paragraf, dan lainnya), _penyimpanan vektor_ (basis data yang dikhususkan untuk menyimpan penyematan), dan _indeks vektor_ (basis data reguler dengan kemampuan penyimpanan vektor). Anda akan mempelajari lebih banyak tentang ini di Bab 2 dan 3.

Untuk CoT, LangChain (melalui perpustakaan LangGraph) menyediakan abstraksi agen yang menggabungkan penalaran rantai-pikiran dan pemanggilan alat, yang pertama dipopulerkan oleh [makalah ReAct](https://oreil.ly/27BIC). Ini memungkinkan membangun aplikasi LLM yang melakukan hal berikut:

1. Bernalar tentang langkah-langkah yang akan diambil.
2. Menerjemahkan langkah-langkah itu ke dalam panggilan alat eksternal.
3. Menerima output dari panggilan alat itu.
4. Ulangi sampai tugas selesai.

Kami membahas ini di Bab 5 hingga 8.

Untuk kasus penggunaan robot obrolan, menjadi berguna untuk melacak interaksi sebelumnya dan menggunakannya saat menghasilkan respons untuk interaksi masa depan. Ini disebut _memori_, dan Bab 4 membahas penggunaannya dalam LangChain.

Terakhir, LangChain menyediakan alat untuk mengkomposisikan blok bangunan ini ke dalam aplikasi yang kohesif. Bab 1 hingga 6 membahas lebih banyak tentang ini.

Selain perpustakaan ini, LangChain menyediakan [LangSmith](https://oreil.ly/geRgx)—platform untuk membantu mencari kesalahan, menguji, memasang, dan mengawasi alur kerja AI—dan Platform LangGraph—platform untuk memasang dan menskalakan agen LangGraph. Kami membahas ini di Bab 9 dan 10.

## Yang Diharapkan dari Buku Ini

Dengan buku ini, kami harap bisa menyampaikan kegembiraan dan kemungkinan menambahkan LLM ke peralatan rekayasa perangkat lunak Anda.

Kami masuk ke pemrograman karena kami suka membangun hal-hal, sampai ke akhir proyek, melihat produk akhir dan menyadari ada sesuatu yang baru di luar sana, dan kami membangunnya. Pemrograman dengan LLM sangat menarik bagi kami karena memperluas hal-hal yang bisa kami bangun, membuat hal-hal yang sebelumnya sulit menjadi mudah (misalnya, mengekstrak angka relevan dari teks panjang) dan hal-hal yang sebelumnya tidak mungkin menjadi mungkin—coba bangun asisten otomatis setahun yang lalu dan Anda akan berakhir dengan _neraka telepon pohon_ yang kita semua kenal dan suka dari menelepon nomor dukungan pelanggan.

Sekarang dengan LLM dan LangChain, Anda benar-benar bisa membangun asisten yang menyenangkan (atau berbagai aplikasi lain) yang mengobrol dengan Anda dan memahami niat Anda dengan tingkat yang sangat masuk akal. Perbedaannya sangat jauh! Jika itu terdengar menarik bagi Anda (seperti bagi kami) maka Anda datang ke tempat yang tepat.

Dalam Kata Pengantar ini, kami telah memberikan Anda penyegaran tentang apa yang membuat LLM berjalan dan mengapa persis itu memberi Anda kekuatan super "pembangun hal". Memiliki model ML yang sangat besar yang memahami bahasa dan bisa menghasilkan jawaban yang ditulis dalam bahasa Inggris percakapan (atau bahasa lain) memberi Anda alat pembuatan bahasa yang _dapat diprogram_ (melalui rekayasa prompt) dan serbaguna. Pada akhir buku, kami harap Anda akan melihat betapa kuat itu bisa.

Kami akan mulai dengan robot obrolan AI yang disesuaikan oleh, sebagian besar, instruksi bahasa Inggris biasa. Itu saja seharusnya menjadi pembuka mata: Anda sekarang bisa "memprogram" bagian dari perilaku aplikasi Anda tanpa kode.

Kemudian datang kemampuan berikutnya: memberikan robot obrolan Anda akses ke dokumen Anda sendiri, yang mengubahnya dari asisten generik menjadi yang berpengetahuan tentang area pengetahuan manusia mana pun yang Anda bisa temukan perpustakaan teks tertulis. Ini akan memungkinkan Anda memiliki robot obrolan yang menjawab pertanyaan atau meringkas dokumen yang Anda tulis, misalnya.

Setelah itu, kami akan membuat robot obrolan mengingat percakapan sebelumnya Anda. Ini akan memperbaikinya dengan dua cara: Akan terasa jauh lebih alami memiliki percakapan dengan robot obrolan yang mengingat apa yang telah Anda obrolkan sebelumnya, dan seiring waktu robot obrolan bisa dipersonalisasi ke preferensi setiap penggunanya secara individual.

Selanjutnya, kami akan menggunakan teknik rantai-pikiran dan pemanggilan alat untuk memberikan robot obrolan kemampuan untuk merencanakan dan bertindak pada rencana itu, secara iteratif. Ini akan memungkinkannya untuk bekerja menuju permintaan yang lebih rumit, seperti menulis laporan penelitian tentang subjek pilihan Anda.

Saat Anda menggunakan robot obrolan untuk tugas yang lebih rumit, Anda akan merasa perlu memberinya alat untuk berkolaborasi dengan Anda. Ini mencakup memberi Anda kemampuan untuk mengganggu atau mengotorisasi tindakan sebelum dilakukan, serta menyediakan robot obrolan dengan kemampuan untuk meminta informasi atau klarifikasi lebih lanjut sebelum bertindak.

Terakhir, kami akan menunjukkan cara memasang robot obrolan Anda ke produksi dan membahas apa yang perlu Anda pertimbangkan sebelum dan sesudah mengambil langkah itu, termasuk latensi, keandalan, dan keamanan. Kemudian kami akan menunjukkan cara mengawasi robot obrolan Anda dalam produksi dan terus memperbaikinya saat digunakan.

Sepanjang jalan, kami akan mengajarkan Anda seluk-beluk setiap teknik ini, sehingga saat Anda menyelesaikan buku, Anda akan benar-benar menambahkan alat baru (atau dua) ke peralatan rekayasa perangkat lunak Anda.

## Konvensi yang Digunakan dalam Buku Ini

Konvensi tipografi berikut digunakan dalam buku ini:

_Miring_
: Menunjukkan istilah baru, URL, alamat email, nama file, dan ekstensi file.

`Lebar konstan`
: Digunakan untuk daftar program, serta dalam paragraf untuk merujuk ke elemen program seperti nama variabel atau fungsi, basis data, tipe data, variabel lingkungan, pernyataan, dan kata kunci.

> Tip
> : Elemen ini menandakan tip atau saran.

> Catatan
> : Elemen ini menandakan catatan umum.

## Menggunakan Contoh Kode

Material pelengkap (contoh kode, latihan, dll.) tersedia untuk diunduh di [https://oreil.ly/supp-LearningLangChain](https://oreil.ly/supp-LearningLangChain).

Jika Anda memiliki pertanyaan teknis atau masalah menggunakan contoh kode, kirim email ke [support@oreilly.com](mailto:support@oreilly.com).

Buku ini ada untuk membantu Anda menyelesaikan pekerjaan Anda. Secara umum, jika kode contoh ditawarkan dengan buku ini, Anda boleh menggunakannya dalam program dan dokumentasi Anda. Anda tidak perlu menghubungi kami untuk izin kecuali Anda mereproduksi bagian signifikan dari kode. Misalnya, menulis program yang menggunakan beberapa potongan kode dari buku ini tidak memerlukan izin. Menjual atau mendistribusikan contoh dari buku O'Reilly memerlukan izin. Menjawab pertanyaan dengan mengutip buku ini dan mengutip kode contoh tidak memerlukan izin. Menggabungkan jumlah signifikan kode contoh dari buku ini ke dalam dokumentasi produk Anda memerlukan izin.

Kami menghargai, tetapi umumnya tidak memerlukan, atribusi. Atribusi biasanya mencakup judul, penulis, penerbit, dan ISBN. Misalnya: "Learning LangChain oleh Mayo Oshin dan Nuno Campos (O'Reilly). Hak Cipta 2025 Olumayowa "Mayo" Olufemi Oshin, 978-1-098-16728-8."

Jika Anda merasa penggunaan contoh kode Anda di luar penggunaan wajar atau izin yang diberikan di atas, jangan ragu untuk menghubungi kami di [permissions@oreilly.com](mailto:permissions@oreilly.com).

## Pembelajaran Online O'Reilly

> Catatan
> Selama lebih dari 40 tahun, [O'Reilly Media](https://oreilly.com) telah menyediakan pelatihan teknologi dan bisnis, pengetahuan, dan wawasan untuk membantu perusahaan berhasil.

Jaringan unik ahli dan inovator kami berbagi pengetahuan dan keahlian mereka melalui buku, artikel, dan platform pembelajaran online kami. Platform pembelajaran online O'Reilly memberi Anda akses sesuai permintaan ke kursus pelatihan langsung, jalur pembelajaran mendalam, lingkungan pengkodean interaktif, dan koleksi teks dan video yang luas dari O'Reilly dan 200+ penerbit lainnya. Untuk informasi lebih lanjut, kunjungi [https://oreilly.com](https://oreilly.com).

## Cara Menghubungi Kami

Silakan alamatkan komentar dan pertanyaan tentang buku ini kepada penerbit:

- O'Reilly Media, Inc.
- 1005 Gravenstein Highway North
- Sebastopol, CA 95472
- 800-889-8969 (di Amerika Serikat atau Kanada)
- 707-827-7019 (internasional atau lokal)
- 707-829-0104 (faks)
- [support@oreilly.com](mailto:support@oreilly.com)
- [https://oreilly.com/about/contact.html](https://oreilly.com/about/contact.html)

Kami memiliki halaman web untuk buku ini, di mana kami mencantumkan kesalahan, contoh, dan informasi tambahan apa pun. Anda bisa mengakses halaman ini di [https://oreil.ly/learning-langchain](https://oreil.ly/learning-langchain).

Untuk berita dan informasi tentang buku dan kursus kami, kunjungi [https://oreilly.com](https://oreilly.com).

Temukan kami di LinkedIn: [https://linkedin.com/company/oreilly-media](https://linkedin.com/company/oreilly-media).

Tonton kami di YouTube: [https://youtube.com/oreillymedia](https://youtube.com/oreillymedia).

## Ucapan Terima Kasih

Kami ingin mengucapkan rasa terima kasih dan penghargaan kami kepada para peninjau—Rajat Kant Goel, Douglas Bailley, Tom Taulli, Gourav Bais, dan Jacob Lee—atas umpan balik teknis berharga untuk memperbaiki buku ini.
