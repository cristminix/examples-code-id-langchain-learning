# Bab 11. Membangun dengan LLM

Salah satu pertanyaan terbuka terbesar di dunia LLM saat ini adalah bagaimana cara terbaik menempatkannya di tangan pengguna akhir. Dalam beberapa hal, LLM sebenarnya adalah antarmuka yang lebih intuitif untuk komputasi daripada yang ada sebelumnya. Mereka jauh lebih toleran terhadap salah ketik, kekeliruan ucapan, dan ketidakpresisian umum manusia, dibandingkan dengan aplikasi komputer tradisional. Di sisi lain, kemampuan untuk menangani masukan yang "sedikit melenceng" datang dengan kecenderungan untuk kadang‑kadang menghasilkan hasil yang juga "sedikit melenceng"—yang juga sangat berbeda dari kecenderungan komputasi sebelumnya.

Faktanya, komputer dirancang untuk mengulang set instruksi yang sama dengan hasil yang sama setiap waktu. Selama beberapa dekade terakhir, prinsip keandalan itu meresapi perancangan antarmuka manusia‑komputer (bervariasi disebut HCI, UX, dan UI) hingga sejauh banyak konstruksi biasa akhirnya kurang memadai untuk digunakan dalam aplikasi yang sangat bergantung pada LLM.

Mari kita ambil contoh: Figma adalah aplikasi perangkat lunak yang digunakan oleh perancang untuk membuat renderan setia desain untuk situs web, aplikasi seluler, sampul buku atau majalah—dan seterusnya. Seperti halnya hampir semua perangkat lunak produktivitas (perangkat lunak untuk pembuatan semacam konten panjang), antarmukanya adalah kombinasi dari berikut:

- Palet alat dan _primitif_ (blok bangunan dasar) yang telah dibangun sebelumnya, dalam hal ini garis, bentuk, alat pilihan dan cat, dan banyak lagi
- Kanvas, di mana pengguna menyisipkan blok bangunan ini dan mengaturnya menjadi kreasi mereka: halaman situs web, layar aplikasi seluler, dan sebagainya

Antarmuka ini dibangun atas premis bahwa kemampuan perangkat lunak diketahui sebelumnya, yang memang benar dalam kasus Figma. Semua blok bangunan dan alat dikodekan oleh insinyur perangkat lunak sebelumnya. Oleh karena itu, mereka diketahui ada pada saat antarmuka dirancang. Kedengarannya hampir konyol untuk menunjukkannya, tetapi hal yang sama tidak sepenuhnya benar untuk perangkat lunak yang banyak menggunakan LLM.

Lihat pengolah kata (misalnya, Microsoft Word atau Google Docs). Ini adalah aplikasi perangkat lunak untuk pembuatan konten teks panjang semacam itu, seperti posting blog, artikel, bab buku, dan sejenisnya. Antarmuka yang tersedia di sini juga terdiri dari kombinasi yang familiar:

- Palet alat dan _primitif_ yang telah dibangun sebelumnya: dalam kasus pengolah kata, primitif yang tersedia adalah tabel, daftar, judul, penampung gambar, dan seterusnya, dan alatnya adalah pemeriksa ejaan, komentar, dan sebagainya.
- _Kanvas_: dalam hal ini, secara harfiah adalah halaman kosong, di mana pengguna mengetik kata‑kata dan dapat menyertakan beberapa elemen yang baru saja disebutkan.

Bagaimana situasi ini berubah jika kita membangun pengolah kata asli‑LLM? Bab ini mengeksplorasi tiga kemungkinan jawaban untuk pertanyaan ini, yang secara luas dapat diterapkan ke aplikasi LLM apa pun. Untuk setiap pola yang kita jelajahi, kita akan membahas konsep kunci apa yang kamu perlukan untuk mengimplementasikannya dengan sukses. Kami tidak bermaksud menyiratkan bahwa ini adalah satu‑satunya, akan butuh waktu sampai debu mengendap pada pertanyaan khusus ini.

Mari kita lihat masing‑masing pola ini, mulai dari yang termudah untuk ditambahkan ke aplikasi yang ada.

## Chatbot Interaktif

Ini bisa dibilang peningkatan termudah untuk ditambahkan ke aplikasi perangkat lunak yang ada. Pada konsepsi paling dasarnya, ide ini hanya menambahkan pendamping AI—untuk bertukar pikiran—sementara semua pekerjaan masih terjadi dalam antarmuka pengguna yang ada dari aplikasi. Contoh di sini adalah GitHub Copilot Chat, yang dapat digunakan di bilah sisi di dalam editor kode VSCode.

Peningkatan untuk pola ini adalah menambahkan beberapa titik komunikasi antara ekstensi pendamping AI dan aplikasi utama. Misalnya, di VSCode, asisten dapat "melihat" konten file yang sedang disunting atau bagian kode apa pun yang dipilih pengguna. Dan di arah sebaliknya, asisten dapat menyisipkan atau menyunting teks di editor terbuka itu, mencapai beberapa bentuk kolaborasi dasar antara pengguna dan LLM.

> Catatan
> Obrolan aliran seperti yang kami gambarkan di sini saat ini adalah aplikasi prototipikal LLM. Ini hampir selalu hal pertama yang dipelajari pengembang aplikasi untuk dibangun dalam perjalanan LLM mereka, dan hampir selalu hal pertama yang dicapai perusahaan saat menambahkan LLM ke aplikasi mereka yang ada. Mungkin ini akan tetap terjadi selama bertahun‑tahun mendatang, tetapi hasil lain yang mungkin adalah obrolan aliran menjadi baris perintah era LLM—yaitu, yang paling dekat dengan akses pemrograman langsung, menjadi antarmuka khusus, seperti yang terjadi pada komputer.

Untuk membangun chatbot paling dasar, kamu harus menggunakan komponen‑komponen ini:

- **Model obrolan**
  Penyetelan dialog mereka cocok dengan interaksi multi‑putaran dengan pengguna. Lihat [Prakata](preface01.xhtml#pr01_preface_1736545679069216) untuk lebih lanjut tentang penyetelan dialog.
- **Riwayat percakapan**
  Chatbot yang berguna harus mampu "melewati halo." Artinya, jika chatbot tidak dapat mengingat masukan pengguna sebelumnya, akan jauh lebih sulit untuk memiliki percakapan bermakna dengannya, yang secara implisit merujuk pesan sebelumnya.

Untuk melampaui dasar‑dasar, kamu mungkin menambahkan hal‑hal berikut:

- **Keluaran aliran**
  Pengalaman chatbot terbaik saat ini mengalirkan keluaran LLM token‑demi‑token (atau dalam potongan lebih besar, seperti kalimat atau paragraf) langsung ke pengguna, yang meringankan latensi yang melekat pada LLM saat ini.
- **Pemanggilan alat**
  Untuk memberi chatbot kemampuan berinteraksi dengan kanvas utama dan alat aplikasi, kamu dapat mengeksposnya sebagai alat yang dapat diputuskan model untuk dipanggil—misalnya, alat "ambil teks terpilih" dan alat "sisipkan teks di akhir dokumen".
- **Manusia‑dalam‑putaran**
  Segera setelah kamu memberi chatbot alat yang dapat mengubah apa yang ada di kanvas aplikasi, kamu menciptakan kebutuhan untuk mengembalikan beberapa kendali kepada pengguna—misalnya, membiarkan pengguna mengonfirmasi, atau bahkan menyunting, sebelum teks baru disisipkan.

## Penyuntingan Kolaboratif dengan LLM

Sebagian besar perangkat lunak produktivitas memiliki beberapa bentuk penyuntingan kolaboratif yang dibangun di dalamnya, yang dapat kita klasifikasikan ke dalam salah satu ember ini (atau di antaranya):

- **Simpan dan kirim**
  Ini adalah versi paling dasar, yang hanya mendukung satu pengguna menyunting dokumen pada satu waktu, sebelum "melempar tongkat" ke pengguna lain (misalnya, mengirim file melalui email) dan mengulangi proses sampai selesai. Contoh paling jelas adalah rangkaian aplikasi Microsoft Office: Excel, Word, PowerPoint.
- **Kontrol versi**
  Ini adalah evolusi simpan dan kirim yang mendukung banyak penyunting bekerja secara bersamaan pada milik mereka sendiri (dan tidak menyadari perubahan satu sama lain) dengan menyediakan alat untuk menggabungkan pekerjaan mereka setelahnya: strategi penggabungan (cara menggabungkan perubahan tidak terkait) dan resolusi konflik (cara menggabungkan perubahan tidak kompatibel). Contoh paling populer saat ini adalah Git/GitHub, digunakan oleh insinyur perangkat lunak untuk berkolaborasi pada proyek perangkat lunak.
- **Kolaborasi waktu‑nyata**
  Ini memungkinkan banyak penyunting bekerja pada dokumen yang sama pada waktu yang sama, sambil melihat perubahan satu sama lain. Ini bisa dibilang bentuk kolaborasi yang diaktifkan perangkat lunak paling alami, dibuktikan oleh popularitas Google Docs dan Google Sheets di antara pengguna komputer teknis dan nonteknis.

Pola pengalaman pengguna LLM ini terdiri dari mempekerjakan agen LLM sebagai salah satu "pengguna" yang berkontribusi pada dokumen bersama ini. Ini dapat mengambil banyak bentuk, termasuk berikut:

- "Kopilot" selalu‑menyala yang memberimu saran tentang cara menyelesaikan kalimat berikutnya
- "Penyusun" asinkron, yang kamu tugaskan, misalnya, pergi dan meneliti topik yang dimaksud dan kembali nanti dengan bagian yang dapat kamu gabungkan dalam dokumen akhirmu

Untuk membangun ini, kamu mungkin memerlukan hal‑hal berikut:

- **Status bersama**
  Agen LLM dan pengguna manusia harus berada pada pijakan yang sama dalam hal akses dan pemahaman status dokumen—yaitu, mereka akan dapat mengurai status dokumen dan menghasilkan suntingan terhadap status itu dalam format yang kompatibel.
- **Pengelola tugas**
  Menghasilkan suntingan yang berguna untuk dokumen akan selalu menjadi proses multi‑langkah, yang dapat memakan waktu dan gagal di tengah jalan. Ini menciptakan kebutuhan penjadwalan dan orkestrasi pekerjaan berjalan panjang yang andal, dengan antrean, pemulihan kesalahan, dan kendali atas tugas yang berjalan.
- **Menggabungkan percabangan**
  Pengguna akan terus menyunting dokumen setelah menugaskan agen LLM, jadi keluaran LLM perlu digabungkan dengan pekerjaan pengguna, baik secara manual oleh pengguna (pengalaman seperti Git) atau otomatis (melalui algoritma resolusi konflik seperti CRDT dan transformasi operasional [OT], digunakan oleh aplikasi seperti Google Docs).
- **Konkurensi**
  Fakta bahwa pengguna manusia dan agen LLM sedang mengerjakan hal yang sama pada waktu yang sama memerlukan kemampuan untuk menangani interupsi, pembatalan, pengalihan (lakukan ini sebagai gantinya), dan antrean (lakukan ini juga).
- **Tumpukan batalkan/ulangi**
  Ini adalah pola yang ada di mana‑mana dalam perangkat lunak produktivitas, yang tak terhindarkan juga diperlukan di sini. Pengguna berubah pikiran dan ingin kembali ke status dokumen sebelumnya, dan aplikasi LLM perlu mampu mengikuti mereka ke sana.
- **Keluaran antara**
  Menggabungkan keluaran pengguna dan LLM menjadi jauh lebih mudah ketika keluaran‑keluaran itu bertahap dan tiba sedikit demi sedikit segera setelah diproduksi, dengan cara yang sama seperti seseorang menulis halaman 10‑paragraf satu kalimat pada satu waktu.

## Komputasi Ambient

Pola UX yang sangat berguna adalah perangkat lunak latar belakang selalu‑menyala yang muncul ketika sesuatu yang "menarik" telah terjadi yang pantas mendapat perhatianmu. Kamu dapat menemukan ini di banyak tempat saat ini. Beberapa contoh adalah:

- Kamu dapat mengatur peringatan di aplikasi pialangmu untuk memberi tahu kamu ketika beberapa saham turun di bawah harga tertentu.
- Kamu dapat meminta Google untuk memberi tahu kamu ketika hasil pencarian baru ditemukan yang cocok dengan beberapa kueri pencarian.
- Kamu dapat mendefinisikan peringatan untuk infrastruktur komputermu untuk memberi tahu kamu ketika sesuatu berada di luar pola perilaku biasa.

Hambatan utama untuk mendeploy pola ini lebih luas mungkin adalah membuat definisi _menarik_ yang andal sebelumnya yang merupakan keduanya berikut:

- **Berguna**
  Ini akan memberi tahu kamu ketika kamu pikir seharusnya.
- **Praktis**
  Kebanyakan pengguna tidak ingin menghabiskan banyak waktu sebelumnya membuat aturan peringatan tak terbatas.

Kemampuan penalaran LLM dapat membuka aplikasi baru dari pola _komputasi ambient_ ini yang secara bersamaan lebih berguna (mereka mengidentifikasi lebih banyak dari apa yang akan kamu anggap menarik) dan lebih sedikit pekerjaan untuk disiapkan (penalaran mereka dapat menggantikan banyak atau semua penyiapan manual aturan).

Perbedaan besar antara _kolaboratif_ dan _ambient_ adalah konkurensi:

- **Kolaboratif**
  Kamu dan LLM biasanya (atau kadang‑kadang) melakukan pekerjaan pada waktu yang sama dan saling memanfaatkan pekerjaan satu sama lain.
- **Ambient**
  LLM terus‑menerus melakukan semacam pekerjaan di latar belakang sementara kamu, pengguna, mungkin melakukan hal lain sama sekali.

Untuk membangun ini, kamu perlu:

- **Pemicu**
  Agen LLM perlu menerima (atau melakukan pengecekan berkala untuk) informasi baru dari lingkungan. Ini sebenarnya yang memotivasi komputasi ambient: sumber informasi baru periodik atau terus‑menerus yang sudah ada sebelumnya yang perlu diproses.
- **Memori jangka panjang**
  Tidak mungkin mendeteksi peristiwa menarik baru tanpa berkonsultasi dengan basis data informasi yang diterima sebelumnya.
- **Refleksi (atau pembelajaran)**
  Memahami apa yang _menarik_ (apa yang pantas mendapat masukan manusia) kemungkinan memerlukan pembelajaran dari setiap peristiwa menarik sebelumnya setelah itu terjadi. Ini biasanya disebut _langkah refleksi_, di mana LLM menghasilkan pembaruan untuk memori jangka panjangnya, mungkin memodifikasi "aturan" internalnya untuk mendeteksi peristiwa menarik di masa depan.
- **Ringkas keluaran**
  Agen yang bekerja di latar belakang kemungkinan akan menghasilkan lebih banyak keluaran daripada yang ingin dilihat pengguna manusia. Ini memerlukan agar arsitektur agen dimodifikasi untuk menghasilkan ringkasan pekerjaan yang dilakukan dan menampilkan kepada pengguna hanya apa yang baru atau patut diperhatikan.
- **Pengelola tugas**
  Memiliki agen LLM yang bekerja terus‑menerus di latar belakang memerlukan penggunaan beberapa sistem untuk mengelola pekerjaan, mengantre proses baru, dan menangani serta memulihkan dari kesalahan.

## Ringkasan

LLM memiliki potensi untuk mengubah tidak hanya [cara kita membangun perangkat lunak](https://oreil.ly/RqnCm), tetapi juga perangkat lunak yang kita bangun itu sendiri. Kemampuan baru yang kita para pengembang miliki untuk menghasilkan konten baru tidak hanya akan meningkatkan banyak aplikasi yang ada, tetapi dapat membuat hal‑hal baru menjadi mungkin yang belum kita impikan.

Tidak ada jalan pintas di sini. Kamu benar‑benar perlu membangun sesuatu yang (s)buruk, berbicara dengan pengguna, dan bilas dan ulangi sampai sesuatu yang baru dan tak terduga keluar di sisi lain.

Dengan bab terakhir ini, dan buku secara keseluruhan, kami telah mencoba memberimu pengetahuan yang kami pikir dapat membantu kamu membangun sesuatu yang unik dan baik dengan LLM. Kami ingin mengucapkan terima kasih karena telah ikut dalam perjalanan ini bersama kami dan mengucapkan semoga sukses dalam karier dan masa depanmu.
