# Research Logbook


## Januari
### 2025-12
- Bimbingan MCT terkait code dan hasil sementara code, code berhasil jalan tapi cukup berat, dan di saran kan untuk memecah gambar menjadi 4 bagian dalam 1 image, dan hasilnya sudah ada split.py yang saat ini saya try terpisah, untuk melakukan splitnya, yang dimana split hanya melakukan jikalau di suatu grid atau bagian yang hanya memiliki segmentasi, dan bagian atau grid yang tidak ada tidak akan termasuk.
- Hasilnya hasil segmentasi bertambah menjadi **8492** yang dimana sebelumnya sebanyak **7.664**, cukup banyak bertambah sebanyak **828** objek yang di dapatkan dari hasil potongan2 gambar. yang dimana total tile nya itu sebanyak **1074** dengan masing2 **537** image dan .json
    - ![Benchmark Awal](figure\counr-cass-split.png) 

## Desember
### 2025-5
- Bimbingan LIA mengenai hasil poin revisi

## November
### 2025-25
- Bimbingan LIA mengenai hasil poin revisi

# After Sempro 21/09/2025

## November
### 2025-07
- Bimbingan with LIA konsultasi mengenai akan ada perubahan variasi penelitian TA kevin dan harus konfirmasi kepada pak bos, jiakalau gas, lets go revisi 

### 2025-04
- Diskusi with LIA mengenai insyallah naik, dan alhamdulillah di acc pak bos, dan di minta untuk membuat berbagai pertanyaan yang dimana di coba anda seolah2 menjadi seorng penguji dan apa yang akan anda tanyakan kepada sang peneliti.

### 2025-03
- Bimbingan with MCT membahas mengenai Variasi lama, dan insyallah ganti

## October
### 2025-30
- Bimbingan with LIA membahas revisi bab 1 dan bab2 bab 3 tipis2 

### 2025-23
- Bimbingan with MCT from Mail, konfirmasi BAB1 dan meminta review all, dan hasil review nya sebagai berikut:
- A. Justifikasi Jumlah Dataset yang Digunakan
    1. Pertanyaan untuk Kamu: Siapkan argumen yang kuat untuk mempertahankan mengapa 142 citra ini dianggap cukup untuk menghasilkan model yang robust.
    2. Apakah ada teknik khusus yang Kamu gunakan, seperti data augmentation yang ekstensif, untuk mengatasi keterbatasan ini? Alasan "keterbatasan SDM" perlu diuraikan lebih dalam pada saat sidang nanti

    - Jawaban : 142 citra dari 1053 citra, di karenakan dari data 1053 itu tidak semuanya citra yang mengandung kasiterit, lalu juga ada beberapa yang resolusinya jelek, || saya berani 142 cukup di karenakan saya sudah melakukan segmentasi datanya dari 142 citra itu dan mendapatkan lebih dari **7.664** objek kasiterit nya, yang dimana sudah lebih dari cukup, seperti pada penelitian yang dilakukan oleh B.A.P. Ferreira yang datasetnya **1.740** partikel kuarsa / objek, || augmentasi ga terlalu banyak hanya general saja, flip, rotate, noise simple, tidak merubah2 kontras lebih lanjut ataupun perubahan warna2 yang kompleks untuk augmentasi, karena mineral cassiterite mirip2 dengan mineral lain, agar tidak menghilangkan detail dari mineral cassiterite 

    - ![Benchmark Awal](/figure/count-cass.png) 

- B. Penajaman Kontribusi pada Latar Belakang (Bagian 1.1):
    1. Kamu sudah menyebutkan akan menerapkan amodal instance segmentation untuk mengatasi oklusi (butir mineral yang tumpang tindih). Ini adalah poin kebaruan yang sangat baik. Namun, alur penyampaiannya bisa sedikit dipertajam.
    2. Coba tambahkan satu kalimat transisi sebelum memperkenalkan amodal instance segmentation untuk lebih menyoroti masalah oklusi sebagai tantangan utama.
    3. Contohnya: "Meskipun Mask R-CNN andal, tantangan spesifik pada citra butir mineral adalah adanya tumpang tindih (oklusi) yang sering terjadi. Untuk mengatasi masalah inilah, penelitian ini mengusulkan penerapan teknik amodal instance segmentation..."


### 2025-21
- Bimbingan with LIA revisi tipis BAB1 and fix BAB1, dan BAB2 fokus di 2.1 Tinjauan pustaka

### 2025-15
- Bimbingan with LIA revisi habis2an BAB1 dan lanjut ke BAB2

### 2025-13
- Bimbingan with MCT konfirmasi our code dan sudah di acc, waktunya NULIS

### 2025-6
- Try buat code menggunakan data sedikit yakni 11 dan menggunakan Cross Validation dan berhasil di fold 1 yakni: 
![Benchmark Awal](/figure/fold1_curves.png) 

## September

### 2025-30 
- Menggunakan saran yang di berikan yakni menggunakan SAM, dan ada salah satu source dari github [labelme-with-segment-anything](https://github.com/originlake/labelme-with-segment-anything) untuk anotasi menggunakan Labelme dengan SAM yang dapat secara otomatis menyesuaikan objek

### 2025-15 
- Diskusi mengenai hasil segmentasi sedikit data untuk dummy yakni baru 10 images dan mendapatkan informasi mengenai SAM (Segment Anything Model) yakni model kecerdasan buatan yang dikembangkan Oleh Meta yang dirancang untuk segmentasi objek dalam gambar dan video.
- Menggunakan referensi yang saya fork dari [Wada_Labelme_Image_Polygonal](https://github.com/wkentaro/labelme) untuk anotasi poligonal pada citra, yang memudahkan proses pelabelan data untuk segmentasi gambar.

### 2025-12 
- Diskusi mengenai mineral cassiterite/ bijih timah dan konsultasi isi Bab 1 penulisan with LIA
- Diskusi mengenai Train yang selalu runtime eror, cross validation, Evaluasi model segmentasi, dengan solusi learn & try Runpod dan di invite pada platform Runpod teamnya pak MCT dan sudah learn dengan ketua grub D 

### 2025-1 
- Diskusi mengenai hasil data yang ada dan model yakni Mask R-CNN

## August

### 2025-21 
- Meminta data ke perusahaan kepada kepala lab dan menambahkan data manual menggunakan mikroskop

### 2025-15 
- Membuat surat pemohonan pengantar penelitian untuk legalitas permintaan data citra laboratorium

### 2025-13 
- Melakukan wawancara bersama kepala laboratorium dan langsung mencoba penggunaan mikroskop dan software pada microskop tersebut

### 2025-12 
- Menemui kepala bidang laboratorium untuk diskusi mengenai penelitian saya dan membuat janji untuk besok hari saya dapat berkunnjung dan melihat langsung proses penentuan mineral pada miskroskop saat ini

### 2025-11 
- Diskusi lagi dengan kepala divisi mengenai mustahilnya atas batasan penelitian saya, dikarenakan secara perhitungan jika ingin menentukan suatu kadar timah itu juga harus menghitung mineral lainnya juga, karena saling berhubungan

### 2025-10 
- Diskusi dengan anak magang lagi di karenakan dari kepala divisi menginginkan outputnya itu langsung dari jumlah kadar timah nya dari suatu citra tersebut

### 2025-09 
- Diskusi dengan pembimbing lapangan beserta kepala divisi mengenai rencana output dari penelitian saya mengenai menghitung jumlah mineral cassiterite pada suatu citra 

### 2025-07 
- Diskusi dengan anak magang dari eksplorasi yang memahami alur dan contoh dari data citra pada laboratorium mengenai kira2 akan apa output dari penelitian saya ini

### 2025-03 
- Diskusi dengan kepala divisi mengenai penelitian saya dan mendapat kan informasi mengenai data pada citra mikroskop labolatorium

### 2025-01 
- output dari XRF itu adalah prediksi kadar dari hasil tembakan XRF yang terintegrasi ke layar XRF dan memprediksi semua mineral yang ada dalam bentuk persentase

## July

### 2025-31 
- Diskusi dengan karyawan mengenai alat saat ini yang di gunakan untuk menentukan kadar timah yang hampir sejenis dengan penelitian saya yang bernama XRF 


### 2025-07 
- Research bagaimana si cara memisahkan ukuran bijih dalam satuan (mesh) menggunakan shaking table
- Mempelajari Instance Segmentation untuk memisahkan kandungan sn dengan mineral lain pada bijih timah


## June

### 2025-06 
- Research jurnal-jurnal mengenai EfficientNet-B0 dan cara menentukan kualitas bijih timah berdasarkan visual
- Mencoba program menggunakan EfficientNet-B0 
- Discuss mengenai judul saya ke seorang penambang dan rekan2 di perusahaan kp