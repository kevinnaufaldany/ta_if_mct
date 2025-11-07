# Review Perkembangan Penelitian Mask R-CNN Kevin
- Variasi TA lama kevin

![Variasi Lama](/figure/VariasiLama.png)
## Eksperimen Arsitektur & Kendala pada Backbone
Sebelumnya, saya mencoba beberapa `variasi arsitektur`, termasuk menambahkan komponen `ASN (Amodal Segmentation Network)` serta menggabungkan varian **FPN (Feature Pyramid Network)** dengan `Multipath FPN`. Namun, hasil dari kombinasi tersebut kurang memuaskan. MFPN yang saya coba sepertinya tidak berjalan optimal Hasilnya tidak memuaskan karena metrik evaluasi masih di angka `AP@0.5:0.95 = 0.000` stagnan, dan `integrasi ASN` juga **tidak memberikan pengaruh signifikan** seperti yang diharapkan.

Saya juga sempat mengalami kendala pada variasi backbone yang digunakan yakni `ResNet101` karena harus ada yang di setting in lagi, di karenakan `pretrainnya` itu menggunakan `ResNet50`, nah jadi **modelnya ga kebaca** dan  Hasilnya belum memuaskan karena metrik evaluasi masih di angka `AP@0.5:0.95 = 0.000`, sehingga performanya tidak sesuai ekspektasi.

## Pre-training
Belakangan, saya menyadari bahwa **pre-training di PyTorch** memiliki dua versi, yaitu versi `v1 (versi awal dari paper original)` dan `v2 (versi yang sudah ada sedikit improvement)`. Fokus saya sekarang adalah mencoba kedua versi pre-trained ini untuk melihat apakah ada perbedaan kinerja. walaupun sudah tau mana yang bakal menang. dengan variasi modelnya juga seperti `Amodal` dan `Standar`, beserta variasi **optimizer** `SGD` dan `adam` karena ada bebrapa paper yang pake `SGD` dan `Adam` walau kebanyakan pake SGD.

## Variasi yang Diuji New
Untuk saat ini, ada tiga variasi utama yang saya uji:
1. **Pre-trained v1 vs v2**: Saya membandingkan hasil dari Mask R-CNN yang di-pre-train dengan COCO v1, dari `maskrcnn_resnet50_fpn` yakni	
***Mask R-CNN model with a ResNet-50-FPN backbone from the Mask R-CNN paper.*** dan yang di-pre-train dengan versi v2 yang lebih baru dari  `maskrcnn_resnet50_fpn_v2` yakni ***Improved Mask R-CNN model with a ResNet-50-FPN backbone from the Benchmarking Detection Transfer Learning with Vision Transformers paper.***
2. **Model Amodal vs Standar**: Saya membandingkan Model yang diuji apakah mampu menangani segmentasi `amodal` (memahami bagian objek yang terhalang/occlusion) atau hanya segmentasi `standar`. untuk melihat hasil evaluasi dari mesin maupun yang di validasi oleh anotator dari data yang ada
3. **Optimizer**: Saya juga mencoba perbandingan optimizer antara `SGD`(dengan momentum) dan `Adam`, untuk melihat mana yang lebih stabil dan cocok untuk skenario ini. yang di ambil bedasarkan paper, walau terjadi kerincuan default lr dari setiap optimizer, cohtoh pada file `kombinasi142.md` sgd dan adam berbeda `--lr` nya
- Variasi TA BARU kevin insyallah di acc
![Variasi Baru](/figure/VariasiBaru.png)
## Code
ada 4 file .py utama yakni `code\datareaders.py`, `code\model.py`, `code\train.py` dan `code\utils.py`, dan untuk variasi kombinasi nya ada di file `code\kombinasi142.md` sama saja sebenernya dengan yang `11` hanya beda dataset saja yang 11 itu pake dataset 11 dan 142 pake dataset 142. untuk size menggunakan 1920:1080, dan kemarin sudah kevin try running dan makan kurang lebih `2-3 jam` pake `dataset11` yang dimana hanya **11 gambar**, itu variasi `1. maskrcnn_resnet50_fpn + Amodal + SGD` dan `2. maskrcnn_resnet50_fpn + Standar + Adam`, yang dimana pake **GPU A40 dengan ke 2 variasi itu di jalankan atau di train berbarengan mamakan waktu 2-3 jam** dengan hasil seperti pada file `figure\prediction_sample_1.png`, untuk grafiknya bisa di cek di wandb link berikut : 
- [Wandb Cassiterite TA](https://wandb.ai/kevinnaufaldany-institut-teknologi-sumatera-/cassiterite-segmentation-ta?nw=nwuserkevinnaufaldany) 

maff pak kalo wandb nya masih kurang oke agak ga rapih numpuk semua fold nya hehe, first time, dan Menurut bapak bagusnya apakah kevin turun size nya jadi lebil kecil contoh `1280 x 720`? yang merupakan salah satu saran dari bu LIA juga, karena sempet kevin iseng2 running di leptop kevin pake `224 x 224` seperti pada file `figure\prediction_sample_224.png`, jadi ga memungkinkan kalo dia ga lanscape dan kecil, karena bentuknyua segmentasi 

## Penutup
Jadi, intinya saat ini saya fokus pada `tiga variasi tersebut` dan mengesampingkan dulu ~~kombinasi backbone / fpn~~ yang lain atau ~~penambahan ASN~~ yang tidak terlalu berpengaruh. kalau bapak ada saran bisa banget pak di sampaikan, ini final kevin, paling nanti tinggal nyeting2 `def postprocess_amodal(self, detections):` agar lebih over power lagi, karena itu masih dikit settingannya, sama visualisasi2nya pak, seperti hasil prediction sampe setiap fold nya dan untuk evaluasi nya juga, dan untuk nama variasi nya apa ya pak kalau kayak gini untuk `Tabel 3.3 Variasi ...`.


