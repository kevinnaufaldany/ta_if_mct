# Research Logbook

## August

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
- Diskusi dengan kepala divisi mengenai penelitian saya dan mendapat kan informasi mengenai data pada citra labolatorium

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

    ```python
    model = tf.keras.Sequential([
            EfficientNetB0(weights='imagenet', # imagenet
                        include_top=False,
                        pooling='avg'), # max
            Dense(1024, activation='relu'),
            Dropout(0.2), 
            Dense(5, activation='softmax')
        ])
        
    model.layers[0].trainable = False

    model.summary()
    ```