# Research Logbook

## August

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