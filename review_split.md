# Code After Split The Result
- `datareader.py` akan langsung melakukan **split grid 2 * 2** seperti pada contoh 
<!-- - ![Benchmark Awal](figure/tiled_paper.png)  -->
<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure/tiled_paper.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">figure/tiled_paper.png</td>
  </tr>
</table>

- atau sesuai pada yang ada di folder `figure\ex_tiled` sudah ada image original yakni `figure\ex_tiled\285-25100F_Ori.jpg` berserta .json ori nya yakni `figure\ex_tiled\285-25100F_Ori.json` \
&
- contoh tiles image dan .json nya yakni `figure\ex_tiled\285-25100F_tile0.png` & `figure\ex_tiled\285-25100F_tile0.json`

## Special Case Split
- Jadi setelah di baca datanya, maka nanti akan di lakukan di lakukan **validasi** bahwa **setiap bagian harus ada segmentasi nya**, jika **tidak** maka **bagian tersebut atau tiled tersebut tidak akan di anggap** atau **tidak di gunakan**, seperti pada folder `figure\tiled_special_case` 
- sebagai contoh dari image dan .json originalnya yakni `figure\tiled_special_case\PMP73-1748J_Ori.jpg` & `figure\tiled_special_case\PMP73-1748J_Ori.json` \
&
- menjadi hanya 1 tile yakni `figure\tiled_special_case\PMP73-1748J_tile0.png` .json nya yakni `figure\tiled_special_case\PMP73-1748J_tile0.png`

## Contoh Visualisasi
### 1. Hasil Validate Model saat Train

<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_val\prediction_sample_2.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_val\prediction_sample_2.png</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_val\prediction_sample_4.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_val\prediction_sample_4.png</td>
  </tr>
</table>

### 2. Hasil Visualisasi Ketika Evaluasi / Testing
- Contoh hasil Visualisasi Evaluasi per Tiled
<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_eval\tile_1.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_eval\tile_1.png</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_eval\tile_1_error_map.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_eval\tile_1.png</td>
  </tr>
</table>

- Contoh hasil Visualisasi Evaluasi Merged
<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_eval\merged_result.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_eval\merged_result.png</td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center" width="100%">
      <img src="figure\hasil_eval\merged_result_error_map.png" alt="Gambar" width="1000"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="100%">hasil_eval\merged_result_error_map.png</td>
  </tr>
</table>