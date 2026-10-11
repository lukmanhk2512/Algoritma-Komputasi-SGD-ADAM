# Kode Eksperimen: SGD vs Adam vs Newton-Raphson

Kode inti tidak memakai library apa pun, termasuk modul standar Python seperti `json`, `os`,
`math`, `random`, dan `fractions`. Vektor ditulis sebagai `list`, eksponen memakai operator `**`,
logaritma natural ditulis sendiri, bilangan acak dibangkitkan dengan LCG 64-bit sehingga hasilnya
dapat diulang persis, pecahan eksak memakai kelas `Pecahan` buatan sendiri, dan berkas hasil
ditulis/dibaca dengan penulis dan pembaca JSON buatan sendiri (`util.py`). Pengecualiannya hanya
dua: modul `time` untuk mengukur waktu, serta `matplotlib` dan Graphviz untuk menggambar grafik.
Setiap baris kode diberi komentar.

## Tiga studi kasus kecil (semua bisa dicek dengan tangan)

| Studi kasus | Masalah | Hasil yang mudah dicek |
|---|---|---|
| SK1 | Tarif parkir: lama parkir x = 1, 2, 3, 4 jam, tarif y = 3, 5, 7, 9 ribu Rp | Jawaban tepat y = 1 + 2x; Newton sampai di (1; 2) dalam satu langkah |
| SK2 | Fungsi dua lembah f(x) = x⁴ − 2x² | Newton x ← 2x³/(3x² − 1): dari 2 ke 16/11 → 1; dari 0,3 ke maksimum x = 0 |
| SK3 | Menghitung √2: minimumkan F(x) = x³/3 − 2x | Newton x ← (x + 2/x)/2: 1 → 3/2 → 17/12 → 577/408; dari x₀ < 0 ke maksimum −√2 |

Untuk eksperimen skala besar dipakai dua model berdata banyak (d = 3):
**SK1 diperbesar** (xⱼ ~ U(0, 5), y = w_trueᵀx + derau normal) dan **RL**, regresi logistik
sintetis (xⱼ ~ U(−2, 2), label dari model logistik).

## Isi folder

```
kode_eksperimen/
├── kode/
│   ├── optimizer.py        modul inti: LCG, ln, SGD, Adam, NewtonRaphson, eliminasi_gauss, SK1–SK3, model RL
│   ├── uji_manual.py       uji kebenaran: 21 nilai hitungan tangan (Tabel 1–4) vs keluaran program, kelas Pecahan
│   ├── util.py             lokasi folder, rata-rata, simpangan baku, kemiringan log-log, JSON buatan sendiri
│   ├── eksperimen_e1.py    E1: waktu terhadap ukuran data n          → results/e1.json
│   ├── eksperimen_e2.py    E2: waktu terhadap dimensi d              → results/e2.json
│   ├── eksperimen_e3.py    E3: grid η dan kecepatan konvergensi      → results/e3.json
│   ├── eksperimen_e4.py    E4: ukuran batch, redaman α, titik awal   → results/e4.json
│   ├── eksperimen_e5.py    E5: SK2 dan SK3, peta titik awal, laju kuadratik → results/e5.json
│   ├── grafik.py           Gambar 1 dan 3–7
│   └── gambar_tambahan.py  Gambar 2 dan 8 (diagram, butuh Graphviz)
├── results/                hasil eksperimen (sudah tersedia)
└── figs/                   semua gambar laporan
```

Folder `results/` dan `figs/` dicari relatif terhadap lokasi file kode, jadi skrip bisa dijalankan
dari folder mana pun. Karena tidak memakai modul `os`, kedua folder tidak dibuat otomatis; keduanya
sudah tersedia di arsip ini, jadi jangan dihapus.

## Cara menjalankan

```bash
cd kode_eksperimen/kode

# 1. Uji kebenaran (kurang dari 1 detik)
python uji_manual.py            # diakhiri "21 dari 21 nilai hitungan tangan cocok."

# 2. Eksperimen (opsional, hasilnya sudah ada di results/)
python eksperimen_e1.py         # ±3 menit, karena n sampai 10^6
python eksperimen_e2.py         # ±1 menit
python eksperimen_e3.py         # ±5 menit
python eksperimen_e4.py         # ±6 menit, membaca η terbaik dari results/e3.json
python eksperimen_e5.py         # beberapa detik

# 3. Gambar
python grafik.py                # Gambar 1, 3–7
python gambar_tambahan.py       # Gambar 2 dan 8
```

Satu-satunya pustaka luar yang dibutuhkan adalah `matplotlib` (untuk grafik) dan program
Graphviz `dot` (untuk diagram). Menjalankan ulang eksperimen akan menimpa file di `results/`.
Hasil loss, galat, dan jumlah iterasi selalu sama karena seed-nya tetap, sedangkan hasil
pengukuran waktu bergantung pada komputer yang dipakai.

## Ringkasan eksperimen (setiap percobaan diulang 5 kali, seed 0–4)

| Kode | Tujuan | Pengaturan |
|---|---|---|
| E1 | Waktu terhadap n | SK1 diperbesar dan RL, d = 3, n = 10²–10⁶; 1 epoch (b = 32) vs 1 iterasi Newton |
| E2 | Waktu terhadap d | RL, n = 1.000, d = 2–64; eliminasi Gauss saja d = 16–256 |
| E3 | Kecepatan konvergensi | SK1 diperbesar dan RL, n = 10⁴, 20 epoch, η terbaik dari grid; Newton sampai ‖g‖ < 10⁻¹⁰ |
| E4 | Sensitivitas parameter | η (grid E3), b ∈ {1, 8, 32, 128, 512}, redaman α ∈ {0,25; 0,5; 1}, titik awal w₀ = (0, c, 0) |
| E5 | Perilaku pada SK2 dan SK3 | 501 titik awal x₀ ∈ [−2,5; 2,5], grid η, laju kuadratik Newton |

## Gambar

| File | Isi |
|---|---|
| `gambar1_studi_kasus.png` | Tiga studi kasus kecil beserta solusi Newton |
| `gambar2_alur_iterasi.png` | Alur satu iterasi ketiga algoritma |
| `gambar3_e1_waktu_n.png` | Waktu terhadap ukuran data n |
| `gambar4_e2_waktu_d.png` | Waktu terhadap dimensi d dan waktu eliminasi Gauss |
| `gambar5_e3_konvergensi.png` | Galat F(w) − F* terhadap epoch/iterasi |
| `gambar6_e4_sensitivitas.png` | Sensitivitas η dan ukuran batch |
| `gambar7_e5_sk2_sk3.png` | SK2: galat dari x₀ = 2; peta titik akhir SK2 dan SK3 |
| `gambar8_panduan.png` | Panduan praktis memilih algoritma |

## Referensi

- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.
- Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.
