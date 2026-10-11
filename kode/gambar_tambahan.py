# Diagram Gambar 2 (alur satu iterasi) dan Gambar 8 (panduan memilih algoritma).  # tujuan skrip
# Pengecualian yang disetujui: membutuhkan program Graphviz `dot` yang dipanggil lewat subprocess.  # alat gambar diagram
import subprocess  # untuk memanggil program dot (bagian dari pengecualian alat gambar)
from util import FIGS, gabung  # lokasi folder figs/ dan penyusun path buatan sendiri

NODE = 'node [shape=box, style="rounded,filled", fontname="Liberation Serif", fontsize=13, color="#555555", penwidth=1.4];'  # gaya kotak
EDGE = 'edge [color="#555555", arrowsize=0.8, fontname="Liberation Serif", fontsize=13, penwidth=1.3];'  # gaya panah

dot2 = "\n".join([  # Gambar 2 dalam bahasa DOT, disusun baris demi baris
    'digraph G {',  # awal graf berarah
    '  rankdir=TB; nodesep=0.4; ranksep=0.45; bgcolor="white";',  # arah atas-ke-bawah, jarak antarsimpul, latar putih
    '  ' + NODE + ' ' + EDGE,  # gaya bawaan kotak dan panah
    '  mulai [label=<Mulai: w<sub>0</sub>, η, kriteria berhenti>, fillcolor="#E8F0F8"];',  # kotak mulai
    '  data  [label="Ambil data\\n(mini-batch b untuk SGD/Adam, seluruh n untuk Newton)", fillcolor="#F3F3F3"];',  # kotak ambil data
    '  grad  [label="Hitung gradien g = ∇F(w)", fillcolor="#F3F3F3"];',  # kotak hitung gradien
    '  subgraph cluster_beda { label="Bagian yang membedakan ketiga algoritma"; labelloc="b";',  # awal kotak putus-putus
    '    fontname="Liberation Serif"; fontsize=13; style="rounded,dashed"; color="#999999";',  # gaya kotak putus-putus
    '    sgd  [label="SGD\\nw ← w − η g", fillcolor="#D6EAF5", color="#2E86AB"];',  # kotak aturan SGD
    '    adam [label=<Adam<br/>m ← β<sub>1</sub>m + (1−β<sub>1</sub>)g<br/>v ← β<sub>2</sub>v + (1−β<sub>2</sub>)g²<br/>w ← w − η m̂ / (√v̂ + ε)>, fillcolor="#FDE6C8", color="#F18F01"];',  # kotak aturan Adam
    '    nr   [label="Newton-Raphson\\nhitung Hessian H = ∇²F(w)\\nselesaikan H p = g (eliminasi Gauss)\\nw ← w − α p", fillcolor="#F6D5CF", color="#C73E1D"];',  # kotak aturan Newton
    '  }',  # akhir kotak putus-putus
    '  cek [label="Konvergen?\\n‖g‖ ≤ tol atau batas iterasi", shape=diamond, style="filled", fillcolor="white", fontsize=12];',  # belah ketupat keputusan
    '  selesai [label="Selesai: w*", fillcolor="#E8F0F8"];',  # kotak selesai
    '  mulai -> data -> grad;',  # panah alur awal
    '  grad -> sgd; grad -> adam; grad -> nr;',  # gradien bercabang ke tiga algoritma
    '  sgd -> cek; adam -> cek; nr -> cek;',  # ketiganya menuju pemeriksaan konvergensi
    '  cek -> selesai [label=" ya"];',  # bila konvergen, selesai
    '  cek -> data [label=" tidak", constraint=false];',  # bila belum, ulangi dari ambil data
    '}',  # akhir graf
])  # gabungkan semua baris dengan baris baru

dot8 = "\n".join([  # Gambar 8 dalam bahasa DOT, disusun baris demi baris
    'digraph G {',  # awal graf berarah
    '  rankdir=TB; nodesep=0.35; ranksep=0.55; bgcolor="white";',  # arah atas-ke-bawah, jarak antarsimpul, latar putih
    '  ' + NODE + ' ' + EDGE,  # gaya bawaan kotak dan panah
    '  q1 [label="Dimensi d kecil\\n(Hessian d×d muat di memori,\\nd³ masih murah)?", shape=diamond, style="filled", fillcolor="white"];',  # pertanyaan 1
    '  q2 [label="Fungsi konveks atau\\ntitik awal sudah dekat\\nsolusi?", shape=diamond, style="filled", fillcolor="white"];',  # pertanyaan 2
    '  q3 [label="Skala fitur timpang /\\nlearning rate sulit dipilih?", shape=diamond, style="filled", fillcolor="white"];',  # pertanyaan 3
    '  nr  [label="Newton-Raphson\\n(konvergen kuadratik)", fillcolor="#F6D5CF", color="#C73E1D"];',  # rekomendasi Newton
    '  nrd [label="Newton teredam (α < 1)\\natau mulai dengan GD\\nlalu beralih ke Newton", fillcolor="#F6D5CF", color="#C73E1D"];',  # rekomendasi Newton teredam
    '  adam [label="Adam\\n(η ≈ 10⁻³ … 10⁻²)", fillcolor="#FDE6C8", color="#F18F01"];',  # rekomendasi Adam
    '  sgd [label="SGD\\n(paling murah, η perlu disetel)", fillcolor="#D6EAF5", color="#2E86AB"];',  # rekomendasi SGD
    '  q1 -> q2 [label=" ya"]; q1 -> q3 [label=" tidak"];',  # cabang pertanyaan 1
    '  q2 -> nr [label=" ya"]; q2 -> nrd [label=" tidak"];',  # cabang pertanyaan 2
    '  q3 -> adam [label=" ya"]; q3 -> sgd [label=" tidak"];',  # cabang pertanyaan 3
    '}',  # akhir graf
])  # gabungkan semua baris dengan baris baru

for dot, nama in [(dot2, "gambar2_alur_iterasi.png"), (dot8, "gambar8_panduan.png")]:  # dua diagram
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", "-o", gabung(FIGS, nama)], input=dot.encode(), check=True)  # render DOT ke PNG
    print("ok", nama)  # beri tahu nama berkas
