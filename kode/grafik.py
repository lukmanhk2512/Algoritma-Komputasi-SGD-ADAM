# Membuat Gambar 1 dan 3-7 laporan dari results/e1..e5.json.  # tujuan skrip
# Pengecualian yang disetujui: matplotlib hanya dipakai di sini, untuk menggambar.  # satu-satunya library di skrip ini
import matplotlib  # library grafik (pengecualian)
matplotlib.use("Agg")  # mode tanpa layar: gambar langsung disimpan ke berkas
import matplotlib.pyplot as plt  # antarmuka plot matplotlib
from matplotlib.patches import Patch  # kotak warna untuk legenda peta titik awal
from optimizer import sk1_kecil, NewtonRaphson, f2, df3  # kode buatan sendiri
from util import FIGS, gabung, muat  # utilitas buatan sendiri (pengganti os dan json)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],  # huruf serif seperti laporan
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,  # ukuran huruf, buang bingkai atas/kanan
                     "axes.grid": True, "grid.color": "#e3e3e3", "legend.frameon": False,  # garis bantu tipis, legenda tanpa bingkai
                     "figure.dpi": 200, "savefig.bbox": "tight", "mathtext.fontset": "dejavuserif"})  # resolusi dan tepi rapat
C = {"SGD": "#2E86AB", "Adam": "#F18F01", "Newton": "#C73E1D", "GD": "#2E86AB", "Gauss": "#7B2CBF"}  # warna tiap algoritma
LBL = {"SGD": "SGD", "Adam": "Adam", "Newton": "Newton-Raphson"}  # label legenda


def koma(x, fmt="{:g}"):  # format angka dengan koma desimal (gaya Indonesia)
    return fmt.format(x).replace(".", ",")  # ganti titik desimal dengan koma


def simpan(fig, nama):  # menyimpan gambar ke folder figs/
    fig.savefig(gabung(FIGS, nama))  # tulis berkas PNG
    plt.close(fig)  # tutup gambar agar memori dibebaskan
    print("ok", nama)  # beri tahu nama berkas


# ------------------------------------------------ Gambar 1: tiga studi kasus  # bagian Gambar 1
fig, axs = plt.subplots(1, 3, figsize=(10, 2.9))  # tiga panel berdampingan
m1 = sk1_kecil()  # data tarif parkir
w1 = NewtonRaphson().run(m1, [0.0, 0.0])["w"]  # solusi Newton dari (0, 0)
xs = [x[1] for x in m1.X]  # lama parkir (kolom ke-2, kolom pertama adalah bias)
axs[0].scatter(xs, m1.y, color="#333", s=40, zorder=3, label="data")  # titik data
axs[0].plot([0.5, 4.5], [w1[0] + w1[1] * 0.5, w1[0] + w1[1] * 4.5], color=C["Newton"], lw=2,  # garis hasil Newton
            label=f"ŷ = {koma(round(w1[0], 2))} + {koma(round(w1[1], 2))}x (Newton, 1 langkah)")  # label persamaan garis
axs[0].set(title="(a) SK1 – tarif parkir", xlabel="lama parkir x (jam)", ylabel="tarif y (ribu Rp)",  # judul dan label sumbu
           xticks=[1, 2, 3, 4], ylim=(0, 11))  # tanda sumbu x dan batas sumbu y
axs[0].legend(fontsize=8.5, loc="upper left")  # legenda panel (a)

xx = [-1.9 + 3.8 * i / 400 for i in range(401)]  # 401 titik x di [-1,9; 1,9]
axs[1].plot(xx, [f2(x) for x in xx], color="#333", lw=2)  # kurva f(x) = x^4 - 2x^2
axs[1].scatter([-1, 1], [-1, -1], color="#2A9D8F", s=45, zorder=3)  # dua titik minimum
axs[1].scatter([0], [0], color=C["Newton"], s=45, zorder=3)  # titik maksimum lokal
axs[1].annotate("maksimum\nlokal x = 0", (0, 0), (0, 0.3), ha="center", fontsize=8.5)  # keterangan maksimum
axs[1].annotate("minimum\nx = −1", (-1, -1), (-1.6, -0.5), ha="center", fontsize=8.5)  # keterangan minimum kiri
axs[1].annotate("minimum\nx = 1", (1, -1), (1.6, -0.5), ha="center", fontsize=8.5)  # keterangan minimum kanan
axs[1].set(title="(b) SK2 – f(x) = x⁴ − 2x²", xlabel="x", ylabel="f(x)", ylim=(-1.6, 2.6))  # judul dan label

xx = [0.6 + 1.6 * i / 300 for i in range(301)]  # (c) SK3: titik x di [0,6; 2,2]
axs[2].plot(xx, [df3(x) for x in xx], color="#333", lw=2, label="F′(x) = x² − 2")  # kurva F'(x)
axs[2].axhline(0, color="#888", lw=1)  # sumbu F'(x) = 0
x = 1.0  # titik awal Newton
for k in range(3):  # tiga langkah Newton
    xn = (x + 2 / x) / 2  # titik potong garis singgung dengan sumbu x
    axs[2].plot([x, xn], [df3(x), 0], color=C["Newton"], lw=1.4, ls="--")  # garis singgung
    axs[2].plot([x, x], [0, df3(x)], color="#aaa", lw=0.8, ls=":")  # garis tegak dari sumbu ke kurva
    axs[2].scatter([x], [df3(x)], color=C["Newton"], s=25, zorder=3)  # titik pada kurva
    x = xn  # lanjut ke titik berikutnya
for xv, lab in [(1.0, "x₀ = 1"), (1.5, "x₁ = 3/2"), (17 / 12, "x₂ = 17/12")]:  # label tiap iterasi
    axs[2].annotate(lab, (xv, 0), (xv, 0.35 if xv != 17 / 12 else -0.55), ha="center", fontsize=8)  # tulis label
axs[2].scatter([2 ** 0.5], [0], color="#2A9D8F", s=45, zorder=4, label="√2 = 1,41421…")  # akar sebenarnya
axs[2].set(title="(c) SK3 – menghitung √2", xlabel="x", ylabel="F′(x)", xlim=(0.8, 1.8), ylim=(-1.2, 1.2))  # judul dan batas
axs[2].legend(fontsize=8, loc="upper left")  # legenda panel (c)
fig.tight_layout()  # rapikan jarak antarpanel
simpan(fig, "gambar1_studi_kasus.png")  # simpan Gambar 1

# ------------------------------------------------ Gambar 3: E1 waktu vs n  # bagian Gambar 3
e1 = muat("e1.json")  # baca hasil E1
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)  # dua panel dengan sumbu y bersama
for ax, sk, judul in [(axs[0], "SK1", "(a) SK1 regresi linear"), (axs[1], "RL", "(b) RL regresi logistik")]:  # dua studi kasus
    ns = e1["n"]  # ukuran data
    ax.plot(ns, [n / 1e6 for n in ns], "k--", lw=1, label="acuan: garis lurus ~ n")  # garis acuan kemiringan 1
    for a, ket in [("SGD", "1 epoch"), ("Adam", "1 epoch"), ("Newton", "1 iterasi")]:  # tiga algoritma
        v = e1[sk][a]  # pasangan (rata-rata, simpangan baku)
        ax.errorbar(ns, [x[0] for x in v], yerr=[x[1] for x in v], color=C[a], marker="o", ms=5, lw=1.8,  # kurva dengan galat
                    capsize=3, label=f"{LBL[a]} ({ket}), kemiringan {koma(e1[sk + '_slope'][a], '{:.2f}')}")  # label dengan kemiringan
    ax.set(xscale="log", yscale="log", xlabel="ukuran data n", title=judul)  # skala log-log
    ax.legend(fontsize=8, loc="upper left")  # legenda
axs[0].set_ylabel("waktu (detik)")  # label sumbu y panel kiri
fig.tight_layout()  # rapikan
simpan(fig, "gambar3_e1_waktu_n.png")  # simpan Gambar 3

# ------------------------------------------------ Gambar 4: E2 waktu vs d  # bagian Gambar 4
e2 = muat("e2.json")  # baca hasil E2
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))  # dua panel
ds = e2["d"]  # daftar dimensi
sl = e2["slope"]  # kemiringan tiap kurva
for k, gaya in [("SGD", dict(color=C["SGD"], marker="o")), ("Adam", dict(color=C["Adam"], marker="s")),  # gaya SGD dan Adam
                ("Newton", dict(color=C["Newton"], marker="^")),  # gaya Newton
                ("Newton:Hessian", dict(color=C["Gauss"], ls="--")), ("Newton:gradien", dict(color="#555", ls=":"))]:  # komponen Newton
    lab = LBL.get(k, k)  # label legenda
    axs[0].plot(ds, [v[0] for v in e2["res"][k]], lw=1.8, ms=5, label=f"{lab} ({koma(sl[k], '{:.2f}')})", **gaya)  # gambar kurva
axs[0].set(xscale="log", yscale="log", xlabel=f"dimensi d (n = {e2['n']:,})".replace(",", "."),  # skala log dan label
           ylabel="waktu (detik)", title="(a) 1 epoch SGD/Adam vs 1 iterasi Newton")  # label y dan judul
axs[0].set_xscale("log", base=2)  # sumbu x basis 2 karena d berlipat dua
axs[0].legend(fontsize=8, title="kemiringan log-log", title_fontsize=8)  # legenda
g = e2["gauss"]  # hasil eliminasi Gauss
axs[1].plot(g["d"], [v[0] for v in g["res"]], color=C["Gauss"], marker="D", ms=6, lw=1.8,  # kurva waktu Gauss
            label=f"eliminasi Gauss, kemiringan {koma(g['slope'], '{:.2f}')}")  # label dengan kemiringan
ref = g["res"][-1][0]  # waktu pada d terbesar sebagai jangkar garis acuan
axs[1].plot(g["d"], [ref * (d / g["d"][-1]) ** 3 for d in g["d"]], "k--", lw=1, label="acuan ~ d³")  # garis acuan d^3
axs[1].set(yscale="log", xlabel="dimensi d", title="(b) penyelesaian H p = g")  # skala dan judul
axs[1].set_xscale("log", base=2)  # sumbu x basis 2
axs[1].legend(fontsize=8.5)  # legenda
fig.tight_layout()  # rapikan
simpan(fig, "gambar4_e2_waktu_d.png")  # simpan Gambar 4

# ------------------------------------------------ Gambar 5: E3 konvergensi  # bagian Gambar 5
e3 = muat("e3.json")  # baca hasil E3
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))  # dua panel
for ax, sk, judul in [(axs[0], "SK1", "(a) SK1 regresi linear"), (axs[1], "RL", "(b) RL regresi logistik")]:  # dua studi kasus
    for a in ["SGD", "Adam", "Newton"]:  # tiga algoritma
        k = e3[sk]["kurva"][a]  # data kurva
        y = [max(abs(v), 1e-16) for v in k["mean"]]  # galat rata-rata (dijepit agar aman di skala log)
        x = list(range(len(y)))  # nomor epoch/iterasi
        lab = LBL[a] if a == "Newton" else f"{a} (η = {koma(e3[sk]['best_eta'][a])})"  # label legenda
        ax.plot(x, y, color=C[a], marker="o", ms=3.5, lw=1.8, label=lab)  # kurva konvergensi
        if a != "Newton":  # pita simpangan baku hanya untuk SGD/Adam
            lo = [max(m - s, m / 5) for m, s in zip(k["mean"], k["std"])]  # batas bawah dijepit agar aman di skala log
            hi = [m + s for m, s in zip(k["mean"], k["std"])]  # batas atas
            ax.fill_between(x, lo, hi, color=C[a], alpha=.15, lw=0)  # pita transparan
    ax.set(yscale="log", xlabel="epoch (SGD/Adam) atau iterasi (Newton)", title=judul)  # skala log dan label
    ax.legend(fontsize=8.5, loc="upper right")  # legenda
axs[0].set_ylabel("F(w) − F*")  # label sumbu y
fig.tight_layout()  # rapikan
simpan(fig, "gambar5_e3_konvergensi.png")  # simpan Gambar 5

# ------------------------------------------------ Gambar 6: sensitivitas (E3 grid + E4 batch)  # bagian Gambar 6
e4 = muat("e4.json")  # baca hasil E4
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))  # tiga panel
for ax, sk, judul in [(axs[0], "SK1", "(a) η pada SK1, 20 epoch"), (axs[1], "RL", "(b) η pada RL, 20 epoch")]:  # panel (a) dan (b)
    for a in ["SGD", "Adam"]:  # dua algoritma
        pts = [(e, r) for e, r in zip(e3["grid"], e3[sk]["grid"][a]) if r is not None]  # buang eta yang divergen
        m = [r[0] for _, r in pts]  # rata-rata galat
        s = [r[1] for _, r in pts]  # simpangan baku
        ax.errorbar([e for e, _ in pts], m, yerr=[[min(si, 0.8 * mi) for mi, si in zip(m, s)], s],  # galat bawah dijepit
                    color=C[a], marker="o", ms=5, lw=1.8, capsize=3, label=a)  # gaya kurva
    ax.set(xscale="log", yscale="log", xlabel="learning rate η", title=judul)  # skala log-log
    ax.legend(fontsize=8.5)  # legenda
axs[0].set_ylabel("F − F* akhir")  # label sumbu y
for a in ["SGD", "Adam"]:  # panel (c): ukuran batch
    axs[2].plot(e4["bs"], [v[0] for v in e4["batch"][a]["galat"]], color=C[a], marker="o", ms=5, lw=1.8, label=a)  # kurva galat vs b
axs[2].set(yscale="log", xlabel="ukuran batch b", title=f"(c) batch b pada RL, {e4['epochs']} epoch")  # skala dan judul
axs[2].set_xscale("log", base=2)  # sumbu x basis 2
axs[2].legend(fontsize=8.5)  # legenda
fig.tight_layout()  # rapikan
simpan(fig, "gambar6_e4_sensitivitas.png")  # simpan Gambar 6

# ------------------------------------------------ Gambar 7: SK2 dan SK3 (E5)  # bagian Gambar 7
e5 = muat("e5.json")  # baca hasil E5
fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.1, 1, 1]})  # tiga panel, kiri sedikit lebih lebar
for a, lab in [("GD", "GD (η = 0,05)"), ("Adam", "Adam (η = 0,1)"), ("Newton", "Newton-Raphson")]:  # tiga algoritma
    gal = [max(v, 1e-16) for v in e5["lintasan"][a]["galat"]]  # galat dijepit agar aman di skala log
    axs[0].plot(range(len(gal)), gal, color=C[a], marker="o", ms=2.5, lw=1.5, label=lab)  # kurva galat
axs[0].set(yscale="log", xlabel="iterasi k", ylabel="$|x_k - 1|$", title="(a) SK2: galat dari $x_0$ = 2")  # skala dan label
axs[0].legend(fontsize=8, loc="center right")  # legenda
warna = {"-1": "#2A9D8F", "+1": "#8ECAE6", "0": C["Newton"], "lain": "#cccccc", "divergen": "#333333",  # warna kelas SK2
         "+akar2": "#8ECAE6", "-akar2": C["Newton"], "singular": "#cccccc"}  # warna kelas SK3


def peta(ax, data, baris, judul):  # menggambar peta titik akhir sebagai pita berwarna
    x0 = data["x0"]  # daftar titik awal
    lebar = x0[1] - x0[0]  # jarak antartitik awal
    for j, (a, _) in enumerate(baris):  # satu pita per algoritma
        ax.bar(x0, [0.75] * len(x0), width=lebar, bottom=j - 0.375,  # batang setinggi 0,75 per titik awal
               color=[warna[k] for k in data[a]], linewidth=0)  # warna sesuai kelas titik akhir
    ax.set(yticks=range(len(baris)), yticklabels=[l for _, l in baris], xlim=(-2.5, 2.5),  # label tiap pita
           xlabel="titik awal $x_0$", title=judul)  # label sumbu dan judul
    ax.grid(False)  # tanpa garis bantu


peta(axs[1], e5["peta"], [("GD", "GD\nη=0,05"), ("Adam", "Adam\nη=0,05"), ("Newton", "Newton")],  # peta SK2
     "(b) SK2: ke mana algoritma berakhir?")  # judul panel (b)
axs[1].legend(handles=[Patch(color=warna["-1"], label="x = −1"), Patch(color=warna["+1"], label="x = +1"),  # legenda warna SK2
                       Patch(color=warna["0"], label="x = 0 (maksimum)")],  # kelas maksimum
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)  # legenda di bawah panel
peta(axs[2], e5["SK3"]["peta"], [("GD", "GD\nη=0,25"), ("Adam", "Adam\nη=0,05"), ("Newton", "Newton")],  # peta SK3
     "(c) SK3: ke mana algoritma berakhir?")  # judul panel (c)
axs[2].legend(handles=[Patch(color=warna["+akar2"], label="+√2 (minimum)"),  # legenda warna SK3
                       Patch(color=warna["-akar2"], label="−√2 (maksimum)"),  # kelas maksimum
                       Patch(color=warna["divergen"], label="divergen")],  # kelas divergen
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)  # legenda di bawah panel
fig.tight_layout()  # rapikan
simpan(fig, "gambar7_e5_sk2_sk3.png")  # simpan Gambar 7
