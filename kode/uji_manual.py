# Uji kebenaran: nilai hitungan tangan pada Subbab 3.1 (Tabel 1-4)  # tujuan skrip
# dibandingkan dengan keluaran program. Semua harus cocok dalam toleransi 5e-5.  # kriteria lulus
# SK1  tarif parkir     : x = 1, 2, 3, 4 jam, y = 3, 5, 7, 9 ribu Rp (y = 1 + 2x)  # studi kasus 1
# SK2  dua lembah       : f(x) = x^4 - 2x^2  # studi kasus 2
# SK3  menghitung akar 2: F(x) = x^3/3 - 2x, Newton x_baru = (x + 2/x) / 2  # studi kasus 3
# Tidak memakai library apa pun: modul fractions diganti kelas Pecahan buatan sendiri.  # batasan
from optimizer import (SGD, Adam, NewtonRaphson, eliminasi_gauss, sk1_kecil,  # kode buatan sendiri
                       df2, d2f2, df3, d2f3)  # turunan SK2 dan SK3

TOL = 5e-5  # toleransi perbedaan program vs hitungan tangan
hasil = []  # daftar status lulus/gagal setiap pengecekan


def fpb(a, b):  # faktor persekutuan terbesar dengan algoritma Euclid
    a, b = abs(a), abs(b)  # pakai nilai mutlak
    while b != 0:  # ulangi sampai sisa bagi nol
        a, b = b, a % b  # (a, b) -> (b, a mod b)
    return a  # a adalah FPB


class Pecahan:  # bilangan rasional eksak p/q (pengganti fractions.Fraction)
    def __init__(self, pembilang, penyebut=1):  # membuat pecahan dan langsung menyederhanakannya
        if penyebut == 0:  # pembagian dengan nol tidak boleh
            raise ZeroDivisionError("penyebut nol")  # hentikan dengan pesan kesalahan
        if penyebut < 0:  # tanda negatif selalu disimpan di pembilang
            pembilang, penyebut = -pembilang, -penyebut  # pindahkan tanda
        g = fpb(pembilang, penyebut)  # FPB untuk menyederhanakan
        self.p = pembilang // g  # pembilang dalam bentuk paling sederhana
        self.q = penyebut // g  # penyebut dalam bentuk paling sederhana

    @staticmethod  # fungsi bantu yang tidak memerlukan self
    def _ke_pecahan(x):  # mengubah int menjadi Pecahan bila perlu
        return x if isinstance(x, Pecahan) else Pecahan(x)  # int n menjadi n/1

    def __add__(self, lain):  # a/b + c/d = (ad + cb) / bd
        lain = Pecahan._ke_pecahan(lain)  # pastikan operand kedua berupa Pecahan
        return Pecahan(self.p * lain.q + lain.p * self.q, self.q * lain.q)  # rumus penjumlahan pecahan

    __radd__ = __add__  # n + pecahan sama dengan pecahan + n

    def __truediv__(self, lain):  # (a/b) / (c/d) = ad / bc
        lain = Pecahan._ke_pecahan(lain)  # pastikan operand kedua berupa Pecahan
        return Pecahan(self.p * lain.q, self.q * lain.p)  # rumus pembagian pecahan

    def __rtruediv__(self, lain):  # n / (a/b) = nb / a
        return Pecahan._ke_pecahan(lain) / self  # ubah n menjadi Pecahan lalu bagi

    def __float__(self):  # konversi ke float untuk dibandingkan dengan hasil program
        return self.p / self.q  # pembagian biasa

    def __str__(self):  # tampilan "p/q" (atau "p" bila penyebutnya 1)
        return str(self.p) if self.q == 1 else str(self.p) + "/" + str(self.q)  # format pecahan


def cek(nama, program, tangan):  # membandingkan satu nilai program dengan hitungan tangan
    ok = abs(program - tangan) <= TOL  # lulus bila selisihnya dalam toleransi
    hasil.append(ok)  # catat status
    print(f"  [{'lulus' if ok else 'GAGAL'}] {nama:<38} program {program:>10.4f}   tangan {tangan:>10.4f}")  # cetak baris laporan


def fmtw(w):  # format vektor w menjadi "(a; b)"
    return "(" + "; ".join(f"{x:.4f}" for x in w) + ")"  # 4 angka desimal dipisah titik koma


def newton_1d(x, k, d1, d2):  # k langkah Newton satu dimensi x <- x - f'(x)/f''(x)
    xs = []  # tampungan jejak
    for _ in range(k):  # ulangi k kali
        x = x - d1(x) / d2(x)  # langkah Newton
        xs.append(x)  # simpan titik baru
    return xs  # kembalikan jejak x_1..x_k


# ------------------------------------------------- SK1: tarif parkir  # bagian SK1
print("SK1 - tarif parkir (Tabel 1)")  # judul bagian
m1 = sk1_kecil()  # model dengan 4 data tarif parkir
g0, H0 = m1.grad([0.0, 0.0]), m1.hessian([0.0, 0.0])  # gradien dan Hessian di titik awal (0, 0)
print(f"  F(0) = {m1.loss([0.0, 0.0]):.4f}, g(0) = {fmtw(g0)}, H = [{fmtw(H0[0])}, {fmtw(H0[1])}]")  # cetak nilai awal

w = [0.0, 0.0]  # SGD eta = 0,1, b = 1, data dipakai berurutan
opt = SGD(2, 0.1)  # buat optimizer SGD
jejak_sgd = []  # tampungan jejak (w, F)
for i in range(4):  # empat langkah, satu per data
    opt.step(w, m1.grad_batch(w, [i]))  # langkah dengan gradien satu sampel ke-i
    jejak_sgd.append((list(w), m1.loss(w)))  # simpan salinan w dan nilai loss
cek("SGD langkah 1: w1", jejak_sgd[0][0][1], 0.3)  # bandingkan dengan Tabel 1
cek("SGD langkah 2: w0", jejak_sgd[1][0][0], 0.71)  # bandingkan dengan Tabel 1
cek("SGD langkah 2: w1", jejak_sgd[1][0][1], 1.12)  # bandingkan dengan Tabel 1
cek("SGD langkah 3: w0", jejak_sgd[2][0][0], 1.003)  # bandingkan dengan Tabel 1
cek("SGD langkah 1: F", jejak_sgd[0][1], 14.0575)  # bandingkan dengan Tabel 1

w = [0.0, 0.0]  # Adam eta = 0,5, b = 1, urutan data sama
opt = Adam(2, 0.5)  # buat optimizer Adam
jejak_adam = []  # tampungan jejak (w, F)
for i in range(4):  # empat langkah
    opt.step(w, m1.grad_batch(w, [i]))  # langkah dengan gradien satu sampel ke-i
    jejak_adam.append((list(w), m1.loss(w)))  # simpan salinan w dan nilai loss
cek("Adam langkah 1: w0 (= eta)", jejak_adam[0][0][0], 0.5)  # langkah pertama Adam selalu sebesar eta
cek("Adam langkah 2: w0", jejak_adam[1][0][0], 1.0005)  # bandingkan dengan Tabel 1

w_nr, _ = NewtonRaphson().step(m1, [0.0, 0.0])  # Newton satu langkah dari w = (0, 0)
cek("Newton langkah 1: w0", w_nr[0], 1.0)  # Newton langsung tepat di w0 = 1
cek("Newton langkah 1: w1", w_nr[1], 2.0)  # Newton langsung tepat di w1 = 2
print("  Jejak lengkap (w / F):")  # judul tabel jejak
for k in range(4):  # cetak keempat langkah
    print(f"   {k + 1}: SGD {fmtw(jejak_sgd[k][0])} / {jejak_sgd[k][1]:.4f}   "  # kolom SGD
          f"Adam {fmtw(jejak_adam[k][0])} / {jejak_adam[k][1]:.4f}")  # kolom Adam
print(f"   Newton {fmtw(w_nr)} / F = {m1.loss(w_nr):.4f}, ||g|| = {sum(v * v for v in m1.grad(w_nr)) ** 0.5:.1e}")  # hasil Newton

# ----------------------------------------------------- SK2: dua lembah  # bagian SK2
print("\nSK2 - f(x) = x^4 - 2x^2 (Tabel 2)")  # judul bagian
nr2 = newton_1d(2.0, 6, df2, d2f2)  # Newton dari x0 = 2
nr03 = newton_1d(0.3, 4, df2, d2f2)  # Newton dari x0 = 0,3 (menuju maksimum)
cek("Newton x1 dari x0 = 2 (= 16/11)", nr2[0], 16 / 11)  # hasil hitungan tangan 16/11
cek("Newton x1 dari x0 = 0,3", nr03[0], -0.0740)  # hasil hitungan tangan -0,0740

x = [2.0]  # GD (SGD dengan gradien penuh) dari x0 = 2
opt = SGD(1, 0.05)  # eta = 0,05
gd = []  # tampungan jejak
for _ in range(4):  # empat langkah
    opt.step(x, [df2(x[0])])  # langkah dengan gradien f'(x)
    gd.append(x[0])  # simpan x
cek("GD (eta = 0,05) x1", gd[0], 0.8)  # bandingkan dengan Tabel 2
cek("GD (eta = 0,05) x2", gd[1], 0.8576)  # bandingkan dengan Tabel 2

x = [2.0]  # Adam dari x0 = 2
opt = Adam(1, 0.1)  # eta = 0,1
ad = []  # tampungan jejak
for _ in range(4):  # empat langkah
    opt.step(x, [df2(x[0])])  # langkah dengan gradien f'(x)
    ad.append(x[0])  # simpan x
cek("Adam (eta = 0,1) x1 (= 2 - eta)", ad[0], 1.9)  # langkah pertama sebesar eta

print("  k   GD        Adam      Newton(2)  Newton(0,3)")  # judul tabel
for k in range(4):  # cetak empat baris
    print(f"  {k + 1}   {gd[k]:.4f}    {ad[k]:.4f}    {nr2[k]:.4f}     {nr03[k]:+.4f}")  # satu baris tabel

# ------------------------------------------------- SK3: menghitung akar 2  # bagian SK3
print("\nSK3 - akar 2, F(x) = x^3/3 - 2x (Tabel 3 dan 4)")  # judul bagian
x = Pecahan(1)  # Newton dengan pecahan eksak, mulai dari x0 = 1
pecahan = []  # tampungan jejak pecahan
for _ in range(4):  # empat langkah
    x = (x + 2 / x) / 2  # rumus Babilonia x <- (x + 2/x) / 2
    pecahan.append(x)  # simpan pecahan
print("  Newton eksak dari x0 = 1:", ", ".join(str(p) for p in pecahan))  # cetak 3/2, 17/12, 577/408, ...
nr3 = newton_1d(1.0, 5, df3, d2f3)  # Newton dengan float
cek("Newton x1 (= 3/2)", nr3[0], 1.5)  # bandingkan dengan 3/2
cek("Newton x2 (= 17/12)", nr3[1], 17 / 12)  # bandingkan dengan 17/12
cek("Newton x3 (= 577/408)", nr3[2], 577 / 408)  # bandingkan dengan 577/408
assert [float(p) for p in pecahan[:3]] == nr3[:3]  # float dan pecahan eksak harus sama persis

x = [1.0]  # GD dari x0 = 1
opt = SGD(1, 0.25)  # eta = 0,25
gd3 = []  # tampungan jejak
for _ in range(8):  # delapan langkah
    opt.step(x, [df3(x[0])])  # langkah dengan gradien F'(x)
    gd3.append(x[0])  # simpan x
cek("GD (eta = 0,25) x1", gd3[0], 1.25)  # bandingkan dengan Tabel 3
cek("GD (eta = 0,25) x2", gd3[1], 1.359375)  # bandingkan dengan Tabel 3

x = [1.0]  # Adam dari x0 = 1
opt = Adam(1, 0.1)  # eta = 0,1
ad3 = []  # tampungan jejak
for _ in range(4):  # empat langkah
    opt.step(x, [df3(x[0])])  # langkah dengan gradien F'(x)
    ad3.append(x[0])  # simpan x
cek("Adam (eta = 0,1) x1 (= 1 + eta)", ad3[0], 1.1)  # langkah pertama sebesar eta

nrm = newton_1d(-1.0, 4, df3, d2f3)  # Newton dari x0 = -1 (menuju maksimum -akar 2)
cek("Newton dari x0 = -1: x1 (= -3/2)", nrm[0], -1.5)  # bandingkan dengan -3/2

akar2 = 2 ** 0.5  # nilai akar 2 sebagai acuan
print("  k   GD (0,25)   Adam (0,1)   Newton (x0=1)   Newton (x0=-1)")  # judul tabel
for k in range(4):  # cetak empat baris
    print(f"  {k + 1}   {gd3[k]:.6f}    {ad3[k]:.4f}       {nr3[k]:.10f}    {nrm[k]:+.6f}")  # satu baris tabel
print("  Laju kuadratik Newton: k, x_k, e_k = |x_k - akar 2|, e_k / e_{k-1}^2")  # judul tabel laju
e_prev = abs(1.0 - akar2)  # galat awal e_0
print(f"   0  1.0000000000  {e_prev:.2e}  --")  # baris k = 0
for k, xk in enumerate(nr3[:4], 1):  # baris k = 1..4
    e = abs(xk - akar2)  # galat e_k
    print(f"   {k}  {xk:.10f}  {e:.2e}  {e / e_prev ** 2:.4f}")  # rasio e_k / e_{k-1}^2
    e_prev = e  # simpan galat untuk langkah berikutnya
print(f"  teori F'''/(2F'') di akar 2 = 1/(2 akar 2) = {1 / (2 * akar2):.4f}")  # konstanta laju kuadratik teoretis

x = eliminasi_gauss([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 3.0]], [5.0, 4.0, 7.0])  # uji eliminasi Gauss (pivot awal nol)
assert all(abs(a - b) < 1e-12 for a, b in zip(x, [1.0, 2.0, 1.0])), x  # solusi yang diketahui adalah (1, 2, 1)

print(f"\n{sum(hasil)} dari {len(hasil)} nilai hitungan tangan cocok.")  # ringkasan
assert all(hasil)  # semua pengecekan harus lulus
print("Semua uji lulus.")  # pesan akhir
