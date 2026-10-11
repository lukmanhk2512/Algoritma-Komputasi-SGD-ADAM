# E5: perilaku pada SK2, f(x) = x^4 - 2x^2, dan SK3, F(x) = x^3/3 - 2x.  # tujuan eksperimen
# (a) lintasan galat dari x0 = 2 (GD eta = 0,05; Adam eta = 0,1; Newton);  # bagian (a)
# (b) titik akhir untuk 501 titik awal x0 in [-2,5; 2,5];  # bagian (b)
# (c) banyak iterasi dari x0 = 2 sampai |x - x*| < 1e-6 untuk grid eta;  # bagian (c)
# (d) laju kuadratik Newton e_k / e_{k-1}^2 pada SK2;  # bagian (d)
# (e) SK3 (akar 2): iterasi dari x0 = 1, laju kuadratik, dan titik akhir untuk  # bagian (e)
#     501 titik awal (Newton ke -akar 2 = maksimum bila x0 < 0).  # rincian bagian (e)
from optimizer import SGD, Adam, df2, d2f2, df3, d2f3, isfinite  # kode buatan sendiri
from util import simpan  # penyimpan JSON buatan sendiri

R = {}  # dictionary hasil


def jalankan(alg, x0, T, eta=None, d1=df2, d2=d2f2):  # mengembalikan daftar x_0..x_T
    xs = [x0]  # jejak dimulai dari titik awal
    if alg == "Newton":  # cabang Newton
        x = x0  # titik saat ini
        for _ in range(T):  # paling banyak T langkah
            h = d2(x)  # turunan kedua
            if h == 0.0:  # f'' = 0
                break  # langkah Newton tidak terdefinisi
            x = x - d1(x) / h  # langkah Newton
            xs.append(x)  # simpan titik baru
            if not isfinite(x) or abs(x) > 1e6:  # meledak
                break  # hentikan
        return xs  # kembalikan jejak Newton
    x = [x0]  # GD/Adam memakai list agar bisa diubah oleh optimizer
    opt = SGD(1, eta) if alg == "GD" else Adam(1, eta)  # pilih optimizer
    for _ in range(T):  # paling banyak T langkah
        try:  # pangkat tiga bilangan besar bisa overflow
            opt.step(x, [d1(x[0])])  # langkah dengan gradien penuh
        except OverflowError:  # bilangan terlalu besar
            xs.append(float("inf"))  # tandai divergen
            break  # hentikan
        xs.append(x[0])  # simpan titik baru
        if not isfinite(x[0]) or abs(x[0]) > 1e6:  # meledak
            break  # hentikan
    return xs  # kembalikan jejak GD/Adam


R["lintasan"] = {}  # (a) lintasan dari x0 = 2
for alg, eta, T in [("GD", 0.05, 60), ("Adam", 0.1, 60), ("Newton", None, 6)]:  # tiga algoritma
    xs = jalankan(alg, 2.0, T, eta)  # jalankan dari x0 = 2
    R["lintasan"][alg] = {"eta": eta, "galat": [abs(x - 1.0) for x in xs]}  # galat terhadap minimum x = 1

X0 = [round(-2.5 + 0.01 * i, 10) for i in range(501)]  # (b) 501 titik awal dengan jarak 0,01


def kelas(x):  # mengelompokkan titik akhir SK2
    if not isfinite(x) or abs(x) > 1e6:  # meledak
        return "divergen"  # kelas divergen
    for nama, c in [("-1", -1.0), ("+1", 1.0), ("0", 0.0)]:  # tiga titik stasioner
        if abs(x - c) < 0.05:  # cukup dekat (Adam dengan eta tetap masih berosilasi kecil)
            return nama  # kelas titik stasioner tersebut
    return "lain"  # tidak dekat titik stasioner mana pun


R["peta"] = {"x0": X0}  # tampungan peta titik awal
for alg, eta, T in [("Newton", None, 100), ("Adam", 0.05, 3000), ("GD", 0.05, 3000)]:  # tiga algoritma
    R["peta"][alg] = [kelas(jalankan(alg, x0, T, eta)[-1]) for x0 in X0]  # kelas titik akhir untuk setiap x0
    hit = {k: R["peta"][alg].count(k) for k in ["-1", "+1", "0", "lain", "divergen"]}  # hitung tiap kelas
    print(alg, hit)  # tampilkan ringkasan
maks = [x for x, k in zip(X0, R["peta"]["Newton"]) if k == "0"]  # titik awal Newton yang berakhir di maksimum
R["newton_ke_maksimum"] = {"jumlah": len(maks), "x0_min": min(maks), "x0_max": max(maks)}  # ringkasan
diam = {a: [x for x, k in zip(X0, R["peta"][a]) if k == "0"] for a in ["GD", "Adam"]}  # titik awal GD/Adam yang berakhir di 0
R["gd_adam_ke_maksimum"] = diam  # hanya x0 = 0 tepat (gradien nol sejak awal)
print("GD/Adam berakhir di 0 untuk x0 =", diam)  # tampilkan
print("Newton ke maksimum:", len(maks), "titik, x0 dari", round(min(maks), 2), "sampai", round(max(maks), 2))  # tampilkan

ETAS = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]  # (c) grid eta


def iterasi_sampai(xs):  # langkah pertama yang mencapai |x - (+-1)| < 1e-6
    for k, x in enumerate(xs):  # telusuri jejak
        for c in (1.0, -1.0):  # dua minimum
            if abs(x - c) < 1e-6:  # sudah cukup dekat
                return k, c  # kembalikan banyak iterasi dan minimum yang dicapai
    return None, ("divergen" if not isfinite(xs[-1]) or abs(xs[-1]) > 1e6 else "tidak tercapai")  # tidak mencapai minimum


R["iterasi"] = {}  # tampungan hasil (c)
for alg in ["GD", "Adam"]:  # dua metode orde pertama
    R["iterasi"][alg] = []  # list hasil per eta
    for eta in ETAS:  # telusuri grid
        k, tujuan = iterasi_sampai(jalankan(alg, 2.0, 5000, eta))  # jalankan sampai 5000 langkah
        R["iterasi"][alg].append({"eta": eta, "iter": k, "tujuan": tujuan})  # simpan hasil
    print(alg, [(r["eta"], r["iter"], r["tujuan"]) for r in R["iterasi"][alg]])  # tampilkan
k, tujuan = iterasi_sampai(jalankan("Newton", 2.0, 50))  # Newton dari x0 = 2
R["iterasi"]["Newton"] = {"iter": k, "tujuan": tujuan}  # simpan hasil Newton
print("Newton", k, tujuan)  # tampilkan

xs = jalankan("Newton", 2.0, 6)  # (d) laju kuadratik Newton
R["laju"] = []  # tampungan hasil (d)
for i, x in enumerate(xs):  # untuk setiap iterasi
    e = abs(x - 1.0)  # galat e_k
    rasio = e / abs(xs[i - 1] - 1.0) ** 2 if i else None  # rasio e_k / e_{k-1}^2 (tidak ada untuk k = 0)
    R["laju"].append({"k": i, "x": x, "e": e, "rasio": rasio})  # simpan
    print(i, f"{x:.12f}", f"{e:.1e}", "--" if rasio is None else f"{rasio:.4f}")  # tampilkan
print("teori f'''(1)/(2 f''(1)) =", 24 / (2 * 8))  # konstanta teoretis = 1,5

A2 = 2 ** 0.5  # (e) SK3: nilai akar 2 sebagai acuan
R["SK3"] = {}  # tampungan hasil (e)
it = {}  # banyak iterasi dari x0 = 1
for alg, eta in [("Newton", None), ("GD", 0.25), ("Adam", 0.1), ("Adam", 0.01)]:  # konfigurasi yang diuji
    xs = jalankan(alg, 1.0, 20000, eta, df3, d2f3)  # jalankan pada SK3
    k = next((i for i, x in enumerate(xs) if abs(x - A2) < 1e-6), None)  # iterasi pertama yang cukup dekat akar 2
    it[f"{alg} {eta}"] = k  # simpan
R["SK3"]["iterasi_x0_1"] = it  # simpan semua
print("SK3 iterasi dari x0 = 1 sampai |x - akar 2| < 1e-6:", it)  # tampilkan
xs = jalankan("Newton", 1.0, 5, None, df3, d2f3)  # jejak Newton untuk laju kuadratik SK3
R["SK3"]["laju"] = [{"k": i, "x": x, "e": abs(x - A2),  # galat e_k
                     "rasio": abs(x - A2) / abs(xs[i - 1] - A2) ** 2 if i else None}  # rasio e_k / e_{k-1}^2
                    for i, x in enumerate(xs) if abs(x - A2) > 0 or i == 0]  # lewati galat nol (tidak bisa dibagi)


def kelas3(xs):  # mengelompokkan titik akhir SK3
    x = xs[-1]  # titik akhir
    if len(xs) == 1:  # tidak ada langkah sama sekali
        return "singular"  # x0 = 0: F''(0) = 0, langkah Newton tak terdefinisi
    if not isfinite(x) or x < -10:  # turun tanpa batas
        return "divergen"  # F tak terbatas di bawah: x terus turun ke -tak hingga
    if abs(x - A2) < 0.05:  # dekat +akar 2
        return "+akar2"  # minimum
    if abs(x + A2) < 0.05:  # dekat -akar 2
        return "-akar2"  # maksimum
    return "lain"  # selain itu


R["SK3"]["peta"] = {"x0": X0}  # peta titik awal SK3
for alg, eta, T in [("Newton", None, 100), ("GD", 0.25, 3000), ("Adam", 0.05, 3000)]:  # tiga algoritma
    hasil = [kelas3(jalankan(alg, x0, T, eta, df3, d2f3)) for x0 in X0]  # kelas titik akhir untuk setiap x0
    R["SK3"]["peta"][alg] = hasil  # simpan
    print("SK3", alg, {k: hasil.count(k) for k in ["+akar2", "-akar2", "singular", "lain", "divergen"]})  # tampilkan ringkasan
for alg in ["GD", "Adam"]:  # rentang titik awal yang divergen
    dv = [x for x, k in zip(X0, R["SK3"]["peta"][alg]) if k == "divergen"]  # titik awal divergen
    print("  ", alg, "divergen untuk x0 dari", min(dv), "sampai", max(dv))  # tampilkan rentangnya

simpan("e5.json", R)  # tulis hasil ke results/e5.json
