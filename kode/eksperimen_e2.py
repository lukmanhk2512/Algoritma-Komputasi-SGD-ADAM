# E2: waktu terhadap dimensi d.  # tujuan eksperimen
# (a) RL, n = 1.000, d = 2..64: satu epoch SGD/Adam vs satu iterasi Newton,  # bagian (a)
#     iterasi Newton dipecah menjadi gradien, Hessian, dan eliminasi Gauss.  # rincian bagian (a)
# (b) eliminasi Gauss saja, d = 16..256.  # bagian (b)
import time  # pengecualian yang disetujui: modul time hanya untuk mengukur waktu
from optimizer import LCG, SGD, Adam, train_epoch, eliminasi_gauss, data_logistik  # kode buatan sendiri
from util import SEEDS, mean, std, slope_loglog, simpan  # utilitas buatan sendiri

N = 1000  # ukuran data tetap
DS = [2, 4, 8, 16, 32, 64]  # dimensi untuk bagian (a)
DG = [16, 32, 64, 128, 256]  # dimensi untuk bagian (b)
B = 32  # ukuran mini-batch
R = {"n": N, "d": DS, "b": B, "res": {k: [] for k in  # dictionary hasil
                                       ["SGD", "Adam", "Newton", "Newton:gradien",  # algoritma utuh dan komponen Newton
                                        "Newton:Hessian", "Newton:Gauss"]}}  # komponen Newton lainnya

for d in DS:  # untuk setiap dimensi
    model = data_logistik(N, 0, d=d)  # data regresi logistik berdimensi d
    acc = {k: [] for k in R["res"]}  # tampungan waktu dari 5 pengulangan
    for s in SEEDS:  # seed 0-4
        for alg, kelas in [("SGD", SGD), ("Adam", Adam)]:  # dua metode orde pertama
            w = [0.0] * d  # titik awal nol
            opt = kelas(d, 0.01)  # buat optimizer
            t0 = time.perf_counter()  # waktu mulai
            train_epoch(model, opt, w, B, LCG(s))  # satu epoch
            acc[alg].append(time.perf_counter() - t0)  # simpan lama epoch
        w = [0.0] * d  # satu iterasi Newton, setiap komponen diukur terpisah
        t0 = time.perf_counter()  # waktu mulai
        g = model.grad(w)  # komponen 1: gradien
        t1 = time.perf_counter()  # waktu setelah gradien
        H = model.hessian(w)  # komponen 2: Hessian
        t2 = time.perf_counter()  # waktu setelah Hessian
        p = eliminasi_gauss(H, g)  # komponen 3: eliminasi Gauss
        w = [a - b for a, b in zip(w, p)]  # langkah Newton
        t3 = time.perf_counter()  # waktu selesai
        acc["Newton:gradien"].append(t1 - t0)  # lama komponen gradien
        acc["Newton:Hessian"].append(t2 - t1)  # lama komponen Hessian
        acc["Newton:Gauss"].append(t3 - t2)  # lama komponen eliminasi Gauss
        acc["Newton"].append(t3 - t0)  # lama satu iterasi utuh
    for k in R["res"]:  # untuk setiap kategori waktu
        R["res"][k].append((mean(acc[k]), std(acc[k])))  # simpan rata-rata dan simpangan baku
    print("d =", d, {k: f"{v[-1][0]:.2e}" for k, v in R["res"].items()}, flush=True)  # tampilkan kemajuan

I8 = DS.index(8)  # kemiringan dihitung pada d >= 8 karena overhead Python dominan pada d kecil
R["slope"] = {k: slope_loglog(DS[I8:], [v[0] for v in R["res"][k][I8:]]) for k in R["res"]}  # kemiringan log-log
i64 = DS.index(64)  # posisi d = 64
R["d64"] = {"newton_per_sgd": R["res"]["Newton"][i64][0] / R["res"]["SGD"][i64][0],  # rasio Newton / SGD pada d = 64
            "porsi_hessian": R["res"]["Newton:Hessian"][i64][0] / R["res"]["Newton"][i64][0]}  # porsi waktu Hessian
print("kemiringan", {k: round(v, 3) for k, v in R["slope"].items()}, R["d64"])  # tampilkan ringkasan

R["gauss"] = {"d": DG, "res": []}  # (b) eliminasi Gauss saja
for d in DG:  # untuk setiap dimensi
    ts = []  # waktu dari 5 pengulangan
    for s in SEEDS:  # seed 0-4
        rng = LCG(500 + s)  # pembangkit acak untuk matriks
        A = [[rng.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(d)]  # matriks acak d x d
        for i in range(d):  # perbesar diagonal
            A[i][i] += d  # dominan diagonal sehingga pasti nonsingular
        b = [rng.uniform(-1.0, 1.0) for _ in range(d)]  # ruas kanan acak
        t0 = time.perf_counter()  # waktu mulai
        eliminasi_gauss(A, b)  # selesaikan A x = b
        ts.append(time.perf_counter() - t0)  # simpan lama penyelesaian
    R["gauss"]["res"].append((mean(ts), std(ts)))  # simpan rata-rata dan simpangan baku
    print("Gauss d =", d, f"{R['gauss']['res'][-1][0]:.3e}", flush=True)  # tampilkan kemajuan
g = [v[0] for v in R["gauss"]["res"]]  # rata-rata waktu Gauss
R["gauss"]["slope"] = slope_loglog(DG, g)  # kemiringan log-log (teori: 3)
R["gauss"]["rasio_128_256"] = g[-1] / g[-2]  # rasio waktu d = 256 terhadap d = 128 (teori: 8)
print("kemiringan Gauss", round(R["gauss"]["slope"], 3), "rasio 128->256", round(g[-1] / g[-2], 2))  # tampilkan ringkasan

simpan("e2.json", R)  # tulis hasil ke results/e2.json
