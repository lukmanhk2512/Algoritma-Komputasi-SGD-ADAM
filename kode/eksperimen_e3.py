# E3: kecepatan konvergensi pada SK1 dan RL (n = 10^4, d = 3, b = 32).  # tujuan eksperimen
# 1. Grid learning rate (20 epoch) untuk memilih eta terbaik SGD/Adam berdasarkan  # langkah 1
#    loss latih akhir. Hasil grid ini juga dipakai untuk Gambar 6a-b.  # kegunaan langkah 1
# 2. Kurva F(w) - F* per epoch dengan eta terbaik; Newton per iterasi sampai ||g|| < 1e-10.  # langkah 2
# F* dihitung dengan Newton sampai ||g|| < 1e-12; pada SK1 juga dicek terhadap  # cara menghitung F*
# solusi eksak kuadrat terkecil (persamaan normal X^T X w = X^T y).  # pengecekan tambahan
import time  # pengecualian yang disetujui: modul time hanya untuk mengukur waktu
from optimizer import (NewtonRaphson, eliminasi_gauss, norm, isfinite, train,  # kode buatan sendiri
                       data_sk1, data_logistik)  # pembangkit data
from util import SEEDS, mean, std, simpan  # utilitas buatan sendiri

N, EP, B = 10**4, 20, 32  # ukuran data, banyak epoch, ukuran batch
GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]  # grid learning rate
R = {"n": N, "epochs": EP, "b": B, "grid": GRID}  # dictionary hasil


def fstar(model):  # nilai minimum F* dihitung dengan Newton yang sangat teliti
    r = NewtonRaphson(tol=1e-12, T=50).run(model, [0.0] * model.d)  # Newton sampai ||g|| < 1e-12
    return model.loss(r["w"]), r["w"]  # kembalikan F* dan w*


for sk, buat in [("SK1", data_sk1), ("RL", data_logistik)]:  # dua studi kasus
    models = [buat(N, s) for s in SEEDS]  # satu himpunan data per seed
    Fs = []  # nilai F* per seed
    for m in models:  # untuk setiap himpunan data
        f, w = fstar(m)  # hitung F* dan w*
        Fs.append(f)  # simpan F*
        if sk == "SK1":  # cek terhadap persamaan normal
            d = m.d  # dimensi
            XtX = [[sum(x[j] * x[k] for x in m.X) for k in range(d)] for j in range(d)]  # matriks X^T X
            Xty = [sum(x[j] * y for x, y in zip(m.X, m.y)) for j in range(d)]  # vektor X^T y
            w_ls = eliminasi_gauss(XtX, Xty)  # solusi kuadrat terkecil eksak
            assert max(abs(a - b) for a, b in zip(w, w_ls)) < 1e-8  # Newton harus sama dengan solusi eksak
    R[sk] = {"Fstar": Fs, "grid": {}, "kurva": {}}  # tampungan hasil studi kasus ini

    best = {}  # 1. eta terbaik per algoritma
    for alg in ["SGD", "Adam"]:  # dua metode orde pertama
        res = []  # hasil untuk setiap eta pada grid
        for eta in GRID:  # telusuri grid
            gal = []  # galat F - F* untuk 5 seed
            for s, m in zip(SEEDS, models):  # untuk setiap seed
                w, _, _ = train(m, alg, eta, [0.0] * m.d, B, EP, s, track=False)  # latih 20 epoch tanpa mencatat loss
                f = m.loss(w) if all(isfinite(v) and abs(v) < 1e100 for v in w) else float("inf")  # loss akhir (inf bila divergen)
                gal.append(f - Fs[s])  # galat terhadap F*
            res.append((mean(gal), std(gal)) if all(isfinite(g) for g in gal) else None)  # None menandai divergen
        R[sk]["grid"][alg] = res  # simpan hasil grid
        ok = [(r[0], e) for r, e in zip(res, GRID) if r is not None]  # pasangan (galat, eta) yang tidak divergen
        best[alg] = min(ok)[1]  # eta dengan galat terkecil
        print(sk, alg, [None if r is None else f"{r[0]:.1e}" for r in res], "terbaik", best[alg], flush=True)  # tampilkan hasil grid
    R[sk]["best_eta"] = best  # simpan eta terbaik

    for alg in ["SGD", "Adam"]:  # 2. kurva konvergensi dengan eta terbaik
        H, T = [], []  # riwayat galat dan total waktu per seed
        for s, m in zip(SEEDS, models):  # untuk setiap seed
            _, h, t = train(m, alg, best[alg], [0.0] * m.d, B, EP, s)  # latih sambil mencatat loss
            H.append([v - Fs[s] for v in h])  # riwayat galat F - F*
            T.append(t[-1])  # total waktu 20 epoch
        hm = [mean(c) for c in zip(*H)]  # rata-rata galat per epoch
        k = [i for i, v in enumerate(hm) if v < 1e-3]  # epoch-epoch dengan galat < 1e-3
        fin = [h[-1] for h in H]  # galat akhir per seed
        R[sk]["kurva"][alg] = {"mean": hm, "std": [std(c) for c in zip(*H)],  # kurva rata-rata dan simpangan baku
                               "final": (mean(fin), std(fin)), "epoch_1e-3": k[0] if k else None,  # galat akhir dan epoch pertama < 1e-3
                               "waktu": mean(T)}  # rata-rata total waktu
        print(sk, alg, "akhir %.1e ± %.1e" % R[sk]["kurva"][alg]["final"],  # tampilkan galat akhir
              "epoch<1e-3:", R[sk]["kurva"][alg]["epoch_1e-3"], "waktu %.3f" % mean(T), flush=True)  # dan waktu

    H, T, IT = [], [], []  # Newton: riwayat galat, waktu, dan banyak iterasi
    for s, m in zip(SEEDS, models):  # untuk setiap seed
        w = [0.0] * m.d  # titik awal nol
        h = [m.loss(w) - Fs[s]]  # galat awal
        el = 0.0  # total waktu (tanpa evaluasi loss)
        it = 0  # banyak iterasi
        nr = NewtonRaphson()  # Newton murni
        while True:  # iterasi sampai konvergen
            t0 = time.perf_counter()  # waktu mulai
            gn = norm(m.grad(w))  # besar gradien
            if gn < 1e-10 or it >= 50:  # kriteria berhenti
                el += time.perf_counter() - t0  # tambahkan waktu pemeriksaan terakhir
                break  # keluar dari perulangan
            w, _ = nr.step(m, w)  # satu iterasi Newton
            el += time.perf_counter() - t0  # tambahkan waktu iterasi
            it += 1  # hitung iterasi
            h.append(m.loss(w) - Fs[s])  # catat galat (di luar pengukuran waktu)
        H.append(h)  # simpan riwayat galat
        T.append(el)  # simpan total waktu
        IT.append(it)  # simpan banyak iterasi
    L = max(len(h) for h in H)  # panjang riwayat terpanjang
    H = [h + [h[-1]] * (L - len(h)) for h in H]  # samakan panjang dengan mengulang nilai terakhir
    hm = [mean(c) for c in zip(*H)]  # rata-rata galat per iterasi
    k = [i for i, v in enumerate(hm) if v < 1e-3]  # iterasi-iterasi dengan galat < 1e-3
    R[sk]["kurva"]["Newton"] = {"mean": hm, "std": [std(c) for c in zip(*H)],  # kurva rata-rata dan simpangan baku
                                "final": (mean(h[-1] for h in H), std([h[-1] for h in H])),  # galat akhir
                                "epoch_1e-3": k[0] if k else None, "waktu": mean(T),  # iterasi pertama < 1e-3 dan waktu
                                "iterasi": IT}  # banyak iterasi per seed
    print(sk, "Newton iterasi", IT, "galat", [f"{v:.0e}" for v in hm], "waktu %.3f" % mean(T), flush=True)  # tampilkan ringkasan Newton

simpan("e3.json", R)  # tulis hasil ke results/e3.json
