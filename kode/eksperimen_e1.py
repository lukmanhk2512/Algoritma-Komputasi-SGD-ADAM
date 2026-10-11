# E1: waktu terhadap ukuran data n (SK1 dan RL, d = 3).  # tujuan eksperimen
# Satu epoch SGD/Adam (b = 32) dibandingkan dengan satu iterasi Newton.  # yang dibandingkan
import time  # pengecualian yang disetujui: modul time hanya untuk mengukur waktu
from optimizer import LCG, SGD, Adam, NewtonRaphson, train_epoch, data_sk1, data_logistik  # kode buatan sendiri
from util import SEEDS, mean, std, slope_loglog, simpan  # utilitas buatan sendiri

NS = [10**2, 10**3, 10**4, 10**5, 10**6]  # ukuran data yang diuji
B = 32  # ukuran mini-batch SGD/Adam
ETA = {"SGD": 0.01, "Adam": 0.001}  # learning rate (tidak memengaruhi waktu)
R = {"n": NS, "b": B}  # dictionary hasil yang akan disimpan

for sk, buat in [("SK1", data_sk1), ("RL", data_logistik)]:  # dua studi kasus berdata banyak
    R[sk] = {"SGD": [], "Adam": [], "Newton": []}  # tampungan (rata-rata, simpangan baku) per algoritma
    for n in NS:  # untuk setiap ukuran data
        model = buat(n, 0)  # data dibangkitkan sekali per n
        d = model.d  # dimensi (= 3)
        for alg in ["SGD", "Adam", "Newton"]:  # untuk setiap algoritma
            ts = []  # waktu dari 5 pengulangan
            for s in SEEDS:  # seed 0-4
                w = [0.0] * d  # titik awal nol
                if alg == "Newton":  # Newton: ukur satu iterasi
                    t0 = time.perf_counter()  # waktu mulai
                    NewtonRaphson().step(model, w)  # satu iterasi Newton
                    ts.append(time.perf_counter() - t0)  # simpan lama iterasi
                else:  # SGD/Adam: ukur satu epoch
                    opt = (SGD if alg == "SGD" else Adam)(d, ETA[alg])  # buat optimizer
                    rng = LCG(s)  # pembangkit acak untuk permutasi
                    t0 = time.perf_counter()  # waktu mulai
                    train_epoch(model, opt, w, B, rng)  # satu epoch pelatihan
                    ts.append(time.perf_counter() - t0)  # simpan lama epoch
            R[sk][alg].append((mean(ts), std(ts)))  # simpan rata-rata dan simpangan baku
        print(sk, n, {a: round(R[sk][a][-1][0], 5) for a in R[sk]}, flush=True)  # tampilkan kemajuan
    R[sk + "_slope"] = {a: slope_loglog(NS, [v[0] for v in R[sk][a]]) for a in R[sk]}  # kemiringan log-log waktu vs n
    print(sk, "kemiringan", {a: round(v, 3) for a, v in R[sk + "_slope"].items()})  # tampilkan kemiringan

simpan("e1.json", R)  # tulis hasil ke results/e1.json
