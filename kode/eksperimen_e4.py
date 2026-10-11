# E4: sensitivitas parameter.  # tujuan eksperimen
# (a) ukuran batch b in {1, 8, 32, 128, 512} pada RL, 10 epoch, eta terbaik dari E3  # bagian (a)
#     (sensitivitas eta sudah dihitung pada grid E3);  # catatan bagian (a)
# (b) Newton: redaman alpha in {1; 0,5; 0,25} dan titik awal w0 = (0, c, 0),  # bagian (b)
#     c = 0..4, pada RL dan SK1 (n = 10^4).  # rincian bagian (b)
from optimizer import NewtonRaphson, train, data_sk1, data_logistik  # kode buatan sendiri
from util import SEEDS, mean, std, simpan, muat  # utilitas buatan sendiri (termasuk pembaca JSON)

N, B_LIST, EP = 10**4, [1, 8, 32, 128, 512], 10  # ukuran data, daftar ukuran batch, banyak epoch
e3 = muat("e3.json")  # baca hasil E3 (butuh eta terbaik dan F*)
R = {"bs": B_LIST, "epochs": EP, "batch": {}, "newton": {}}  # dictionary hasil

models = [data_logistik(N, s) for s in SEEDS]  # (a) data RL untuk 5 seed
Fs = e3["RL"]["Fstar"]  # nilai F* dari E3
for alg in ["SGD", "Adam"]:  # dua metode orde pertama
    eta = e3["RL"]["best_eta"][alg]  # eta terbaik dari E3
    R["batch"][alg] = {"eta": eta, "galat": [], "waktu_epoch": []}  # tampungan hasil
    for b in B_LIST:  # untuk setiap ukuran batch
        gal, ts = [], []  # galat akhir dan waktu per epoch
        for s, m in zip(SEEDS, models):  # untuk setiap seed
            w, _, t = train(m, alg, eta, [0.0] * m.d, b, EP, s, track=False)  # latih 10 epoch
            gal.append(m.loss(w) - Fs[s])  # galat terhadap F*
            ts.append(t[-1] / EP)  # rata-rata waktu per epoch
        R["batch"][alg]["galat"].append((mean(gal), std(gal)))  # simpan galat
        R["batch"][alg]["waktu_epoch"].append((mean(ts), std(ts)))  # simpan waktu
    print(alg, "galat", [f"{v[0]:.1e}" for v in R["batch"][alg]["galat"]],  # tampilkan galat
          "waktu/epoch", [f"{v[0]:.3f}" for v in R["batch"][alg]["waktu_epoch"]], flush=True)  # dan waktu

for sk, buat in [("RL", data_logistik), ("SK1", data_sk1)]:  # (b) Newton: redaman dan titik awal
    ms = models if sk == "RL" else [buat(N, s) for s in SEEDS]  # pakai ulang data RL, buat data SK1
    R["newton"][sk] = {}  # tampungan hasil studi kasus ini
    for alpha in [1.0, 0.5, 0.25]:  # tiga faktor redaman
        baris = []  # hasil untuk setiap titik awal
        for c in range(5):  # titik awal w0 = (0, c, 0), c = 0..4
            ok, its, alasan = 0, [], []  # banyak berhasil, iterasi, dan alasan gagal
            for m in ms:  # untuk setiap seed
                r = NewtonRaphson(alpha=alpha, tol=1e-10, T=500).run(m, [0.0, float(c), 0.0])  # jalankan Newton
                if r["ok"]:  # konvergen
                    ok += 1  # hitung keberhasilan
                    its.append(r["iter"])  # simpan banyak iterasi
                else:  # gagal
                    alasan.append(r["alasan"])  # simpan alasan kegagalan
            baris.append({"c": c, "berhasil": ok, "iter_mean": mean(its) if its else None,  # ringkasan titik awal ini
                          "iter": its, "gagal": alasan})  # rincian iterasi dan kegagalan
        R["newton"][sk][str(alpha)] = baris  # simpan per nilai alpha
        print(sk, "alpha", alpha, [f"{b['berhasil']}/5·{b['iter_mean']:.0f}" if b["iter_mean"]  # tampilkan berhasil/5 dan iterasi
                                   else f"0/5·gagal" for b in baris], flush=True)  # atau tanda gagal

simpan("e4.json", R)  # tulis hasil ke results/e4.json
