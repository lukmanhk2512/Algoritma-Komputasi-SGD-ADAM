# Implementasi inti SGD, Adam, dan Newton-Raphson dengan Python murni.  # tujuan modul
#  # (baris kosong komentar)
# Tidak memakai library apa pun. Vektor ditulis sebagai list, matriks sebagai list of list.  # batasan
# Pembangkit bilangan acak (LCG 64-bit), logaritma natural, permutasi Fisher-Yates,  # yang ditulis sendiri
# dan eliminasi Gauss semuanya ditulis sendiri. Pengecualian: modul time hanya  # pengecualian
# dipakai di fungsi train() untuk mengukur waktu.  # alasan pengecualian
#  # (baris kosong komentar)
# Isi modul:  # daftar isi
#   - utilitas numerik : LCG, ln, ln1p, eksp, sigmoid, softplus, dot, norm  # bagian 1
#   - studi kasus kecil: SK1 tarif parkir, SK2 fungsi dua lembah, SK3 akar 2  # bagian 2
#   - model skala besar: RegresiLinear (SK1 diperbesar) dan RegresiLogistik (RL)  # bagian 3
#   - optimizer        : SGD, Adam, NewtonRaphson, eliminasi_gauss  # bagian 4
#   - pelatihan        : train_epoch dan train (mini-batch untuk SGD/Adam)  # bagian 5

# ====================================================== utilitas numerik  # bagian 1
E = 2.718281828459045  # bilangan Euler e
LN2 = 0.6931471805599453  # nilai ln 2, dipakai oleh fungsi ln()


class LCG:  # pembangkit bilangan acak linear kongruensial 64-bit (Knuth, MMIX)
    A = 6364136223846793005  # pengali a pada x_{k+1} = (a x_k + c) mod 2^64
    C = 1442695040888963407  # penambah c
    MASK = (1 << 64) - 1  # 2^64 - 1, operasi "& MASK" sama dengan mod 2^64

    def __init__(self, seed=0):  # membuat pembangkit dari sebuah seed
        self.state = (seed * 0x9E3779B97F4A7C15 + 0x2545F4914F6CDD1D) & self.MASK  # acak seed agar seed 0,1,2 tidak mirip
        for _ in range(4):  # buang 4 bilangan pertama
            self.next_int()  # supaya state benar-benar tercampur

    def next_int(self):  # satu langkah LCG
        self.state = (self.A * self.state + self.C) & self.MASK  # x_{k+1} = (a x_k + c) mod 2^64
        return self.state  # kembalikan bilangan bulat 64-bit

    def uniform(self, a=0.0, b=1.0):  # bilangan acak seragam di [a, b)
        return a + (b - a) * ((self.next_int() >> 11) / 9007199254740992.0)  # 53 bit teratas / 2^53, lalu diskalakan

    def normal(self):  # bilangan acak hampir normal baku
        return sum(self.uniform() for _ in range(12)) - 6.0  # Irwin-Hall: jumlah 12 U(0,1) dikurangi 6

    def randbelow(self, m):  # bilangan bulat acak di {0, ..., m-1}
        return int(self.uniform() * m)  # U(0,1) dikali m lalu dibulatkan ke bawah

    def permutation(self, n):  # permutasi acak Fisher-Yates, Theta(n)
        idx = list(range(n))  # mulai dari urutan 0, 1, ..., n-1
        for i in range(n - 1, 0, -1):  # telusuri dari belakang ke depan
            j = self.randbelow(i + 1)  # pilih posisi acak j di 0..i
            idx[i], idx[j] = idx[j], idx[i]  # tukar elemen i dan j
        return idx  # kembalikan urutan yang sudah diacak


def ln1p(t):  # ln(1 + t) untuk 0 <= t <= 1 lewat deret atanh
    s = t / (2.0 + t)  # ln(1+t) = 2 (s + s^3/3 + s^5/5 + ...), dengan s = t/(2+t) <= 1/3
    s2 = s * s  # s^2 dipakai berulang untuk menaikkan pangkat
    term = s  # suku pangkat ganjil saat ini, mulai dari s^1
    total = 0.0  # jumlah deret
    k = 1  # penyebut suku (1, 3, 5, ...)
    while True:  # tambahkan suku sampai cukup kecil
        add = term / k  # suku ke-k: s^k / k
        total += add  # akumulasi ke jumlah
        if abs(add) < 1e-18 * (abs(total) + 1e-300):  # suku sudah tidak berpengaruh pada presisi double
            break  # deret dianggap konvergen
        term *= s2  # naikkan pangkat dua tingkat (s^k -> s^(k+2))
        k += 2  # penyebut berikutnya
    return 2.0 * total  # kalikan 2 sesuai rumus deret atanh


def ln(x):  # logaritma natural untuk x > 0
    if x <= 0.0:  # ln tidak terdefinisi untuk x <= 0
        raise ValueError("ln hanya untuk x > 0")  # hentikan dengan pesan kesalahan
    k = 0  # pangkat dua: x = m * 2^k
    while x >= 2.0:  # selama x terlalu besar
        x *= 0.5  # bagi dua
        k += 1  # catat satu pangkat dua
    while x < 1.0:  # selama x terlalu kecil
        x *= 2.0  # kali dua
        k -= 1  # catat pangkat dua negatif
    return k * LN2 + ln1p(x - 1.0)  # ln x = k ln 2 + ln(m), dengan m di [1, 2)


def eksp(z):  # e^z memakai operator ** bawaan Python
    if z > 700.0:  # e^710 sudah melampaui batas double
        z = 700.0  # potong agar tidak overflow
    elif z < -700.0:  # sisi negatif juga dipotong
        z = -700.0  # hasilnya tetap mendekati nol
    return E ** z  # pangkatkan bilangan Euler


def sigmoid(z):  # sigma(z) = 1 / (1 + e^-z), ditulis stabil
    if z >= 0.0:  # untuk z tak negatif e^-z <= 1, aman
        return 1.0 / (1.0 + eksp(-z))  # rumus langsung
    ez = eksp(z)  # untuk z negatif, pakai e^z agar tidak overflow
    return ez / (1.0 + ez)  # bentuk setara: e^z / (1 + e^z)


def softplus(z):  # ln(1 + e^z) yang stabil
    if z > 0.0:  # untuk z positif e^z bisa sangat besar
        return z + ln1p(eksp(-z))  # ln(1 + e^z) = z + ln(1 + e^-z)
    return ln1p(eksp(z))  # untuk z <= 0, e^z <= 1 sehingga aman


def dot(u, v):  # hasil kali titik dua vektor
    s = 0.0  # akumulator jumlah
    for a, b in zip(u, v):  # pasangkan elemen-elemen yang seposisi
        s += a * b  # tambahkan hasil kali elemen
    return s  # kembalikan u . v


def norm(u):  # norma Euclid ||u||
    return dot(u, u) ** 0.5  # akar dari u . u


def isfinite(x):  # True jika x bukan NaN dan bukan tak hingga
    return x == x and x not in (float("inf"), float("-inf"))  # NaN tidak sama dengan dirinya sendiri


# ========================================================= studi kasus  # bagian 2 dan 3
class Model:  # kerangka model dengan data X (baris sudah memuat bias 1) dan y
    def __init__(self, X, y):  # menyimpan data
        self.X = X  # matriks fitur n x d (list of list)
        self.y = y  # vektor target panjang n
        self.n = len(X)  # banyak sampel
        self.d = len(X[0])  # banyak parameter (termasuk bias)

    def grad_batch(self, w, idx):  # gradien mini-batch (baris 6 Algoritma 1): 4bd + d operasi
        d = self.d  # simpan d di variabel lokal agar cepat
        g = [0.0] * d  # vektor gradien mulai dari nol
        X, y = self.X, self.y  # variabel lokal agar akses lebih cepat
        for i in idx:  # untuk setiap sampel di dalam batch
            xi = X[i]  # baris fitur sampel ke-i
            r = self.residu(dot(w, xi), y[i])  # residu r = turunan galat terhadap z = w . x_i
            for j in range(d):  # untuk setiap parameter
                g[j] += r * xi[j]  # gradien sampel = r * x_i
        inv = 1.0 / len(idx)  # 1/b untuk merata-ratakan
        for j in range(d):  # untuk setiap komponen gradien
            g[j] *= inv  # bagi dengan ukuran batch
        return g  # kembalikan gradien rata-rata

    def grad(self, w):  # gradien penuh (baris 3 Algoritma 2): 4nd + d operasi
        return self.grad_batch(w, range(self.n))  # sama dengan mini-batch berisi seluruh data

    def hessian(self, w):  # Hessian penuh (baris 5 Algoritma 2): 2nd^2 + O(nd) operasi
        d = self.d  # dimensi
        H = [[0.0] * d for _ in range(d)]  # matriks d x d berisi nol
        X = self.X  # variabel lokal agar cepat
        for i in range(self.n):  # untuk setiap sampel
            xi = X[i]  # baris fitur sampel ke-i
            c = self.bobot_hessian(dot(w, xi))  # bobot c_i (1 untuk linear, p(1-p) untuk logistik)
            for j in range(d):  # baris Hessian
                cj = c * xi[j]  # c_i * x_ij dihitung sekali per baris
                Hj = H[j]  # referensi baris ke-j
                for k in range(d):  # kolom Hessian
                    Hj[k] += cj * xi[k]  # tambahkan c_i x_ij x_ik
        inv = 1.0 / self.n  # 1/n untuk merata-ratakan
        for j in range(d):  # setiap baris
            for k in range(d):  # setiap kolom
                H[j][k] *= inv  # bagi dengan n
        return H  # kembalikan matriks Hessian


class RegresiLinear(Model):  # SK1 (dan versi besarnya): F(w) = (1/2n) sum (w.x_i - y_i)^2
    def residu(self, z, yi):  # turunan (1/2)(z - y)^2 terhadap z
        return z - yi  # residu biasa

    def bobot_hessian(self, z):  # turunan kedua (1/2)(z - y)^2 terhadap z
        return 1.0  # konstan, karena fungsinya kuadratik

    def loss(self, w):  # nilai fungsi galat F(w)
        s = 0.0  # akumulator jumlah kuadrat residu
        for xi, yi in zip(self.X, self.y):  # telusuri setiap sampel
            r = dot(w, xi) - yi  # residu prediksi
            s += r * r  # tambahkan kuadratnya
        return s / (2.0 * self.n)  # bagi dengan 2n


class RegresiLogistik(Model):  # RL: F(w) = (1/n) sum [ln(1 + e^z_i) - y_i z_i], z_i = w.x_i
    def residu(self, z, yi):  # turunan galat logistik terhadap z
        return sigmoid(z) - yi  # sigma(z) - y

    def bobot_hessian(self, z):  # turunan kedua galat logistik terhadap z
        p = sigmoid(z)  # peluang prediksi
        return p * (1.0 - p)  # p(1 - p)

    def loss(self, w):  # nilai fungsi galat F(w)
        s = 0.0  # akumulator
        for xi, yi in zip(self.X, self.y):  # telusuri setiap sampel
            z = dot(w, xi)  # skor linear
            s += softplus(z) - yi * z  # galat log-likelihood negatif yang stabil
        return s / self.n  # rata-rata


def sk1_kecil():  # SK1 tarif parkir: x (jam) dan y (ribu Rp), tepat pada y = 1 + 2x
    xs, ys = [1, 2, 3, 4], [3, 5, 7, 9]  # empat data contoh manual
    return RegresiLinear([[1.0, float(x)] for x in xs], [float(v) for v in ys])  # tambahkan kolom bias 1


def f2(x):  # SK2: fungsi dua lembah f(x) = x^4 - 2x^2
    return x ** 4 - 2.0 * x * x  # nilai fungsi


def df2(x):  # turunan pertama SK2
    return 4.0 * x ** 3 - 4.0 * x  # f'(x) = 4x^3 - 4x


def d2f2(x):  # turunan kedua SK2
    return 12.0 * x * x - 4.0  # f''(x) = 12x^2 - 4


def f3(x):  # SK3: F(x) = x^3/3 - 2x, minimum di x = akar 2 untuk x > 0
    return x ** 3 / 3.0 - 2.0 * x  # nilai fungsi


def df3(x):  # turunan pertama SK3
    return x * x - 2.0  # F'(x) = x^2 - 2, akarnya adalah akar 2


def d2f3(x):  # turunan kedua SK3; langkah Newton menjadi (x + 2/x)/2 (metode Babilonia)
    return 2.0 * x  # F''(x) = 2x


W_SK1 = [2.1, 4.4, -1.5]  # bobot "sebenarnya" SK1 diperbesar (d = 3)
W_LOG = [-0.5, 1.2, -0.8]  # bobot "sebenarnya" RL (d = 3)


def data_sk1(n, seed=0, w_true=W_SK1):  # data sintetis SK1: x_j ~ U(0, 5), y = w_true . x + derau N(0, 1)
    rng = LCG(1000 + seed)  # pembangkit acak khusus data (seed digeser 1000)
    d = len(w_true)  # dimensi
    X, y = [], []  # tampungan data
    for _ in range(n):  # buat n sampel
        xi = [1.0] + [rng.uniform(0.0, 5.0) for _ in range(d - 1)]  # bias 1 diikuti fitur acak
        X.append(xi)  # simpan baris fitur
        y.append(dot(w_true, xi) + rng.normal())  # target = model sebenarnya + derau
    return RegresiLinear(X, y)  # bungkus menjadi model regresi linear


def data_logistik(n, seed=0, d=3, w_true=None):  # data sintetis RL: x_j ~ U(-2, 2), y ~ Bernoulli(sigma(w_true . x))
    rng = LCG(2000 + seed)  # pembangkit acak khusus data (seed digeser 2000)
    if w_true is None:  # bobot sebenarnya tidak diberikan
        if d == len(W_LOG):  # untuk d = 3 pakai bobot tetap
            w_true = W_LOG  # bobot RL standar
        else:  # untuk d lain (eksperimen E2) bobot dibangkitkan acak
            w_true = [rng.uniform(-1.0, 1.0) / (d ** 0.5) * 2.0 for _ in range(d)]  # skala 2/akar(d) agar z tidak terlalu besar
    X, y = [], []  # tampungan data
    for _ in range(n):  # buat n sampel
        xi = [1.0] + [rng.uniform(-2.0, 2.0) for _ in range(d - 1)]  # bias 1 diikuti fitur acak
        X.append(xi)  # simpan baris fitur
        y.append(1.0 if rng.uniform() < sigmoid(dot(w_true, xi)) else 0.0)  # label acak sehingga data tidak terpisah sempurna
    return RegresiLogistik(X, y)  # bungkus menjadi model regresi logistik


# ============================================================ optimizer  # bagian 4
class SGD:  # w <- w - eta g (baris 7s), ruang tambahan 0 selain g
    name = "SGD"  # nama untuk keperluan cetak

    def __init__(self, d, eta=0.01):  # d tidak dipakai, hanya agar antarmuka sama dengan Adam
        self.eta = eta  # learning rate

    def step(self, w, g):  # satu langkah pembaruan (w diubah langsung)
        eta = self.eta  # variabel lokal agar cepat
        for j in range(len(w)):  # untuk setiap parameter
            w[j] -= eta * g[j]  # 2d operasi: kali dan kurang


class Adam:  # Kingma & Ba (2015), baris 7a-9a, ruang tambahan 2d (m dan v)
    name = "Adam"  # nama untuk keperluan cetak

    def __init__(self, d, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):  # parameter bawaan dari makalah asli
        self.eta, self.b1, self.b2, self.eps = eta, beta1, beta2, eps  # simpan parameter
        self.m = [0.0] * d  # momen pertama (rata-rata gradien), mulai nol
        self.v = [0.0] * d  # momen kedua (rata-rata kuadrat gradien), mulai nol
        self.t = 0  # penghitung langkah untuk koreksi bias

    def step(self, w, g):  # satu langkah pembaruan (w diubah langsung)
        self.t += 1  # naikkan nomor langkah
        b1, b2, eta, eps = self.b1, self.b2, self.eta, self.eps  # variabel lokal agar cepat
        c1 = 1.0 - b1 ** self.t  # faktor koreksi bias momen pertama
        c2 = 1.0 - b2 ** self.t  # faktor koreksi bias momen kedua
        m, v = self.m, self.v  # referensi ke list momen
        for j in range(len(w)):  # untuk setiap parameter
            m[j] = b1 * m[j] + (1.0 - b1) * g[j]  # 3d: perbarui momen pertama
            v[j] = b2 * v[j] + (1.0 - b2) * g[j] * g[j]  # 4d: perbarui momen kedua
            w[j] -= eta * (m[j] / c1) / ((v[j] / c2) ** 0.5 + eps)  # 7d: langkah dengan m-topi / (akar v-topi + eps)


class HessianSingular(Exception):  # kesalahan khusus bila Hessian tidak dapat dibalik
    pass  # tidak perlu isi tambahan


def eliminasi_gauss(A, b):  # menyelesaikan A x = b dengan eliminasi Gauss + pivot parsial (baris 9-15 Algoritma 2)
    d = len(A)  # ukuran sistem
    M = [list(A[i]) + [b[i]] for i in range(d)]  # matriks diperbesar [A | b], salinan sehingga A dan b tidak berubah
    for k in range(d):  # kolom pivot ke-k
        p = k  # baris 10: calon baris pivot
        best = abs(M[k][k])  # nilai mutlak pivot saat ini
        for i in range(k + 1, d):  # periksa baris-baris di bawahnya
            if abs(M[i][k]) > best:  # ada elemen yang lebih besar
                best, p = abs(M[i][k]), i  # jadikan calon pivot baru
        if p != k:  # pivot terbaik bukan di baris k
            M[k], M[p] = M[p], M[k]  # tukar baris
        if not best >= 1e-14:  # baris 11: pivot terlalu kecil (atau NaN)
            raise HessianSingular("Hessian singular")  # matriks dianggap singular
        Mk = M[k]  # baris pivot
        piv = Mk[k]  # nilai pivot
        for i in range(k + 1, d):  # baris 12-14: eliminasi di bawah diagonal
            Mi = M[i]  # baris yang akan dieliminasi
            f = Mi[k] / piv  # faktor pengali
            if f != 0.0:  # lewati bila sudah nol
                for j in range(k, d + 1):  # kolom k sampai kolom b
                    Mi[j] -= f * Mk[j]  # kurangi kelipatan baris pivot
    x = [0.0] * d  # baris 15: vektor solusi untuk substitusi balik
    for i in range(d - 1, -1, -1):  # dari baris terakhir ke baris pertama
        s = M[i][d]  # mulai dari ruas kanan b_i
        Mi = M[i]  # baris ke-i
        for j in range(i + 1, d):  # kolom di kanan diagonal
            s -= Mi[j] * x[j]  # kurangi kontribusi x_j yang sudah diketahui
        x[i] = s / Mi[i]  # bagi dengan elemen diagonal
    return x  # kembalikan solusi x


class NewtonRaphson:  # Algoritma 2: H(w_t) p_t = grad F(w_t), w_{t+1} = w_t - alpha p_t
    name = "Newton-Raphson"  # nama untuk keperluan cetak

    def __init__(self, alpha=1.0, tol=1e-10, T=500):  # alpha = 1 Newton murni, alpha < 1 Newton teredam
        self.alpha, self.tol, self.T = alpha, tol, T  # redaman, toleransi gradien, batas iterasi

    def step(self, model, w):  # satu iterasi Newton, mengembalikan (w baru, ||g|| sebelum langkah)
        g = model.grad(w)  # 4nd + d: gradien penuh
        H = model.hessian(w)  # 2nd^2 + O(nd): Hessian penuh
        p = eliminasi_gauss(H, g)  # 2d^3/3 + O(d^2): selesaikan H p = g
        a = self.alpha  # faktor redaman
        return [wj - a * pj for wj, pj in zip(w, p)], norm(g)  # 2d: w - alpha p

    def run(self, model, w0, callback=None):  # iterasi sampai ||g|| <= tol atau T iterasi
        w = list(w0)  # salin titik awal agar w0 tidak berubah
        for t in range(self.T + 1):  # paling banyak T langkah (+1 untuk pemeriksaan akhir)
            g = model.grad(w)  # gradien pada titik saat ini
            gn = norm(g)  # besar gradien
            if callback is not None:  # bila ada fungsi pemantau
                callback(t, w, gn)  # laporkan keadaan iterasi
            if not isfinite(gn) or gn > 1e100:  # gradien meledak
                return {"w": w, "iter": t, "ok": False, "alasan": "divergen"}  # gagal karena divergen
            if gn <= self.tol:  # baris 4: kriteria berhenti terpenuhi
                return {"w": w, "iter": t, "ok": True, "alasan": "konvergen"}  # berhasil
            if t == self.T:  # batas iterasi tercapai
                break  # keluar dari perulangan
            try:  # eliminasi Gauss bisa gagal
                p = eliminasi_gauss(model.hessian(w), g)  # arah Newton
            except HessianSingular:  # Hessian tidak dapat dibalik
                return {"w": w, "iter": t, "ok": False, "alasan": "Hessian singular"}  # gagal
            w = [wj - self.alpha * pj for wj, pj in zip(w, p)]  # langkah Newton (teredam)
            if not all(isfinite(v) and abs(v) < 1e100 for v in w):  # parameter meledak
                return {"w": w, "iter": t + 1, "ok": False, "alasan": "divergen"}  # gagal karena divergen
        return {"w": w, "iter": self.T, "ok": False, "alasan": "batas iterasi"}  # tidak konvergen dalam T iterasi


OPTS = {"SGD": SGD, "Adam": Adam}  # peta nama optimizer ke kelasnya


def make_opt(name, d, eta, **kw):  # membuat optimizer dari namanya
    return OPTS[name](d, eta=eta, **kw)  # panggil kelas yang sesuai


# ============================================== pelatihan mini-batch umum  # bagian 5
def train_epoch(model, opt, w, b, rng):  # satu epoch Algoritma 1 (baris 3-9)
    n = model.n  # banyak sampel
    idx = rng.permutation(n)  # Theta(n): acak urutan data
    for k in range(0, n, b):  # ceil(n/b) langkah
        g = model.grad_batch(w, idx[k:k + b])  # 4bd + d: gradien mini-batch
        opt.step(w, g)  # 2d (SGD) atau 14d (Adam): perbarui parameter


def train(model, name, eta, w0, b=32, epochs=20, seed=0, track=True):  # melatih dengan SGD/Adam
    import time  # pengecualian yang disetujui: modul time hanya untuk mengukur waktu
    rng = LCG(seed)  # pembangkit acak untuk permutasi
    w = list(w0)  # salin titik awal
    opt = make_opt(name, model.d, eta)  # buat optimizer
    hist = [model.loss(w)] if track else []  # riwayat loss, termasuk epoch 0
    times = [0.0]  # riwayat waktu kumulatif, epoch 0 = 0 detik
    elapsed = 0.0  # total waktu pelatihan
    for _ in range(epochs):  # ulangi sebanyak epoch
        t0 = time.perf_counter()  # catat waktu mulai
        train_epoch(model, opt, w, b, rng)  # satu epoch pelatihan
        elapsed += time.perf_counter() - t0  # tambahkan lama epoch (loss tidak ikut diukur)
        times.append(elapsed)  # simpan waktu kumulatif
        if track:  # bila riwayat loss diminta
            if not all(isfinite(v) and abs(v) < 1e100 for v in w):  # parameter meledak
                hist.extend([float("inf")] * (epochs + 1 - len(hist)))  # isi sisa riwayat dengan tak hingga
                break  # hentikan pelatihan
            hist.append(model.loss(w))  # loss penuh di luar pengukuran waktu
    return w, hist, times  # kembalikan parameter akhir, riwayat loss, dan riwayat waktu
