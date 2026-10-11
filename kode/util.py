# Utilitas kecil untuk skrip eksperimen: lokasi folder, statistik, simpan/baca JSON.  # tujuan modul
# Tidak memakai library apa pun: os dan json diganti dengan kode buatan sendiri.  # batasan modul
from optimizer import ln  # logaritma natural buatan sendiri (file kode kita sendiri, bukan library)


# ------------------------------------------------------------- lokasi folder  # bagian lokasi folder
def _folder_dari(path):  # mengambil nama folder dari sebuah path berkas (pengganti os.path.dirname)
    path = path.replace("\\", "/")  # samakan pemisah folder Windows "\" menjadi "/"
    posisi = path.rfind("/")  # cari pemisah "/" terakhir dalam path
    if posisi == -1:  # jika tidak ada pemisah, berkas berada di folder kerja saat ini
        return "."  # "." berarti folder kerja saat ini
    if posisi == 0:  # jika pemisah satu-satunya di awal (misalnya "/util.py")
        return "/"  # foldernya adalah akar sistem berkas
    return path[:posisi]  # selain itu, folder adalah semua teks sebelum pemisah terakhir


KODE = _folder_dari(__file__)  # folder tempat file kode ini berada (kode/)
ROOT = KODE + "/.."  # folder induknya (kode_eksperimen/), cukup naik satu tingkat
RESULTS = ROOT + "/results"  # folder hasil eksperimen (sudah ada di dalam arsip)
FIGS = ROOT + "/figs"  # folder gambar (sudah ada di dalam arsip)

SEEDS = range(5)  # setiap percobaan diulang 5 kali, seed 0-4


def gabung(folder, nama):  # menyusun path folder + nama berkas (pengganti os.path.join)
    return folder + "/" + nama  # cukup disambung dengan pemisah "/"


# ------------------------------------------------------------------ statistik  # bagian statistik
def mean(xs):  # rata-rata aritmetika
    xs = list(xs)  # ubah ke list agar bisa dihitung panjangnya (generator tidak bisa)
    return sum(xs) / len(xs)  # jumlah semua nilai dibagi banyaknya nilai


def std(xs):  # simpangan baku populasi
    xs = list(xs)  # ubah ke list agar bisa dipakai dua kali
    m = mean(xs)  # hitung rata-rata lebih dulu
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5  # akar dari rata-rata kuadrat selisih


def log10(x):  # logaritma basis 10
    return ln(x) / 2.302585092994046  # log10(x) = ln(x) / ln(10)


def slope_loglog(xs, ys):  # kemiringan garis kuadrat terkecil log10(y) terhadap log10(x)
    lx = [log10(x) for x in xs]  # transformasi log pada sumbu x
    ly = [log10(y) for y in ys]  # transformasi log pada sumbu y
    mx, my = mean(lx), mean(ly)  # rata-rata kedua sumbu
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))  # kovarians (tanpa dibagi n)
    den = sum((a - mx) ** 2 for a in lx)  # varians x (tanpa dibagi n)
    return num / den  # kemiringan = kovarians / varians


# ------------------------------------------------- penulis JSON buatan sendiri  # bagian penulis JSON
def _teks_json(s):  # mengubah string Python menjadi string JSON berkutip
    keluar = ['"']  # mulai dengan tanda kutip pembuka
    for ch in s:  # periksa setiap karakter
        if ch == '"':  # tanda kutip harus di-escape
            keluar.append('\\"')  # ditulis sebagai \"
        elif ch == "\\":  # garis miring terbalik harus di-escape
            keluar.append("\\\\")  # ditulis sebagai \\
        elif ch == "\n":  # baris baru
            keluar.append("\\n")  # ditulis sebagai \n
        elif ch == "\t":  # tab
            keluar.append("\\t")  # ditulis sebagai \t
        elif ch == "\r":  # carriage return
            keluar.append("\\r")  # ditulis sebagai \r
        elif ord(ch) < 32:  # karakter kontrol lain
            keluar.append("\\u%04x" % ord(ch))  # ditulis sebagai \uXXXX
        else:  # karakter biasa (termasuk huruf non-ASCII seperti η)
            keluar.append(ch)  # ditulis apa adanya
    keluar.append('"')  # tutup dengan tanda kutip
    return "".join(keluar)  # gabungkan semua potongan menjadi satu string


def _angka_json(x):  # mengubah float menjadi teks angka JSON
    if x != x:  # NaN adalah satu-satunya nilai yang tidak sama dengan dirinya sendiri
        return "NaN"  # ditulis sama seperti modul json standar
    if x == float("inf"):  # tak hingga positif
        return "Infinity"  # ditulis sama seperti modul json standar
    if x == float("-inf"):  # tak hingga negatif
        return "-Infinity"  # ditulis sama seperti modul json standar
    return repr(x)  # repr float Python sudah berupa teks terpendek yang tepat


def ke_json(data, indent=1, level=0):  # mengubah data Python menjadi teks JSON (pengganti json.dumps)
    spasi = " " * (indent * (level + 1))  # indentasi untuk isi pada level ini
    spasi_tutup = " " * (indent * level)  # indentasi untuk kurung penutup
    if data is None:  # None Python
        return "null"  # menjadi null di JSON
    if data is True:  # True Python (dicek sebelum int karena bool turunan int)
        return "true"  # menjadi true di JSON
    if data is False:  # False Python
        return "false"  # menjadi false di JSON
    if isinstance(data, int):  # bilangan bulat
        return str(data)  # ditulis apa adanya
    if isinstance(data, float):  # bilangan pecahan
        return _angka_json(data)  # pakai fungsi khusus agar inf/NaN tertangani
    if isinstance(data, str):  # string
        return _teks_json(data)  # pakai fungsi escape di atas
    if isinstance(data, dict):  # dictionary menjadi objek JSON
        if not data:  # dictionary kosong
            return "{}"  # objek kosong
        baris = []  # tampung setiap pasangan kunci-nilai
        for k, v in data.items():  # telusuri setiap pasangan
            baris.append(spasi + _teks_json(str(k)) + ": " + ke_json(v, indent, level + 1))  # kunci selalu string
        return "{\n" + ",\n".join(baris) + "\n" + spasi_tutup + "}"  # susun objek dengan baris baru
    if isinstance(data, (list, tuple, range)):  # list/tuple/range menjadi array JSON
        data = list(data)  # samakan menjadi list
        if not data:  # list kosong
            return "[]"  # array kosong
        baris = [spasi + ke_json(v, indent, level + 1) for v in data]  # ubah setiap elemen
        return "[\n" + ",\n".join(baris) + "\n" + spasi_tutup + "]"  # susun array dengan baris baru
    raise TypeError("tipe tidak didukung JSON: " + str(type(data)))  # tipe lain tidak didukung


# ------------------------------------------------- pembaca JSON buatan sendiri  # bagian pembaca JSON
class _PembacaJSON:  # parser recursive descent sederhana (pengganti json.loads)
    def __init__(self, teks):  # menyimpan teks yang akan dibaca
        self.s = teks  # seluruh teks JSON
        self.i = 0  # posisi karakter yang sedang dibaca

    def lewati_spasi(self):  # melompati spasi, tab, dan baris baru
        while self.i < len(self.s) and self.s[self.i] in " \t\n\r":  # selama karakter berupa spasi
            self.i += 1  # maju satu karakter

    def nilai(self):  # membaca satu nilai JSON apa pun
        self.lewati_spasi()  # abaikan spasi di depan nilai
        ch = self.s[self.i]  # lihat karakter pertama untuk menentukan jenis nilai
        if ch == "{":  # awal objek
            return self.objek()  # baca objek
        if ch == "[":  # awal array
            return self.array()  # baca array
        if ch == '"':  # awal string
            return self.string()  # baca string
        for kata, hasil in (("true", True), ("false", False), ("null", None),  # kata kunci JSON
                            ("NaN", float("nan")), ("Infinity", float("inf")),  # nilai khusus float
                            ("-Infinity", float("-inf"))):  # tak hingga negatif
            if self.s.startswith(kata, self.i):  # cocokkan kata kunci pada posisi ini
                self.i += len(kata)  # lompati kata kunci
                return hasil  # kembalikan nilai Python-nya
        return self.angka()  # selain itu pasti angka

    def objek(self):  # membaca objek {"kunci": nilai, ...}
        hasil = {}  # dictionary hasil
        self.i += 1  # lompati "{"
        self.lewati_spasi()  # abaikan spasi
        if self.s[self.i] == "}":  # objek kosong
            self.i += 1  # lompati "}"
            return hasil  # kembalikan dictionary kosong
        while True:  # baca pasangan satu per satu
            self.lewati_spasi()  # abaikan spasi sebelum kunci
            kunci = self.string()  # kunci selalu berupa string
            self.lewati_spasi()  # abaikan spasi sebelum ":"
            self.i += 1  # lompati ":"
            hasil[kunci] = self.nilai()  # baca nilainya dan simpan
            self.lewati_spasi()  # abaikan spasi sesudah nilai
            if self.s[self.i] == ",":  # masih ada pasangan berikutnya
                self.i += 1  # lompati ","
            else:  # karakter berikutnya pasti "}"
                self.i += 1  # lompati "}"
                return hasil  # objek selesai dibaca

    def array(self):  # membaca array [nilai, ...]
        hasil = []  # list hasil
        self.i += 1  # lompati "["
        self.lewati_spasi()  # abaikan spasi
        if self.s[self.i] == "]":  # array kosong
            self.i += 1  # lompati "]"
            return hasil  # kembalikan list kosong
        while True:  # baca elemen satu per satu
            hasil.append(self.nilai())  # baca satu elemen
            self.lewati_spasi()  # abaikan spasi sesudah elemen
            if self.s[self.i] == ",":  # masih ada elemen berikutnya
                self.i += 1  # lompati ","
            else:  # karakter berikutnya pasti "]"
                self.i += 1  # lompati "]"
                return hasil  # array selesai dibaca

    def string(self):  # membaca string "..."
        self.i += 1  # lompati kutip pembuka
        keluar = []  # tampung karakter hasil
        while True:  # baca sampai kutip penutup
            ch = self.s[self.i]  # karakter saat ini
            if ch == '"':  # kutip penutup
                self.i += 1  # lompati kutip penutup
                return "".join(keluar)  # gabungkan karakter menjadi string
            if ch == "\\":  # awal escape
                kode = self.s[self.i + 1]  # huruf sesudah garis miring terbalik
                if kode == "u":  # escape \uXXXX
                    keluar.append(chr(int(self.s[self.i + 2:self.i + 6], 16)))  # 4 digit heksadesimal
                    self.i += 6  # lompati \uXXXX
                    continue  # lanjut ke karakter berikutnya
                peta = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\", '"': '"'}  # escape satu huruf
                keluar.append(peta[kode])  # terjemahkan escape
                self.i += 2  # lompati dua karakter escape
                continue  # lanjut ke karakter berikutnya
            keluar.append(ch)  # karakter biasa langsung disimpan
            self.i += 1  # maju satu karakter

    def angka(self):  # membaca angka bulat atau pecahan
        awal = self.i  # catat posisi awal angka
        while self.i < len(self.s) and self.s[self.i] in "+-0123456789.eE":  # karakter yang boleh ada pada angka
            self.i += 1  # maju selama masih bagian angka
        teks = self.s[awal:self.i]  # potongan teks angka
        if "." in teks or "e" in teks or "E" in teks:  # ada titik atau eksponen berarti float
            return float(teks)  # ubah menjadi float (fungsi bawaan Python)
        return int(teks)  # selain itu bilangan bulat


def dari_json(teks):  # mengubah teks JSON menjadi data Python (pengganti json.loads)
    return _PembacaJSON(teks).nilai()  # buat parser lalu baca nilai teratas


# --------------------------------------------------------- simpan dan muat  # bagian berkas hasil
def simpan(nama, data):  # menyimpan data ke results/<nama> dalam format JSON
    path = gabung(RESULTS, nama)  # path lengkap berkas tujuan
    try:  # open() bisa gagal bila folder results/ belum ada
        with open(path, "w", encoding="utf-8") as f:  # buka berkas untuk ditulis (open adalah fungsi bawaan)
            f.write(ke_json(data))  # tulis teks JSON buatan sendiri
    except FileNotFoundError:  # folder tujuan tidak ditemukan
        print("Folder results/ belum ada, buat dulu foldernya:", RESULTS)  # tanpa os, folder dibuat manual
        raise  # hentikan program dengan pesan kesalahan aslinya
    print("tersimpan:", path)  # beri tahu lokasi berkas


def muat(nama):  # membaca results/<nama> yang berformat JSON
    with open(gabung(RESULTS, nama), encoding="utf-8") as f:  # buka berkas untuk dibaca
        return dari_json(f.read())  # baca seluruh isi lalu ubah menjadi data Python
