# DNN_PSO: Deep Neural Network Training via Particle Swarm Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PySwarms](https://img.shields.io/badge/pyswarms-1.3.0-orange.svg)](https://github.com/ljvmiranda921/pyswarms)
[![Tests](https://img.shields.io/badge/tests-16_passing-brightgreen.svg)](#-testler-ve-ölçümler)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**DNN_PSO**, yapay sinir ağlarının ağırlıklarını klasik geriye yayılım (backpropagation / gradient descent) **kullanmadan**, doğadan esinlenen **Parçacık Sürü Optimizasyonu (Particle Swarm Optimization - PSO)** ile eğiten meta-sezgisel bir yapay zeka algoritmasıdır. Ağın mimarisini de tek bir sayıdan (`transition_per`) kendisi kurar.

| | |
| :--- | :--- |
| 🧠 **Orijinal algoritma** | `class_prodnn.py` (`prodnnv10` sınıfı) ve `dnn+pso.py` - İlhan Koçaslan, 2022. **Değiştirilmez**; bir test bu dosyaların parmak izini denetler. |
| ⚡ **`dnnpso` katmanı** | Aynı modeli NumPy ile hesaplayan hızlı motor (orijinalle **birebir aynı** sonuç), `fit` / `predict` / `save` arayüzü, komut satırı, kara kutu eğitimi. |
| 📈 **Ölçülen** | 1 000 örnekte orijinalden **1 881 kat hızlı**, **103 kat az bellek**; aynı tohumla aynı eğitim sonucu. |

---

## 📑 İçindekiler

- [Neden gradyan inişi yerine PSO?](#-neden-gradyan-i̇nişi-yerine-pso)
- [Özgün kullanım alanları](#-özgün-kullanım-alanları)
- [Hızlı başlangıç](#-hızlı-başlangıç)
- [Ölçülen sonuçlar](#-ölçülen-sonuçlar)
- [Sınırlar: ne zaman PSO değil?](#%EF%B8%8F-sınırlar-ne-zaman-pso-değil)
- [Matematiksel altyapı](#-matematiksel-ve-algoritmik-altyapı)
- [Dinamik katman geçişi (`transition_per`)](#%EF%B8%8F-dinamik-katman-geçişi-transition_per)
- [Orijinal kodun davranışları ve `dnnpso` karşılıkları](#-orijinal-kodun-davranışları-ve-dnnpso-karşılıkları)
- [`dnnpso` kılavuzu](#-dnnpso-kılavuzu)
- [`prodnnv10` sınıf kılavuzu](#-prodnnv10-sınıf-kılavuzu)
- [Proje dosya yapısı](#-proje-dosya-yapısı)
- [Testler ve ölçümler](#-testler-ve-ölçümler)
- [Kaynaklar](#-kaynaklar)

---

## 📌 Neden Gradyan İnişi Yerine PSO?

Geleneksel derin öğrenme modelleri ağırlık güncellemelerini geriye yayılım (Backpropagation) ve gradyan inişi (Gradient Descent) ile gerçekleştirir. Ancak bu yaklaşımın bazı temel sınırları vardır:

1. **Türev Zorunluluğu:** Aktivasyon ve maliyet fonksiyonlarının kesinlikle sürekli ve türevlenebilir olması gerekir.
2. **Yerel Minimumlar (Local Minima):** Gradyan inişi yerel minimumlara, eyer noktalarına (saddle points) veya sıfır gradyan platolarına takılabilir.
3. **Gradyan Kaybolması / Patlaması:** Çok katmanlı derin ağlarda gradyanlar kaybolabilir (vanishing gradient) veya kararsızlaşabilir (exploding gradient).

### 🚀 PSO'nun Sağladığı Avantajlar:
- **Türev Gerektirmez (Derivative-Free):** Sadece ileri besleme (forward propagation) ile hesaplanan hata değerini kullanır. Hata **herhangi bir sayı** olabilir: doğruluk, kâr, bir simülasyonun sonucu.
- **Küresel Arama:** Sürüdeki parçacıklar arama uzayının farklı noktalarını aynı anda tarar; tek bir noktadan inen gradyan yöntemlerine göre yerel minimumlara daha az takılır.
- **Dinamik Ağ Yapısı:** Katman boyutları ve nöron sayıları `transition_per` parametresi ile otomatik ve orantılı olarak şekillendirilir.

---

## 🌍 Özgün Kullanım Alanları

PSO ile eğitim, büyük derin ağlarda geri yayılımın yerini tutmaz (bkz. [Sınırlar](#%EF%B8%8F-sınırlar-ne-zaman-pso-değil)). Gerçek gücü, **geri yayılımın hiç çalışamadığı** yerlerdedir. Her alan için depoda çalışan bir örnek vardır:

| Alan | Neden PSO? | Örnek | Literatür |
| :--- | :--- | :--- | :--- |
| **Türevi olmayan nöronlar** (0/1 "basamak" nöronlar, mikrodenetleyici / FPGA / mantık devresi) | Basamak fonksiyonunun türevi her yerde 0'dır; geri yayılım hiç öğrenemez, PSO öğrenir | [`examples/03_basamak_noronlar.py`](examples/03_basamak_noronlar.py): test doğruluğu **%97.5** | |
| **Türevi olmayan hedefler** (doğruluk, F1, işlem maliyetli kâr, Sharpe oranı) | Asıl ölçütü doğrudan en iyiler; vekil bir kayıp (MSE) gerekmez | [`examples/05_sharpe_hedefi.py`](examples/05_sharpe_hedefi.py): 5 piyasanın **4'ünde** daha yüksek Sharpe, yarı işlem hacmi | PSO ile Sharpe oranına göre alım-satım kuralı eniyilemesi [7] |
| **Kara kutu simülasyon / kontrol / pekiştirmeli öğrenme** | Etiket yok, model denklemi yok; yalnızca "bu ağ ne kadar iyi yönetti?" sayısı var | [`examples/04_su_tanki_kontrolcu.py`](examples/04_su_tanki_kontrolcu.py): ayarlı bir PI düzeyinde kontrolcü, 5.5 sn'de | Türevsiz popülasyon yöntemleri derin RL'de rekabetçi [4][5] |
| **Küçük ağ, az veri** | Az sayıda ağırlıkta PSO hızlı yakınsar | [`examples/01_orijinal_ve_hizli_motor.py`](examples/01_orijinal_ve_hizli_motor.py) | Küçük ağlarda PSO, geri yayılımdan hızlı yakınsadı [2] |
| **PSO + geri yayılım (hibrit)** | PSO iyi bir başlangıç bulur, gradyan yöntemi ince ayar yapar | `to_prodnn_weights()` ile ağırlıklar dışa aktarılır | PSO-BP hibrit eğitimi [3] |
| **Otomatik mimari** | Tek bir oranla giderek daralan katmanlar | `transition_per` (orijinal fikir) | Masters'ın "geometrik piramit kuralı"nın [6] genelleştirilmiş bir biçimi |

---

## 🚀 Hızlı Başlangıç

```bash
pip install -r requirements.txt
```

**1. Orijinal hızlı başlangıç** (orijinal `prodnnv10` sınıfıyla, Windows'ta `run.bat`):

```bash
python example.py
```

**2. `dnnpso` ile kendi verinizde:**

```python
from dnnpso import PSONetwork, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
net = PSONetwork(transition_per=2/3, particles=30, iterations=200, restarts=3, seed=1)
net.fit(X_train, y_train)            # 0/1, sınıf etiketleri ya da sayısal hedef: kendisi anlar
print(net.summary())                 # mimari, ağırlık sayısı, maliyet, süre
print(net.score(X_test, y_test))     # doğruluk (sınıflandırma) / R² (regresyon)
net.save("model.json")               # sonra: PSONetwork.load("model.json").predict(X_yeni)
```

**3. Komut satırından** (JSON ya da CSV):

```bash
python -m dnnpso train   --csv veri.csv --target etiket --test-size 0.25 --out model.json --plot egri.png
python -m dnnpso predict --model model.json --csv yeni.csv --out tahminler.csv
python -m dnnpso info    --model model.json
```

**4. Kara kutu: maliyet veren herhangi bir fonksiyonla eğitim:**

```python
from dnnpso import fit_blackbox

def degerlendir(politika):           # politika(girdiler) -> ağ çıktısı
    ...bir simülasyon ya da geriye dönük test çalıştır...
    return maliyet                   # küçük olan iyidir

ag, maliyet, gecmis = fit_blackbox(degerlendir, inputs=3, outputs=1)
```

**5. Örnekler:**

```bash
python examples/01_orijinal_ve_hizli_motor.py   # orijinal sınıf ve hızlı motor: aynı sonuç
python examples/02_xor_bias_sutunu.py           # XOR neden öğrenilemiyor, nasıl öğrenilir
python examples/03_basamak_noronlar.py          # türevi olmayan 0/1 nöronlar
python examples/04_su_tanki_kontrolcu.py        # yalnızca simülasyonla eğitilen kontrolcü
python examples/05_sharpe_hedefi.py             # hedef doğrudan Sharpe oranı (sentetik veri)
```

---

## 📈 Ölçülen Sonuçlar

Tüm sayılar bu depodaki betiklerle üretildi (Python 3.11, NumPy 2.3, pyswarms 1.3).

**Eşdeğerlik:** hızlı motor, 6 farklı mimaride `prodnnv10` ile aynı çıktıyı ve maliyeti verir (fark < 10⁻¹²). Aynı tohumla eğitildiğinde iki motor **aynı** sonuca ulaşır; hızlı motorun ağırlıkları orijinal sınıfa yüklendiğinde orijinal `propagate_forward` aynı tahminleri üretir.

**Hız ve bellek** ([`benchmarks/speed_memory.py`](benchmarks/speed_memory.py), 1 000 örnek, 8 özellik, 20 parçacık × 20 iterasyon):

| | Süre | En yüksek bellek | Doğruluk |
| :--- | ---: | ---: | ---: |
| Orijinal `prodnnv10` | 362.8 sn | 357 MB | 0.566 |
| `dnnpso` hızlı motor | **0.19 sn** | **3.5 MB** | 0.566 (aynı) |
| `dnnpso`, varsayılan bütçe (30 × 200 × 3) | 7.1 sn | | **0.945** |

**Varsayılan ayarlar ölçülerek seçildi** ([`benchmarks/defaults_study.py`](benchmarks/defaults_study.py), 6 problem × 3 tohum, bölünmüş verilerde test skoru):

| Problem | Orijinal ayar | **Bias sütunu (varsayılan)** | Bias sütunu + ağırlık sınırı ±10 |
| :--- | ---: | ---: | ---: |
| 3-bit (depodaki veri) | 0.958 | **1.000** | 0.875 |
| XOR | 0.333 | **0.917** | 0.833 |
| 4-bit parite | **0.875** | 0.833 | 0.646 |
| İki hilal (300) | 0.893 | 0.919 | **0.922** |
| Halkalar (300) | 0.511 | **0.789** | 0.748 |
| Sinüs regresyonu (R²) | 0.122 | **0.720** | 0.720 |
| **Ortalama** | 0.615 | **0.863** | 0.791 |

Girişe sabit 1 değerli bir sütun eklemek ortalamayı **0.62 → 0.86** çıkardı; ağırlık sınırları ise kötüleştirdi, bu yüzden varsayılan olarak kapalı.

**Örnekler:** XOR orijinal ayarla 10 denemenin 0'ında, bias sütunuyla 8'inde çözüldü · basamak nöronlu ağ test doğruluğu %97.5 · simülasyonla eğitilen tank kontrolcüsü görülmemiş hedeflerde ayarlı PI ile aynı hata (0.078 m) · Sharpe ile eğitilen ağ 5 sentetik piyasanın 4'ünde MSE ile eğitilenden yüksek Sharpe.

---

## ⚠️ Sınırlar: ne zaman PSO değil?

- **Çok sayıda ağırlık:** PSO'nun arama uzayı ağırlık sayısıyla büyür. Literatür, PSO'nun yüksek boyutlu ağlarda iyi ölçeklenmediğini gösteriyor [8]. Binlerce ağırlıktan büyük ağlarda (görüntü, dil modelleri) geri yayılım kullanın. Bu projenin tatlı noktası onlarca ile birkaç yüz ağırlıktır.
- **Sigmoid doygunluğu:** PSO ile eğitilen sigmoid ağlarda nöronlar sık sık 0 ya da 1'e yapışır [8][9]. `dnnpso` girişleri [0, 1]'e ölçekler (`scale="minmax"`).
- **Piramit kuralı:** gizli katmanlar her zaman girişten küçüktür. 2-3 girişli problemlerde ağ çok küçük kalır; `hidden_layers=[6]` gibi elle belirtilebilir.
- **Basamak hedefler:** doğruluk gibi hedefler geniş düzlükler oluşturur. `objective="mse+accuracy"` hem yön hem hedef verir.
- **Metriği doğrudan en iyilemek istenmeyen çözümlere izin verebilir:** örnek 5'te sürekli pozisyonla PSO, Sharpe oranının pozisyon büyüklüğünden bağımsız olmasını "keşfetti" ve neredeyse hiç işlem yapmayan stratejiler buldu. Hedefi bu kaçış yollarını kapatacak biçimde tanımlayın.

---

## 🧠 Matematiksel ve Algoritmik Altyapı

### 1. Parçacık Temsili
Ağın tüm katmanlarındaki ağırlık matrisleri $D$ boyutlu tek bir vektör olarak düzleştirilir (flatten):
$$\mathbf{X}_i = [w_1, w_2, \dots, w_D]$$
Burada her $\mathbf{X}_i$ parçacığı, yapay sinir ağının eksiksiz bir aday ağırlık çözümüdür.

### 2. İleri Besleme ve Maliyet Fonksiyonu
Her parçacığın ağırlıkları ağa yüklenir ve eğitim verisi ileri yayılır:
$$z = \mathbf{x} \cdot \mathbf{w}^T, \quad a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Tahmin edilen çıktılar ile gerçek etiketler arasındaki Ortalama Kare Hata (MSE) hesaplanır:
$$\text{Cost} = 100 \times \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### 3. PSO Hız ve Konum Güncellemesi
Sürüdeki her parçacık, kişisel en iyi deneyimi ($\mathbf{p}_i$) ve sürünün küresel en iyisi ($\mathbf{g}$) doğrultusunda güncellenir [1]:
$$\mathbf{v}_i^{(t+1)} = w \cdot \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^{(t)})$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
- $w = 0.9$: Eylemsizlik ağırlığı (Inertia weight)
- $c_1 = 1.0$: Bilişsel öğrenme katsayısı (Cognitive parameter)
- $c_2 = 0.8$: Sosyal öğrenme katsayısı (Social parameter)
- $r_1, r_2 \sim U(0, 1)$: Rastgele katsayılar

`dnnpso` aynı katsayıları kullanır; tek fark, sürünün tamamının maliyetini tek bir matris işlemiyle hesaplamasıdır.

---

## 🏗️ Dinamik Katman Geçişi (`transition_per`)

Model, giriş boyutu ile çıkış boyutu arasında katmanları elle belirleme zorunluluğunu ortadan kaldırır. `transition_per` parametresi ile katmanlar kademeli olarak daralarak inşa edilir:

```
input_neuron_numbers (örnek: 3)
     │
     ▼  (3 × 2/3 = 2 nöron)
Hidden Layer: 2 nöron
     │
     ▼  (2 × 2/3 = 1 -> çıkış nöronuna ulaşıldı, döngü sonlanır)
Output Layer: 1 nöron
```

Sonuç Mimarisi: **`[3, 2, 1]`** (Toplam 8 ağırlık boyutu).

| Giriş → çıkış | `transition_per` | Mimari | Ağırlık |
| :--- | :---: | :--- | ---: |
| 3 → 1 | 2/3 | `[3, 2, 1]` | 8 |
| 8 → 1 | 2/3 | `[8, 5, 3, 2, 1]` | 63 |
| 16 → 1 | 0.5 | `[16, 8, 4, 2, 1]` | 170 |
| 10 → 3 | 0.8 | `[10, 8, 6, 4, 3]` | 164 |
| 2 → 1 | herhangi | `[2, 1]` (gizli katman yok) | 2 |

---

## 🔍 Orijinal Kodun Davranışları ve `dnnpso` Karşılıkları

Aşağıdakiler orijinal kodda ölçülerek bulundu. Orijinal dosyalar değiştirilmedi; her biri `dnnpso` tarafında karşılanır.

| Orijinalde | Ölçülen etkisi | `dnnpso`'da |
| :--- | :--- | :--- |
| Nöronlarda bias yok | Sıfır girişte ilk katman hep 0.5 verir; 2 girişli problemlerde gizli katman oluşmaz. Depodaki `xor_x.json` / `xor_y.json` **öğrenilemez** (çıktı her girişte 0.5) | `bias="input"` (varsayılan): sabit 1 sütunu. `bias="neuron"`: her nörona bias |
| `transition_per >= 1` | Mimari döngüsü **hiç bitmez** | Açık bir hata mesajı |
| `output_neuron_numbers > 1` | Maliyete ve çıktıya yalnızca **ilk** çıkış nöronu girer | Çok sınıflı hedefler: her sınıfa bir çıkış nöronu |
| Veri yalnızca JSON dosya yolundan | Yeni veride tahmin yapılamaz | `predict(X)`, dizi / liste / CSV |
| Her değerlendirmede tüm ara çıktılar saklanır (`full_outputstotal`, `costs`) | 1 000 örnekte 20 × 20'lik eğitim 357 MB bellek, 6 dakika | Hiçbir şey birikmez: 3.5 MB, 0.19 sn |
| Girişler ölçeklenmez | Büyük değerlerde sigmoid doyar | `scale="minmax"` (varsayılan) |
| Ağırlık kaydı / yükleme yok | | `save()` / `load()`, `to_prodnn_weights()` |

`PSONetwork.original()` orijinal davranışı birebir seçer (bias yok, ölçekleme yok, sınırsız ağırlık, MSE). `engine="original"` ise eğitimi doğrudan orijinal `prodnnv10` sınıfıyla yapar.

---

## 📖 `dnnpso` Kılavuzu

### `PSONetwork`

| Parametre | Varsayılan | Açıklama |
| :--- | :---: | :--- |
| `transition_per` | `2/3` | Katman daralma oranı (0-1), orijinal kural |
| `hidden_layers` | `None` | Gizli katmanları elle vermek için, ör. `[8, 4]` (None = piramit kuralı) |
| `particles` / `iterations` / `restarts` | `30` / `200` / `3` | Parçacık sayısı, PSO adımı, farklı başlangıçtan tekrar sayısı |
| `options` | `{"c1": 1, "c2": 0.8, "w": 0.9}` | PSO katsayıları (orijinaldeki değerler) |
| `bias` | `"input"` | `"input"`: sabit 1 sütunu, `"neuron"`: nöron başına bias, `"none"`: orijinal |
| `activation` | `"sigmoid"` | `"sigmoid"`, `"step"` (0/1), `"tanh"`, `"relu"` |
| `scale` | `"minmax"` | Girişleri [0, 1]'e ölçekler; `None` = ölçeklemez |
| `objective` | `"mse"` | `"mse"` (orijinal, 100 × MSE), `"accuracy"`, `"f1"`, `"mse+accuracy"` ya da `f(y, çıktı) -> maliyet` |
| `weight_bounds` | `None` | Ağırlıkları sınırlar, ör. `(-10, 10)` |
| `tol` / `tol_iter` | `None` / `20` | Erken durdurma (pyswarms `ftol`) |
| `task` | `"auto"` | `"binary"`, `"multiclass"`, `"regression"`; otomatik algılanır |
| `engine` | `"fast"` | `"original"`: eğitim orijinal `prodnnv10` ile |
| `seed` | `None` | Tekrarlanabilir sonuç |

Yöntemler: `fit(X, y)`, `predict(X)`, `predict_raw(X)`, `score(X, y)`, `summary()`, `plot_history()`, `save(yol)`, `PSONetwork.load(yol)`, `to_prodnn_weights()`, `PSONetwork.original(...)`.

### `fit_blackbox(evaluate, ...)`

`evaluate(politika)` bir maliyet döndüren herhangi bir fonksiyondur. `inputs` / `outputs` / `transition_per` ile piramit mimarisi ya da `layers=[3, 4, 1]` ile açık mimari verilir. `batch=True` ise `politika` her parçacık için bir satır alır ve `evaluate` her parçacığın maliyetini döndürür: NumPy ile yazılmış bir simülasyon tüm sürüyü aynı anda yürütür (örnek 4'te eğitim 27 sn → 5.5 sn, üstelik 6.7 kat büyük bütçeyle).

### `FastProdnn`

Motorun kendisi: `FastProdnn.pyramid(3, 1, 2/3)`, `FastProdnn.from_prodnn(model)`, `forward(X)`, `swarm_cost(X, y, konumlar)`, `to_prodnn_weights()`.

---

## 📖 `prodnnv10` Sınıf Kılavuzu

Kendi kodunuzda `class_prodnn.py` dosyasını değiştirmeden içe aktararak kullanabilirsiniz:

```python
from class_prodnn import prodnnv10

# Modeli tanımla
model = prodnnv10(
    input_neuron_numbers=3,      # Giriş özellik sayısı
    output_neuron_numbers=1,     # Çıkış hedef sayısı
    transition_per=2/3,          # Katman daralma oranı (örn: 0.666)
    x_input="data/data_x.json",  # Giriş verisi JSON dosya yolu
    y_input="data/data_y.json",  # Çıkış verisi JSON dosya yolu
    train_size=100,              # Her döngüdeki PSO iterasyon sayısı
    particle=20,                 # Sürüdeki parçacık sayısı
    loop_size=3                  # Global optimum için PSO döngü sayısı
)

# Eğitimi başlat
cost_history = model.trainer()

# En iyi ağırlıkları ağa yükle
model.bestweight_upload()

# İleri yayılım yaparak tahminleri al
model.propagate_forward()
predictions = model.full_outputs_lastmend

# Eğitim maliyet eğrisini çizdir
model.cost_effect()
```

| Parametre | Tip | Açıklama |
| :--- | :---: | :--- |
| `input_neuron_numbers` | `int` | Giriş katmanındaki nöron / özellik sayısı |
| `output_neuron_numbers` | `int` | Çıkış katmanındaki nöron sayısı |
| `transition_per` | `float` | Katmanlar arası nöron geçiş / daralma oranı ($0 < r < 1$) |
| `x_input` | `str` | Giriş verilerini içeren `.json` dosyasının yolu |
| `y_input` | `str` | Hedef etiketleri içeren `.json` dosyasının yolu |
| `train_size` | `int` | Bir PSO koşusundaki iterasyon / adım sayısı |
| `particle` | `int` | Sürüdeki parçacık sayısı (popülasyon boyutu) |
| `loop_size` | `int` | Farklı başlangıç noktalarından tekrarlanacak PSO koşu sayısı |

Veri biçimi (JSON): giriş `[[0.5, 1.2, -0.3], [1.0, 0.2, 0.8]]`, çıkış `[1, 0]`.

---

## 📁 Proje Dosya Yapısı

```
DNN_PSO/
├── class_prodnn.py                 # prodnnv10 sınıfı (orijinal algoritma, değiştirilmez)
├── dnn+pso.py                      # Prototip PSO betiği (orijinal geliştirme kodu, değiştirilmez)
├── example.py / run.bat            # Orijinal sınıfla hızlı başlangıç
├── dnnpso/                         # Kullanıma hazır katman
│   ├── engine.py                   #   FastProdnn: aynı model, NumPy ile, tüm sürü tek seferde
│   ├── model.py                    #   PSONetwork: fit / predict / score / save / load
│   ├── blackbox.py                 #   fit_blackbox: simülasyon, kâr, herhangi bir maliyet
│   ├── objectives.py               #   mse, doğruluk, F1, mse+doğruluk
│   ├── original.py                 #   Eğitimi orijinal prodnnv10 ile yaptırma
│   └── cli.py                      #   python -m dnnpso train / predict / info
├── examples/                       # 5 çalışan örnek (yukarıda)
├── benchmarks/                     # hız-bellek ve varsayılan ayar ölçümleri
├── tests/test_dnnpso.py            # eşdeğerlik, API, komut satırı, orijinal dosya koruması
├── data/                           # data_x/y.json (3 giriş), xor_x/y.json (2 giriş)
├── particle-swarm-optimization.pdf # PSO teorik makalesi
└── requirements.txt
```

---

## 🧪 Testler ve Ölçümler

```bash
python -m unittest discover -s tests -v       # 16 test, ~2 sn
python benchmarks/speed_memory.py             # orijinal vs hızlı motor (~6 dk; --skip-original ile saniyeler)
python benchmarks/defaults_study.py           # varsayılan ayar çalışması (~30 sn)
```

`test_original_files_are_unchanged`, `class_prodnn.py` ve `dnn+pso.py` dosyalarının SHA-256 parmak izini denetler; orijinal kod yanlışlıkla değişirse test başarısız olur.

---

## 📚 Kaynaklar

1. J. Kennedy, R. Eberhart, "Particle swarm optimization", *Proc. ICNN'95*, vol. 4, pp. 1942–1948, 1995. [IEEE](https://ieeexplore.ieee.org/document/488968)
2. V. G. Gudise, G. K. Venayagamoorthy, "Comparison of particle swarm optimization and backpropagation as training algorithms for neural networks", *Proc. IEEE Swarm Intelligence Symposium*, pp. 110–117, 2003.
3. J.-R. Zhang, J. Zhang, T.-M. Lok, M. R. Lyu, "A hybrid particle swarm optimization–back-propagation algorithm for feedforward neural network training", *Applied Mathematics and Computation* 185(2), pp. 1026–1037, 2007. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0096300306008277)
4. T. Salimans, J. Ho, X. Chen, I. Sutskever, "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", 2017. [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)
5. F. P. Such, V. Madhavan, E. Conti, J. Lehman, K. O. Stanley, J. Clune, "Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning", 2017. [arXiv:1712.06567](https://arxiv.org/abs/1712.06567)
6. T. Masters, *Practical Neural Network Recipes in C++*, Academic Press, 1993 (geometrik piramit kuralı).
7. "PSO for the Sharpe Ratio in a Financial Trading System Based on Technical Analysis", Springer. [Springer](https://link.springer.com/chapter/10.1007/978-3-031-64273-9_16)
8. "An Analysis of Activation Function Saturation in Particle Swarm Optimization Trained Neural Networks", *Neural Processing Letters*, 2020. [Springer](https://link.springer.com/article/10.1007/s11063-020-10290-z)
9. A. Rakitianskaia, A. Engelbrecht, "Saturation in PSO neural network training: Good or evil?". [Semantic Scholar](https://www.semanticscholar.org/paper/Saturation-in-PSO-neural-network-training:-Good-or-Rakitianskaia-Engelbrecht/9fde48babd417cfb7217f5312b90b6b435fc883c)

---

## 👨‍💻 Yazar

- **İlhan Koçaslan** — [GitHub: @Proaiml](https://github.com/Proaiml)
