# DNN_PSO: Deep Neural Network Training via Particle Swarm Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PySwarms](https://img.shields.io/badge/pyswarms-1.3.0-orange.svg)](https://github.com/ljvmiranda921/pyswarms)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**DNN_PSO**, derin yapay sinir ağlarının (Deep Neural Network - DNN) ağırlıklarını klasik geriye yayılım (backpropagation / gradient descent) **kullanmadan**, doğadan esinlenen **Parçacık Sürü Optimizasyonu (Particle Swarm Optimization - PSO)** ile eğiten meta-sezgisel bir yapay zeka algoritmasıdır.

---

## 📌 Neden Gradyan İnişi Yerine PSO?

Geleneksel derin öğrenme modelleri ağırlık güncellemelerini geriye yayılım (Backpropagation) ve gradyan inişi (Gradient Descent) ile gerçekleştirir. Ancak bu yaklaşımın bazı temel sınırları vardır:

1. **Türev Zorunluluğu:** Aktivasyon ve maliyet fonksiyonlarının kesinlikle sürekli ve türevlenebilir olması gerekir.
2. **Yerel Minimumlar (Local Minima):** Gradyan inişi yerel minimumlara, eyer noktalarına (saddle points) veya sıfır gradyan platolarına takılabilir.
3. **Gradyan Kaybolması / Patlaması:** Çok katmanlı derin ağlarda gradyanlar kaybolabilir (vanishing gradient) veya kararsızlaşabilir (exploding gradient).

### 🚀 PSO'nun Sağladığı Avantajlar:
- **Türev Gerektirmez (Derivative-Free):** Sadece ileri besleme (forward propagation) ile hesaplanan hata değerini (Cost/MSE) kullanarak optimizasyon yapar.
- **Küresel Arama Yeteneği (Global Search):** Sürüdeki parçacıklar arama uzayının farklı noktalarını aynı anda tarar; böylece yerel minimumlardan kaçma şansı çok daha yüksektir.
- **Dinamik Ağ Yapısı:** Katman boyutları ve nöron sayıları `transition_per` parametresi ile otomatik ve orantılı olarak şekillendirilir.

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
Sürüdeki her parçacık, kişisel en iyi deneyimi ($\mathbf{p}_i$) ve sürünün küresel en iyisi ($\mathbf{g}$) doğrultusunda güncellenir:
$$\mathbf{v}_i^{(t+1)} = w \cdot \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^{(t)})$$
$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
- $w = 0.9$: Eylemsizlik ağırlığı (Inertia weight)
- $c_1 = 1.0$: Bilişsel öğrenme katsayısı (Cognitive parameter)
- $c_2 = 0.8$: Sosyal öğrenme katsayısı (Social parameter)
- $r_1, r_2 \sim U(0, 1)$: Rastgele katsayılar

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

---

## 📁 Proje Dosya Yapısı

```
DNN_PSO/
│
├── class_prodnn.py              # prodnnv10 sınıfı (Orijinal algoritma çekirdeği)
├── dnn+pso.py                   # Prototip PSO betiği (Orijinal geliştirme kodu)
├── example.py                   # Kullanıma hazır hızlı başlangıç ve test betiği
├── run.bat                      # Windows tek tıkla çalıştırma başlatıcısı
├── requirements.txt             # Gerekli Python kütüphaneleri
├── README.md                    # Proje dokümantasyonu
├── particle-swarm-optimization.pdf # PSO teorik makalesi
│
└── data/                        # Örnek hazır veri setleri
    ├── data_x.json              # 3-girişli örnek veri (8 örnek)
    ├── data_y.json              # 3-girişli hedef etiketler
    ├── xor_x.json               # Klasik 2-girişli XOR veri seti
    └── xor_y.json               # 2-girişli XOR hedef etiketleri
```

---

## 🚀 Hızlı Başlangıç (Quickstart)

### 1. Kütüphanelerin Yüklenmesi
Terminal veya komut satırından bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Örnek Scripti Çalıştırma
Örnek veri seti üzerinde eğitimi başlatmak için:
```bash
python example.py
```
*(Windows kullanıcıları doğrudan **`run.bat`** dosyasına çift tıklayarak da başlatabilir).*

### Konsol Çıktısı Örneği:
```text
======================================================================
       DNN + PSO: Deep Neural Network Trained by Particle Swarms
======================================================================

[*] Architecture Overview:
    - Input Layer:      3 neurons
    - Hidden Layer(s):  [2] neurons
    - Output Layer:     1 neurons
    - Layer Structure:  [3, 2, 1]
    - PSO Search Space: 8 weight dimensions to optimize

[+] Starting PSO Optimization (2 cycle(s) x 50 iterations)...
Optimization finished | best cost: 4.278784

==================================================
               PREDICTION RESULTS
==================================================
Sample   Input (X)          Actual (Y)   Raw Output     Binary Pred 
-----------------------------------------------------------------
#1       [0, 0, 0]          1            0.9597         1            CORRECT
#2       [0, 0, 1]          0            0.1256         0            CORRECT
#3       [0, 1, 0]          0            0.0000         0            CORRECT
#4       [0, 1, 1]          0            0.0000         0            CORRECT
#5       [1, 0, 0]          0            0.1373         0            CORRECT
#6       [1, 0, 1]          1            0.4468         0            WRONG
#7       [1, 1, 0]          0            0.0000         0            CORRECT
#8       [1, 1, 1]          0            0.0000         0            CORRECT
-----------------------------------------------------------------
Accuracy: 87.5% (7/8 correct)
==================================================
```

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

### ⚙️ Parametre Tablosu

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

---

## 📊 Kendi Veri Setinizle Kullanma

Model verileri standart JSON formatında kabul eder:

**Giriş Verisi (`my_x.json`):**
```json
[
  [0.5, 1.2, -0.3],
  [1.0, 0.2,  0.8]
]
```

**Çıkış Verisi (`my_y.json`):**
```json
[1, 0]
```

---

## 👨‍💻 Yazar

- **İlhan Koçaslan** — [GitHub: @Proaiml](https://github.com/Proaiml)