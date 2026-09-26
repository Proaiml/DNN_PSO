"""
Örnek 3 - Türevi olmayan ağ: "basamak" (0/1) nöronlar.

    python examples/03_basamak_noronlar.py

Basamak fonksiyonunun türevi her yerde 0'dır; geri yayılım (backprop) böyle
bir ağı hiç eğitemez. PSO yalnızca maliyete baktığı için eğitir.

Neden ilginç? Eğitilen ağın her nöronu yalnızca "toplam > 0 mı?" sorusunu
sorar: çarpma sonrası karşılaştırma. Mikrodenetleyici, FPGA ya da mantık
devresi gibi kayan nokta birimi zayıf donanımlarda doğrudan çalışabilir.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import PSONetwork, train_test_split  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.default_rng(3)
t = rng.uniform(0, np.pi, 200)
X = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]]) + rng.normal(0, 0.12, (400, 2))
y = np.r_[np.zeros(200), np.ones(200)].astype(int)                       # "iki hilal" veri seti
Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed=0)

print("İki hilal (400 örnek, %30 test)\n")
# 2 girişli bir problemde piramit kuralı en fazla 2 gizli nöron verir; burada gizli katman elle seçiliyor
for title, kw in (("sigmoid nöronlar                ", dict(activation="sigmoid")),
                  ("basamak nöronlar, hedef=doğruluk", dict(activation="step", objective="accuracy"))):
    started = time.perf_counter()
    net = PSONetwork(hidden_layers=[6], particles=60, iterations=300, seed=1, **kw).fit(Xtr, ytr)
    print(f"{title}: mimari {net.layers_}, test doğruluğu {net.score(Xte, yte):.3f}, "
          f"{time.perf_counter() - started:.1f} sn")

W = net.net_.weights
print("\nBasamak ağının ilk katmanı (her satır bir nöron: x1, x2, bias):")
print(np.round(W[0], 2))
print("Tahmin için gereken işlem: her nöronda birkaç çarpma + toplama + 1 karşılaştırma.")
