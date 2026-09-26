"""
Örnek 1 - Orijinal sınıf ile hızlı motor aynı sonucu verir.

    python examples/01_orijinal_ve_hizli_motor.py

1. Aynı veri, aynı PSO bütçesi ve aynı tohum (seed) ile önce orijinal
   prodnnv10 sınıfı (class_prodnn.py, değiştirilmeden), sonra hızlı motor eğitilir.
2. Hızlı motorun bulduğu ağırlıklar prodnnv10 biçimine çevrilip gerçek bir
   prodnnv10 nesnesine yüklenir; orijinal kodun kendi ileri beslemesi aynı
   tahminleri üretir.
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from class_prodnn import prodnnv10  # noqa: E402  (orijinal kod)
from dnnpso import PSONetwork  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

X = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
y = np.array([1, 0, 0, 0, 0, 1, 0, 0])
budget = dict(particles=15, iterations=50, restarts=2, seed=7)

print("Veri: data/data_x.json, data/data_y.json (3 giriş, 8 örnek)\n")
for engine in ("original", "fast"):
    started = time.perf_counter()
    net = PSONetwork.original(engine=engine, **budget).fit(X, y)
    seconds = time.perf_counter() - started
    print(f"motor={engine:<8} mimari {net.layers_}  en iyi maliyet {net.best_cost_:.6f}  "
          f"süre {seconds:.2f} sn  tahmin {np.round(net.predict_raw(X), 3).tolist()}")

print("\nHızlı motorun ağırlıkları orijinal sınıfa yükleniyor...")
weights = net.to_prodnn_weights()                   # {"w1_0_0": ..., ...}
model = prodnnv10(3, 1, 2 / 3, str(ROOT / "data" / "data_x.json"), str(ROOT / "data" / "data_y.json"), 1, 2, 1)
model.all_Weights.update(weights)
with contextlib.redirect_stdout(io.StringIO()):
    cost = model.propagate_forward()
print(f"orijinal propagate_forward maliyeti: {cost:.6f}")
print(f"orijinal tahminler: {np.round(model.full_outputs_lastmend, 3).tolist()}")
print(f"hızlı motor ile fark: {np.max(np.abs(np.array(model.full_outputs_lastmend) - net.predict_raw(X))):.2e}")
