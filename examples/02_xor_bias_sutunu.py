"""
Örnek 2 - XOR: orijinal mimari öğrenemez, bir "bias sütunu" ile öğrenir.

    python examples/02_xor_bias_sutunu.py

prodnnv10'da bias yoktur ve katmanlar transition_per ile daralır. 2 girişli
XOR için: int(2 * 2/3) = 1, çıkış sayısından (1) büyük değil -> gizli katman
oluşmaz, ağ [2, 1] olur. Tek nöron, biassız: XOR imkânsızdır (her girişte 0.5).

Girişlere sabit 1 değerli bir sütun eklemek (bias="input") iki sorunu
birden çözer: ağ [3, 2, 1] olur ve ilk katmanın biası olur.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import PSONetwork  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

for title, make in (("orijinal (bias yok)", lambda s: PSONetwork.original(seed=s)),
                    ("bias sütunu ile   ", lambda s: PSONetwork(seed=s))):
    solved, outputs = 0, None
    for seed in range(10):
        net = make(seed).fit(X, y)
        ok = net.score(X, y) == 1.0
        solved += ok
        if outputs is None or ok:
            outputs = np.round(net.predict_raw(X), 3).tolist()
    print(f"{title}: mimari {net.layers_}, 10 denemenin {solved}'inde XOR çözüldü, örnek çıktı {outputs}")
