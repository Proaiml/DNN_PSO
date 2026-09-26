"""
Örnek 4 - Kara kutu: yalnızca simülasyonla eğitilen sinir ağı kontrolcüsü.

    python examples/04_su_tanki_kontrolcu.py

Bir su tankının seviyesi bir vana ile ayarlanıyor. Tank doğrusal değil:
çıkış debisi seviyenin karekökü ile artar ve vana en fazla %100 açılır.
"Doğru vana açıklığı" etiketi yoktur; elimizde yalnızca bir simülatör var.

Eğitim: her aday ağ tankı 400 adım yönetir; maliyet = ortalama takip hatası
+ vana titreşimi cezası. PSO bu tek sayıya bakarak ağı iyileştirir (türev,
etiket, tank modelinin denklemi gerekmez; simülatör gerçek bir test düzeneği
de olabilir). batch=True: tüm sürü simülasyonu aynı anda, NumPy ile yürütür.

Karşılaştırma: aynı simülatörde 525 katsayı çifti taranarak bulunmuş klasik
PI kontrolcü. İkisi de hiç görmedikleri bir hedef dizisinde ve kesiti %30
büyütülmüş (modeli değişmiş) bir tankta da denenir.

Ölçülen sonuç (ortalama |hata|, m): eğitim 0.035 / PI 0.040, görülmemiş
hedefler 0.078 / 0.078, değişmiş tank 0.103 / 0.100. Yani sıfırdan, yalnızca
simülasyonla eğitilen ağ, dikkatle ayarlanmış bir PI'ın düzeyine ulaşıyor.
Bu basit tank için PI zaten yeterli; yöntemin değeri PI'ın yetmediği (çok
girişli, gecikmeli, kısıtlı, maliyeti alışılmadık) sistemlerde ortaya çıkar.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import fit_blackbox  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DT, Q_MAX, H_MAX, K_OUT = 1.0, 0.08, 2.0, 0.03


def simulate(controller, setpoints, n, area=1.0):
    """n kontrolcüyü aynı anda simüle eder. controller(e, h, sp, integral) -> vana (0-1), hepsi (n,) dizileri.
    Döner: (maliyet, ortalama |hata|), ikisi de (n,)."""
    h = np.full(n, 0.2)
    u_prev, integral = np.zeros(n), np.zeros(n)
    err_sum, move_sum = np.zeros(n), np.zeros(n)
    for sp in setpoints:
        e = sp - h
        integral = np.clip(integral + e * DT, -50, 50)
        u = np.clip(controller(e, h, np.full(n, sp), integral), 0.0, 1.0)
        h = np.clip(h + DT * (Q_MAX * u - K_OUT * np.sqrt(h)) / area, 0.0, H_MAX)
        err_sum += np.abs(e)
        move_sum += (u - u_prev) ** 2
        u_prev = u
    mae = err_sum / len(setpoints)
    return mae + 0.5 * move_sum / len(setpoints), mae


TRAIN = np.repeat([0.5, 1.2, 0.8, 1.5], 100)
TEST = np.repeat([1.0, 0.4, 1.3, 0.7, 1.1], 80)


def nn_controller(policy):
    # ağın girişleri (ölçekli): hata, hedef seviye, hatanın integrali
    return lambda e, h, sp, i: policy(np.c_[e / H_MAX, sp / H_MAX, i / 10.0])[:, 0]


started = time.perf_counter()
net, cost, _ = fit_blackbox(lambda policy: simulate(nn_controller(policy), TRAIN, n=60)[0],
                            layers=[3, 4, 1], particles=60, iterations=200, restarts=2, batch=True, seed=4)
nn_seconds = time.perf_counter() - started
nn = nn_controller(lambda x: net.forward(x))

# klasik PI: 25 x 21 Kp/Ki çifti aynı simülatörde, aynı maliyetle taranır (o da toplu simülasyon)
KP, KI = np.meshgrid(np.linspace(0.1, 40, 25), np.linspace(0.0, 0.3, 21))
KP, KI = KP.ravel(), KI.ravel()
pi_cost, _ = simulate(lambda e, h, sp, i: KP * e + KI * i, TRAIN, n=KP.size)
kp, ki = KP[pi_cost.argmin()], KI[pi_cost.argmin()]
pi = (lambda e, h, sp, i: kp * e + ki * i)

print(f"Sinir ağı kontrolcüsü: mimari {net.layers} (+bias), {net.dimensions} ağırlık, "
      f"yalnızca simülasyonla {nn_seconds:.1f} sn'de eğitildi")
print(f"Klasik PI kontrolcü  : Kp={kp:.2f}, Ki={ki:.3f} (525 çift tarandı)\n")
print(f"{'senaryo':<44}{'sinir ağı':>11}{'PI':>9}   ortalama |hata| (m)")
for title, sp, area in (("eğitim hedefleri", TRAIN, 1.0),
                        ("hiç görülmemiş hedefler", TEST, 1.0),
                        ("görülmemiş hedefler + tank kesiti %30 büyük", TEST, 1.3)):
    print(f"{title:<44}{simulate(nn, sp, 1, area)[1][0]:>11.4f}{simulate(pi, sp, 1, area)[1][0]:>9.4f}")
