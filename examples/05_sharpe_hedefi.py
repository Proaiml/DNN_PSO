"""
Örnek 5 - Hedef doğrudan Sharpe oranı (işlem maliyeti dahil).

    python examples/05_sharpe_hedefi.py

YALNIZCA YÖNTEM GÖSTERİMİDİR, yatırım tavsiyesi değildir; veri SENTETİKTİR.

Klasik yaklaşım: ağı "yarın fiyat artar mı?" etiketine göre (MSE) eğitmek,
sonra çıktıyı pozisyona çevirmek. Oysa asıl hedef tahmin doğruluğu değil,
işlem maliyeti düşüldükten sonraki risk-getiri oranıdır (Sharpe). Sharpe'ın
pozisyon değişimlerine ve maliyete bağlı hâli türevlenebilir bir kayıp değildir;
PSO ise doğrudan onu en iyiler.

Sentetik piyasa iki rejim arasında geçiş yapar: düşük oynaklıkta trend
(getiriler kendini sürdürür), yüksek oynaklıkta ters dönüş. Ağ girişleri:
son iki günün getirisi ve 10 günlük oynaklık. Pozisyon ya 1 (al) ya 0 (nakit):
ağ çıktısı >= 0.5 ise 1. Eğitim ilk %70, sonuçlar hiç görülmemiş son %30 üzerindedir.

Denenmiş bir tuzak: pozisyon sürekli (0-1 arası) bırakıldığında PSO, Sharpe
oranının pozisyon büyüklüğünden bağımsız olmasını "keşfetti": neredeyse sıfır
pozisyonla, getirisi ~%0 olan ama Sharpe'ı yüksek görünen stratejiler buldu.
Metriği doğrudan en iyilemenin bedeli budur: hedef, istenmeyen çözümlere izin
vermeyecek biçimde tanımlanmalı. Burada pozisyonun 0/1 olması bunu önler.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import PSONetwork, fit_blackbox  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
COST = 0.0005                                               # pozisyon değişimi başına %0.05


def market(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    regime, left = 0, 0
    for t in range(2, n):
        if left <= 0:
            regime, left = rng.integers(0, 2), rng.integers(30, 120)
        left -= 1
        if regime == 0:                                     # sakin trend
            r[t] = 0.35 * r[t - 1] + rng.normal(0.0002, 0.006)
        else:                                               # oynak, ters dönüşlü
            r[t] = -0.35 * r[t - 1] + rng.normal(0.0, 0.018)
    return r


def stats(position, returns):
    """(yıllık Sharpe, toplam getiri %, işlem hacmi) - işlem maliyeti düşülmüş."""
    position = np.asarray(position, dtype=float)
    pnl = position * returns - COST * np.abs(np.diff(np.r_[0.0, position]))
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)
    return sharpe, pnl.sum() * 100, np.abs(np.diff(position)).sum()


def run(seed):
    r = market(seed=seed)
    vol = np.array([r[max(0, t - 10):t].std() if t > 1 else 0.0 for t in range(len(r))])
    X = np.c_[r[1:-1], r[:-2], vol[2:]]                     # t günü bilinenler (t-1, t-2 getirisi, oynaklık)
    cut = int(len(X) * 0.7)
    X = X / np.abs(X[:cut]).max(axis=0)                     # yalnızca eğitim döneminden ölçek
    future = r[2:]                                          # t gününün getirisi: pozisyon bir gün önceden alınır

    def neg_sharpe(policy):
        return -stats(policy(X[:cut])[:, 0] >= 0.5, future[:cut])[0]

    net, _, _ = fit_blackbox(neg_sharpe, layers=[3, 4, 1], particles=40, iterations=150, restarts=2, seed=3)
    pos_sharpe = (net.forward(X)[:, 0] >= 0.5).astype(float)
    clf = PSONetwork(hidden_layers=[4], seed=3, scale=None, bias="neuron").fit(
        X[:cut], (future[:cut] > 0).astype(int))
    pos_mse = clf.predict(X).astype(float)                  # "yarın artar mı?" tahmini evet ise al
    return [stats(pos[cut:], future[cut:]) for pos in (np.ones(len(X)), pos_mse, pos_sharpe)]


print("Test dönemi (hiç görülmemiş son %30), 5 farklı sentetik piyasa\n")
print(f"{'piyasa':<8}{'al ve tut':>11}{'MSE ağı':>10}{'Sharpe ağı':>12}   {'işlem hacmi MSE / Sharpe':>26}")
wins = 0
for seed in (5, 11, 21, 33, 47):
    (bh, _, _), (ms, _, mt), (ss, _, st) = run(seed)
    wins += ss > ms
    print(f"{seed:<8}{bh:>11.2f}{ms:>10.2f}{ss:>12.2f}   {mt:>14.0f} / {st:<8.0f}")
print(f"\nDoğrudan Sharpe ile eğitilen ağ 5 piyasanın {wins}'inde daha yüksek Sharpe verdi "
      f"(tablodaki değerler Sharpe oranıdır).")
print("Not: sentetik veri; gerçek piyasada örnek dışı test, işlem maliyeti ve aşırı uyum kontrolü şarttır.")
