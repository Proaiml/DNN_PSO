"""
README görsellerini üretir:

    python docs/make_figures.py

docs/images/egitim_ve_karar_siniri.png : iki hilal verisinde karar bölgeleri + eğitim maliyet eğrisi
docs/images/su_tanki.png               : simülasyonla eğitilen sinir ağı kontrolcüsü ve PI, görülmemiş hedeflerde
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dnnpso import PSONetwork, train_test_split  # noqa: E402

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2563eb", "#ea580c", "#6b7280"


def decision_figure():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, np.pi, 200)
    X = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]]) + rng.normal(0, 0.12, (400, 2))
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed=0)
    net = PSONetwork(hidden_layers=[6], particles=60, iterations=300, seed=1).fit(Xtr, ytr)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.1, 1.6, 220))
    prob = net.predict_raw(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    a.contourf(gx, gy, prob, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    a.contour(gx, gy, prob, levels=[0.5], colors="k", linewidths=1.2)
    for cls, color in ((0, BLUE), (1, ORANGE)):
        a.scatter(*Xte[yte == cls].T, s=16, color=color, edgecolor="white", linewidth=0.4,
                  label=f"sınıf {cls} (test)")
    a.set_title(f"Karar sınırı - mimari {net.layers_}, test doğruluğu %{100 * net.score(Xte, yte):.1f}")
    a.set_xlabel("x1")
    a.set_ylabel("x2")
    a.legend(loc="upper right", frameon=False)

    b.plot(net.cost_history_, color=BLUE, lw=1.6)
    for k in range(1, net.restarts):
        b.axvline(k * net.iterations, color=GREY, lw=0.8, ls="--")
    b.set_yscale("log")
    b.set_title("PSO eğitimi: sürünün en iyi maliyeti")
    b.set_xlabel(f"iterasyon ({net.restarts} ayrı başlangıç, kesikli çizgiler)")
    b.set_ylabel("maliyet (100 × MSE, log)")
    fig.tight_layout()
    path = OUT / "egitim_ve_karar_siniri.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def tank_figure():
    spec = importlib.util.spec_from_file_location("tank", ROOT / "examples" / "04_su_tanki_kontrolcu.py")
    tank = importlib.util.module_from_spec(spec)
    sys.stdout = open(__import__("os").devnull, "w", encoding="utf-8")
    try:
        spec.loader.exec_module(tank)                     # eğitir ve PI'ı ayarlar (örnek 4 ile aynı)
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def trace(controller, setpoints):
        h, integral, levels, valves = 0.2, 0.0, [], []
        for sp in setpoints:
            e = sp - h
            integral = float(np.clip(integral + e * tank.DT, -50, 50))
            u = float(np.clip(controller(np.array([e]), np.array([h]), np.array([sp]), np.array([integral]))[0],
                              0.0, 1.0))
            h = float(np.clip(h + tank.DT * (tank.Q_MAX * u - tank.K_OUT * np.sqrt(h)), 0.0, tank.H_MAX))
            levels.append(h)
            valves.append(u)
        return np.array(levels), np.array(valves)

    sp = tank.TEST
    h_nn, u_nn = trace(tank.nn, sp)
    h_pi, u_pi = trace(tank.pi, sp)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a.step(range(len(sp)), sp, where="post", color="k", lw=1.2, ls="--", label="hedef seviye")
    a.plot(h_nn, color=BLUE, lw=1.8, label="sinir ağı (yalnızca simülasyonla, PSO)")
    a.plot(h_pi, color=ORANGE, lw=1.4, alpha=0.9, label=f"klasik PI (Kp={tank.kp:.1f}, Ki={tank.ki:.2f})")
    a.set_ylim(0.15, 1.62)                                  # room for the legend above the curves
    a.set_ylabel("seviye (m)")
    a.set_title("Su tankı - eğitimde hiç görülmemiş hedef dizisi")
    a.legend(loc="upper right", frameon=False, ncol=3)
    b.plot(u_nn, color=BLUE, lw=1.2)
    b.plot(u_pi, color=ORANGE, lw=1.0, alpha=0.8)
    b.set_ylabel("vana (0-1)")
    b.set_xlabel("adım")
    fig.tight_layout()
    path = OUT / "su_tanki.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (decision_figure(), tank_figure()):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1024:.0f} KB")
