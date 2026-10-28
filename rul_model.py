import os, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, "scaler.json")))
SENSORS = cfg["sensors"]
MEAN = np.array(cfg["mean"], dtype=np.float64)
SCALE = np.array(cfg["scale"], dtype=np.float64)
CAP, L = cfg["rul_cap"], cfg["window"]
_w = np.load(os.path.join(BASE, "weights.npz"))
W = {k: _w[k].astype(np.float64) for k in _w.files}

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def _lstm(X, wi, wh, bi, bh):
    H = wh.shape[1]
    h, c, out, b = np.zeros(H), np.zeros(H), [], bi + bh
    for x in X:
        g = wi @ x + wh @ h + b
        i, f, gg, o = _sig(g[:H]), _sig(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), _sig(g[3 * H:])
        c = f * c + i * gg
        h = o * np.tanh(c)
        out.append(h)
    return np.array(out)

def predict_rul(rows):
    """rows: list of cycles (oldest first), each a list of the 14 raw sensor values -> (RUL, n_cycles, warning)"""
    a = (np.array(rows, dtype=np.float64) - MEAN) / SCALE
    n, warning = len(a), None
    if n < L:
        a = np.vstack([np.repeat(a[:1], L - n, axis=0), a])
        warning = f"Only {n} cycles received; {L} recommended. Prediction may be less accurate."
    a = a[-L:]
    o1 = _lstm(a, W["lstm_weight_ih_l0"], W["lstm_weight_hh_l0"], W["lstm_bias_ih_l0"], W["lstm_bias_hh_l0"])
    o2 = _lstm(o1, W["lstm_weight_ih_l1"], W["lstm_weight_hh_l1"], W["lstm_bias_ih_l1"], W["lstm_bias_hh_l1"])
    z = np.maximum(0, W["fc_0_weight"] @ o2[-1] + W["fc_0_bias"])
    y = float((W["fc_2_weight"] @ z + W["fc_2_bias"])[0]) * CAP
    return float(np.clip(y, 0, CAP)), n, warning
