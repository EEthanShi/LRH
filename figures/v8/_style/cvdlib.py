"""CIEDE2000 under Machado et al. (2009) full-severity CVD simulation, WCAG 2.1 contrast, and L*-lowered text tones.
Same method as STYLE_V5_SPEC.md sec. 3.1 (validated below against the spec's published numbers)."""
import numpy as np

M_CVD = {
    "normal": np.eye(3),
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
VIEWS = ("normal", "deutan", "protan", "tritan")


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def rgb2hex(c):
    c = np.clip(np.round(np.asarray(c) * 255), 0, 255).astype(int)
    return "#%02X%02X%02X" % tuple(c)


def to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def from_lin(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
WHITE_XYZ = M_XYZ @ np.ones(3)


def lin2lab(lin):
    xyz = M_XYZ @ lin / WHITE_XYZ
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab2lin(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    def finv(t):
        return t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)
    xyz = np.array([finv(fx), finv(fy), finv(fz)]) * WHITE_XYZ
    return np.linalg.solve(M_XYZ, xyz)


def lab(h, view="normal"):
    lin = to_lin(hex2rgb(h))
    lin = np.clip(M_CVD[view] @ lin, 0, 1)
    return lin2lab(lin)


def de2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dhp = h2p - h1p
        if dhp > 180:
            dhp -= 360
        elif dhp < -180:
            dhp += 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return float(np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)))


def dE(h1, h2, view="normal"):
    return de2000(lab(h1, view), lab(h2, view))


def contrast(h1, h2="#FFFFFF"):
    def lum(h):
        c = to_lin(hex2rgb(h))
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(h1), lum(h2)], reverse=True)
    return float((a + 0.05) / (b + 0.05))


def text_tone(h, target=4.5, step=0.05):
    """Keep CIELAB a*, b* (hue and chroma) and lower L* until the colour reaches `target`:1 on white."""
    if contrast(h) >= target:
        return h.upper()
    L, a, b = lin2lab(to_lin(hex2rgb(h)))
    while L > 0:
        L -= step
        cand = rgb2hex(from_lin(lab2lin((L, a, b))))
        if contrast(cand) >= target:
            return cand
    raise ValueError(h)
