# Exploratory (not citable): per-LM median offset c_alpha - Bbar vs PR*, from design/known_offsets.txt
import numpy as np
d = {"prh": [(133.9,.00219,.00280),(124.5,.00223,.00281),(242.4,.00195,.00213),(248.2,.00190,.00226),(249.3,.00194,.00226),
             (201.9,.00190,.00211),(228.6,.00193,.00213),(257.3,.00184,.00214),(308.6,.00182,.00212),(359.2,.00171,.00190)],
     "l2":  [(79.6,.00231,.00246),(93.2,.00221,.00250),(115.7,.00228,.00295),(157.2,.00207,.00363),(141.9,.00219,.00321),
             (160.1,.00202,.00241),(180.0,.00204,.00239),(131.4,.00214,.00246),(157.0,.00210,.00243),(162.4,.00209,.00243)]}
for k, rows in d.items():
    a = np.array(rows); x = 1/np.sqrt(a[:,0]); y = a[:,1]; mx = a[:,2]
    X = np.c_[np.ones_like(x), x]; b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s = np.sqrt((r**2).sum()/(len(y)-2)); exc = np.max(mx - y)
    print(k, "fit med = %.5f + %.5f/sqrt(PR*)" % tuple(b), "resid sd %.5f" % s, "max(pair max - LM median) %.5f" % exc,
          "corr %.2f" % np.corrcoef(x, y)[0,1])
    for pr in (20, 30, 40, 60, 80, 120):
        m = b[0] + b[1]/np.sqrt(pr); print("   PR*=%3d  median %.5f  pair max ~ %.5f" % (pr, m, m + exc))
