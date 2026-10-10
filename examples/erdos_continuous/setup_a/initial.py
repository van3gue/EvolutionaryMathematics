"""
Setup A: SOTA refinement.

What's evolved: the heights array (starting point) AND the local polisher
that refines it. This file must be fully self-contained (no importing
sibling files like common.py) because ShinkaEvolve copies this file into
an isolated folder for each generation - only evaluate.py, which stays in
its original location, can safely import shared helpers.
"""

import base64
import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate


def _worst_case_overlap(h: np.ndarray) -> float:
    """Self-contained copy of the correlation objective (see common.py for
    the fully documented version - kept in sync manually since this file
    must stand alone)."""
    n = h.shape[0]
    w = 2.0 / n
    g = 1.0 - h
    corr = correlate(h, g, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    mask = lags != 0
    return float(np.max(corr[mask]) * w)

# The n=600 warm start is embedded here (float64, base64) instead of being
# loaded from a sibling .npy file: ShinkaEvolve copies this file alone into an
# isolated per-generation folder, so any sibling data file silently goes
# missing. Keep this OUTSIDE the evolve block so mutations cannot corrupt it.
_WARM_START_N600_B64 = (
    "AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/BBR1XfH/7z8AAAAAAADwPwAAAAAAAPA/8saw1Z397z8AAAAAAADwPwAA"
    "AAAAAPA/AAAAAAAA8D+e4HaXuP/vPwAAAAAAAPA/AAAAAAAA8D/9W3Gu7//vP4/O3WwU/O8/AAAAAAAA8D+kp1eT"
    "bf3vPwAAAAAAAPA/AAAAAAAA8D9V4lnv2f/vP20KINWT/e8/a3G0dl/47z8AAAAAAADwPwWISg3M/+8/2gu/R5X4"
    "7z/9wCkCXOvvPwAAAAAAAPA/AAAAAAAA8D+QLkNuRf3vP8yTMiUa/O8/iAul+Mz/7z8AAnwrBFjtP25dk4BA/+8/"
    "SYQTXI3/7z+9aVSxNvvvPwp3tM1u/O8/ZYPZaVjl7j9sovtDpcPvP9ckM7ph8+8/fiZmL8SK6z/SqNXbpJ7vP/gg"
    "eerZ9O8/sZUYDLbs7z8AAAAAAADwP7zLJdDWiu8/g5A9oTm97j+q1KJQZPzvP+Hd+8M9Ie0/6TlvUaRP7T/UFmqQ"
    "A/bnP5BzJHi5veM/gZc2nvFU4j+oOOksFxPaP/tgi09jM+s/1x4MxSkt5j8ktR32ZnfrP/Q6C4WX8Ok/++jUcMjb"
    "5z86oqSFZgvmPwJkqhPBkuY/RtDlZEVQ6T8oJlkLHP3kP4LParBku+A/PiniXcgz4j/qBSxXx7TlP32Qn92AWuE/"
    "SV85w0Sl5j+u5PxrDIbiP9XLuzx78eQ/TZ1VXoqo5D9qYqLahWTcP4s9lqPC8eM/bOwHnvqE4z/0D4saNM3mPxar"
    "uAdvCOY/XgqFGNqg5T8oa/3NwPTcP3k+4D/9TOI/HmZhXbkt6D/KWnSb4nnmP1xjk2vuJeU/fyZIa7w85T9P+JDQ"
    "YOLZP+YB+fF6N+c/pK/rHe+/5T/j0rcPh/zfP9n7EQZHieY/Z7ZkTEEf5j8/26I1LGXdP4lY0yTNPeY/Svv1BtSa"
    "6D/pjQoBwSjkP/gP4zuM6uQ/w5hxfNpCxj+2KzGwjy/pP8+2IdCWo+M/36UEEKG34T95mra+VNzlP2WDR5pyT+E/"
    "VLAmRKoK4z8JDfjO6K/jPxarKrtSIuA/i5uDJp3P5D/pB5LJLqLjP0ncZVrb1d4/PzZBL92Z6D/pRcm3fRXhPxX4"
    "DMpzx+Q/Zn1aKrSk4T886ydJMrPoP6NYp46D19I/Meo+y+pt2T/XpbJ8kAPkP6fxzWPwuuE/Y6gRIVBT4D888Br6"
    "kbreP4zaH6KHg94/KSHx9h+h5D8zSC2J/trbP7gX2UmBNd0/aM/r6Nqi4T/aAi3uQXPgP080ZqVYzd8/sYuuiF1v"
    "5D8nKlwrU4TZP8sAVtYsMNo/3/W29CZx4j/+1mkWFu3hP5q2DSCobOA/abTZVQ5I4T9LkT7b/+PaP8lkpSAF7uM/"
    "erhQ27RM3j/6h50smFbSP4Ptc1pPSN4/JFslJGKL4T8332wKf2fiPxXF6Jk0L90/wAKZCZDG4z83jPS7anTaPwnF"
    "07Ems+A/qu+/RfkP4j+vDSR1+jjkP7rYK/o3bsw/kbxjQHK84z9AylHkUQ/fP9x4sQc0ZuE/FubJjk4t5T9Kn1aE"
    "AubiPwRXBMGgrd4/+6CsyGjI2j8JwV7MMiviP6ev37rlIeY/E3lhYdQa2T/bO7l48CTbP03qaFQb1d4/UvLDUXVH"
    "3T+vY5g0lHXXPykXSQwIDeM/GJMBO/C10D8I7UsyV0vgPwu31SmrceA/PwbA/z0j1j/cct002e7UP1y5qmlAD98/"
    "92TvDdBP4D9u9FFCP7fhP/BHbVMbid0/roFsA2SxzT86C0q9uVzeP8HUjxfRcuM/NN5LkJz32T+fZIDPYsHYP7LQ"
    "nuo/keE/pFBQqHY71D+vBKm/taPUP2d+G6ixyuM/rZWCu+2X3T/waxrpEF7LP83LRJ4FUts/tjMikjRb4D8k56To"
    "1v7aP1E+Rg8BjNs/dJoQghfZ2D8F+p1eP8DcP2fMfBstctw/q8ejOOAk2j+kFQmdmBPiP0r4ULGwkNU/ThL0s9Qi"
    "3z9h7pypFMzYP/hdHkHwB94/d5NFPmFbzD8tw+0+6xjiP6X4s3g5Gtw/zKZbKZxS1T90RAJ33NHcP0V6cMQdvN8/"
    "McsITDjd2j+6mGl9oO/fP/mCAfHeY9U/QrThgIkZ2j+UUf1jnoPWP/Cq79OQItc/ljZh2eE23D80fLTiuP7NP13i"
    "kGtpDtc/KQh3ZYdL2z/VJaOyFiHTPxS3EQjxMdM/+/292vp+0z9S68g7APnLPy3++AbpGNU/V5+kpg690z+k1us+"
    "adbTPzb6Ha5XkMw/1kevKOKZxD//Eb1nBcvRPy4YEdR2qNE/gii7K167xj9bjrkpspPRPwJeVpjOgdI/AqPYAZlT"
    "xz9w7RIVjlrEP/5BUcGyqNA/ZXJ1ezhNyj/Gx8TinRTSP+cmq5BSX8Y/uvoNPTtLzT/TzJDIN0e/PwTo/q2Q3Mo/"
    "EzW5ohGB2D+VGxfxXsHWP/V2Aim7A7k/QjhUa+DHwz8WsA8tiTvPP/8G0rb5f9M/s5T9gbTmzz/LM0F/NxnDP9Wh"
    "1kHTmM8/XE36EE8u1D9LtjJImLHIPz3GE2cBhtg/hBV9ntXEyz/kW3uuJzPIPy554cNNEtE/dApTwIfQ0T9pzRAH"
    "id3WP87tw3iN994/tdP2D4kSzT+UcQX3TALfP4Le8FfA2NA/Ea4B/bPu0z/v7/rvc03UP9Xvl2ixqr8/zPJGttpi"
    "2T9WjJyxGP7dPzP2+D7h2sM/eyZ20Voqyj8TqOMHrmnKP5u+fDSwgNA/TnP8kcyFyT+ywuu0omDGPwlEaZAH8sM/"
    "nfXWevdWxz9WtOEyXsXDP/UCmV2AQdM/9p5/sAKrwD8brY6naVvKP0jvgiAZbMk/ELFwZSDCtT+6ZBC0Fne+P9Vo"
    "0/DAK80/LXgoFuQt0D92KX/g60bGP8DTziH/Gck/P3+2TSNaxz9B83nWE/jGP1j1ic51zr0/nzLVOimZzj9hNwOa"
    "uAXDP08UxjJ40cU/H1G4PW5/wz+msisrLZ/GP9a3ZKaA67w/k/gxtImSvj9VCIe8sgfMP5xioyiF+L4/mMMYsaHM"
    "wD830VOyZiqwP+Egf89C+rI/EdHW9uktsT/q3djypADHP6XxyQYc8tA/264xLz9Pyj8nv02fZjnOP0nXWw/ZDMw/"
    "JSqkauE30D+PVd7j4pGyP2fI0Png9MQ/gyX6oJx5zj9nVuob4VpWP6Py66voQ54/ZnCMpSSStj9dU1wqPniuPwQ9"
    "7igXbV4/Rk+4C+X7oj/M4rngYFW5P8+hAYGdc7E/c5XbbcfIsj9aJ9RTdd+JP8kC6lGjZJk/UJhVOcWJrj/B+D+D"
    "6XB2P66euuHTNrE/H1UdfgTQpD8BIMCJnFunP62K+r7ZlaM/gyN7JCT0mz/L2f9Hb3uBP7Gk4ldvFpU/loFUiaua"
    "Jj9OfJCXp4ChP0Li+8TKn0o/26p1b4fjWT8DhjMTS+W7PwXabapCY5w/OrkRNZwpuD90Kaauko2mP1IZdw1TNrM+"
    "HRJ5ihLNaj+QOl1jUgGZP3byNOiivZ0/FIfJIPHe/T6K8E0ACXelP81OX/FVsqo/joYKuk7nhD+chkbKFrOwP6xY"
    "NeaNm6I/vo9pKAWDsD/GO+/jO3eTPw37oBKadas/xuuk2ZTVtD9NYn3iSc2yP/YqaH6vgq4/AQHI3Y9Tvj81UAIz"
    "ZXS6P8KzBYYstqw/T/Yj6ofTwD+qjZ3XzvqlP2t/9KdLSLU/h+SMQex+nj+9Rz90ThubP0kSAScg1Lk/OZ90D1EJ"
    "zj/X7PzSmRC+P7mSiXJjx9E/wOXXNBSwwD8aQiAW7ebXP8fgPGFobdI/P92m/yUrwD/iRvwLPJzQP9J132tZhNk/"
    "P+Jrdlzv2T/icbGWFYjTP0YjjHcNvcw/EiIsD2Bm2z8tKfWD5TvbP7DzXDh36Mc/or5sPFFC2z/+/E0bTobgP6FF"
    "via7c8s/XQmc7Oap3D8UnpFiKS/NP2eNHA7YJNk/Hc6rkDad1j/mnV4DRIm5P5cP20NwPtY/8OPKw8LH2D/gI/Db"
    "GjzVP3neMpsSIMo/wagjYPMR2z/gccDJCfLgPyoWksUuZNU/SisSeTv2zz+N+b59yljYP0Ln2j3Lvtw/7K3GilOD"
    "2z+i/4+KTOvWPx0i3lJv48g/KtVRl7E9zD85ZAFhcrLWP37k6Nwq3dw/c8wo+H/3xz9BLE1j2vjaP3G9RqTaGeE/"
    "Fe6JjMf/1z/+xCADsyHUP1dXIfVBQtw/gpQ/+32Q3T/mtdvW9dTTP8/ZJrQ/O+M/P3ghTfMy0j/LsDI47RjlP2AH"
    "viGTZOY/kb+A54945z+UZ6d2THPgP3eaMMb5U9U/IEfnWKeX1z9UIj26jZfaP1fowhlNruQ//dS73cVrzD//bSYj"
    "zKraPyEaAfmuvc8/SHRImbWT0j/kNNDGGvvNP0p/EsngJ+I/QPpn0Fx14D/xHdjVjePLP7FBWsQpUds/s27urqBh"
    "2z9laXWSQiPYP0BQJtKczNk/07cg1tMN0T+rnV5/zNfSPxqQ/Ae31Nk/qP/lHTAn1j/xoLoi1IrcP+lsyxAqp+Q/"
    "GBYTPhVG2j9n+nZstt/YP1Jt38dabts/o3NvLtVn4z+fDVAP2CXWPyHq+GlKstA/8Fv36OHW4z+4A0fj0BndP2zy"
    "51iLgtU/zH6jgZRl2z8QnQS7pbHgP6GNPHM1YeE/TmbZiuqS2D+BcHD589XYP8XAP5gWeto/38hkqT9I4T9STfGU"
    "ES7SPyxp/iqH8eU/t6hPULgb3j+30odLsTXYP8MsVYc/x+E/kWApaEnT5z8vYOickKvoP0ttrBOrrtM/mLeGTlIY"
    "2j/mhsCbEu/nP72AxPd11uA/fWaFSBx/2j+2fJCyvHvgP0jwU0rwstQ/Q3m9VhMt5T+wBkJiOfnlP45N2jSNhOI/"
    "2AkyeUIm2D/RMy1xH87aP+4tDjCBU9c/ulBu98gb4j/iX/fbwnzjP9fTj0zmxtw/LkMGoyf/1j8JGoiYCxHhP6vc"
    "NIIVOeE/zIYeXyE44T/fgopWmpXgP/mqQq9pzNo/FNagpHip4D/0zghPVGfkP+AvHHGtO9c/X/D8hhY66D+CKoB9"
    "/+XSP5gIJRW8AOI/Tqh5xcNV5T/hB/dQkWzcP8O30rccn+M/q1KKX/Xf2T8aHkZMvyLQP+DVqTANLeI/3JRAJ2I1"
    "4j9K6xcMBbPhP/SsbcYDweA/2NSu+V5M5D+UAPqKcDzXP/s2oMXxmtg/T0kOKBg95j/cKH9x8ADlP/ijky3Gi90/"
    "J3DUt6eE5T9uUASPoozgP3wC47I6Eec/sJE8pdkq6D+BASGLWZDkP2bK4df60+Y/Nyg7cRGg2z9wGyF39wziP5g5"
    "1lbY8ec/zBy9thlx2j8cTjx09UDiP0UgDBCPZt0/8gEAbWG25D9YgzDYx3bkP65AnLWgcuQ/XzB95hkY6T+qKIAE"
    "IbHgP0lct1m+dOI/gO4sGXnD5j8Tmi2+egXiP9Lx0I0ho+Q/+wYsrUSO5j8JizRFy5jiP54Lirn/Sec/JovsiHsu"
    "5z9DQlYMpwfjP57/8+SYQOQ/oRao8XIg2j/uNx5kgLvkP8SVkQCwUek/7W9oIJG04j+Ks33KCDTiP9Nwsmkr6uE/"
    "L/grYPgl5T9X1zTnb3jjP0tCFx8Cgts/aHsmmwnm5T8b0FlTYerjPx+2gJUQSeI/e8jamMYL6D/7IdPYTD7rP3W8"
    "FdzUgt8/KIAm1Bcb6z9pZw569+bfPz0uF9SJW+I/WnHGKmNO6D9NOTfy5ZjoPzGDatQE9+M/0CMgMtw75D80OMH9"
    "qDblP2BwoDUrLeg//0aM02CP6D8ggeM8GGPmP1OHNBAew+g/Sm0LRq836j/Sd2vY+yTmP80UGPMUs+o/24xjJFvE"
    "5z8AAAAAAADwP/K32Pvsxe8/Bq99Nh1W6z8zOuplCF3vPxFP80TVQe0/TMF8UluL7j+e/z3wed3vP57vBR9s7e8/"
    "Me7Scd7g7z/Tmzi9MebvPwF/WfRI2O8/2JQ3GCX57z/AZVAUCTftP+5Syy9W/e8/B/OxpBXt7z+4+x16/v/vP1Mx"
    "XaFj/u8/AAAAAAAA8D+JQaCqs/LvPwAAAAAAAPA/V1DLdT7t7z8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D/OqDaN"
    "5ubvPwAAAAAAAPA/xFcRfvD+7z/E/zGx6P/vP1oBMOmW/+8/AAAAAAAA8D9aFETWg//vP06HToGp/O8/AAAAAAAA"
    "8D9X1+MFAP7vPwAAAAAAAPA/0zO9d0r/7z8AAAAAAADwPwAAAAAAAPA/NZd5Luv+7z+D3NTfKv7vPxsagLvZ/+8/"
    "AAAAAAAA8D8I3A2v4v/vPwAAAAAAAPA/AAAAAAAA8D+vaKx18P/vP9qsXOPv/+8/"
)


def _embedded_warm_start_n600() -> np.ndarray:
    raw = base64.b64decode(_WARM_START_N600_B64)
    h0 = np.frombuffer(raw, dtype=np.float64).copy()
    if h0.shape != (600,):
        raise RuntimeError(f"embedded warm start has wrong shape {h0.shape}")
    return h0


# EVOLVE-BLOCK-START
"""Evolve this part: the starting construction and/or the polish routine."""


def load_starting_heights(n):
    """
    Load our best available starting construction. This is a SELF-DERIVED
    warm start (multi-restart local optimization, see make_warm_start.py),
    NOT the literal published record - we could not recover Haugland's or
    AlphaEvolve's actual numeric heights from the source papers.
    """
    h0 = _embedded_warm_start_n600()
    if n == len(h0):
        return h0
    # Other resolutions: interpolate the embedded warm start.
    old_x = (np.arange(len(h0)) + 0.5) * (2.0 / len(h0))
    new_x = (np.arange(n) + 0.5) * (2.0 / n)
    return np.clip(np.interp(new_x, old_x, h0), 0.0, 1.0)


def polish(h0, n):
    """
    Local refinement via SLSQP directly on the true (non-smoothed) max.
    This is deliberately simple - Setup B is where a smarter, smoothed
    objective and better handling of tied maximizing shifts belongs.
    Evolution here should focus on things like: better initial guesses,
    multiple restarts kept within the per-eval time budget, tweaks to
    solver tolerances, or simple post-processing of the polished result.
    """
    w = 2.0 / n

    def objective(h_free):
        return _worst_case_overlap(np.clip(h_free, 0.0, 1.0))

    cons = [{"type": "eq", "fun": lambda h_free: np.sum(h_free) * w - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        objective, h0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 150, "ftol": 1e-12},
    )
    return np.clip(res.x, 0.0, 1.0)


def construct_heights():
    n = 600
    h0 = load_starting_heights(n)
    h = polish(h0, n)
    return h, n


# EVOLVE-BLOCK-END


def run_setup_a():
    """Fixed entry point. Normalizes and reports the final construction."""
    h, n = construct_heights()
    w = 2.0 / n
    h = np.clip(h, 0.0, 1.0)
    total = np.sum(h) * w
    if total > 1e-9:
        h = np.clip(h / total, 0.0, 1.0)
    c_max = _worst_case_overlap(h)
    return h, c_max, n
