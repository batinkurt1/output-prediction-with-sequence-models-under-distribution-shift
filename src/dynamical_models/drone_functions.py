import numpy as np
from utility_funcs.cfg_loader import load_yaml

def _wrap_angle(phi: float) -> float:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi

# ---------------- dynamics / measurement ----------------
def drone_state_transition_function(x, u):
    cfg = load_yaml("./configs/drone.yaml")  

    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    m  = float(cfg["m"])
    l  = float(cfg["l"])
    J  = float(cfg["J"])
    g  = float(cfg["g"])
    dt = float(cfg["dt"])

    STATE_ORDER = ["x", "z", "phi", "vx", "vz", "phidot"]          
    mins = np.array([cfg[f"{k}min"] for k in STATE_ORDER], dtype=float)
    maxs = np.array([cfg[f"{k}max"] for k in STATE_ORDER], dtype=float)
    limits = np.vstack([mins, maxs]) 

    X, Z, PHI, VX, VZ, PHIDOT = x

    # drift f(x)
    f = np.array([
        VX * np.cos(PHI) - VZ * np.sin(PHI),   # x_dot
        VX * np.sin(PHI) + VZ * np.cos(PHI),   # z_dot
        PHIDOT,                                 # phi_dot
        VZ * PHIDOT - g * np.sin(PHI),          # vx_dot
        -VX * PHIDOT - g * np.cos(PHI),         # vz_dot
        0.0,                                    # phiddot
    ], dtype=float)

    # input matrix B
    B = np.array([
        [0.0,   0.0],
        [0.0,   0.0],
        [0.0,   0.0],
        [0.0,   0.0],
        [1.0/m, 1.0/m],
        [ l/J, -l/J],
    ], dtype=float)

    process_noise_mean = np.asarray(cfg["process_noise_mean"], dtype=float).reshape(-1)

    x_next = x + (f + B @ u) * dt + process_noise_mean
    x_next[2] = _wrap_angle(x_next[2])  # wrap phi


    if cfg["clip_state"]:
        limits = np.asarray(limits, dtype=float)  # shape (2,6)
        clip_idx = np.array([2, 3, 4, 5], dtype=int)      # phi, vx, vz, phidot
        x_next[clip_idx] = np.clip(x_next[clip_idx], limits[0, clip_idx], limits[1, clip_idx])

    return x_next

def drone_measurement_function(x):
    cfg = load_yaml("./configs/drone.yaml")
    x = np.asarray(x, dtype=float).reshape(-1)
    C = cfg["C"]
    measurement_noise_mean = np.asarray(cfg["measurement_noise_mean"], dtype=float).reshape(-1)
    if C is None:
        return x.copy()
    C = np.asarray(C, dtype=float)
    return C @ x + measurement_noise_mean
