import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm, trange

from utility_funcs.cfg_loader import load_yaml
from utility_funcs.load_model import load_model
from dynamical_models.lti_system import simulate_lti_system
from dynamical_models.drone import simulate_drone
from datasources.normalization import denorm_y

from estimators.KF import KalmanFilter
from estimators.EKF import ExtendedKalmanFilter

from dynamical_models.drone_functions import drone_state_transition_function, drone_measurement_function


def test():
    # 1) Read test config
    cfg = load_yaml("configs/test.yaml")
    system          = cfg["system"]                  # "lti" | "drone"
    n_traj          = int(cfg["n_traj"])
    H               = int(cfg["H"])                  # eval horizon
    L_test          = cfg.get("L", None)             # int or None (full-context)
    model_run_dirs  = [Path(p) for p in cfg["model_run_dirs"]]
    save_dir        = Path(cfg["save_dir"])

    # 2) Load sim config
    if system == "lti":
        sim_cfg = load_yaml(f"configs/{system}_system.yaml")
    else:  # "drone"
        sim_cfg = load_yaml(f"configs/{system}.yaml")

    # 3) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4) Monte Carlo trajectories (shared for fairness)
    trajs = []
    for _ in trange(n_traj, desc="Simulating trajectories"):
        traj = simulate_lti_system(sim_cfg) if system == "lti" else simulate_drone(sim_cfg)
        trajs.append(traj)

    # 5) Baseline SE over trajectories: (A, H, p)
    baseline_se = []
    for traj in tqdm(trajs, desc="Baseline (KF/EKF) over trajectories"):
        base_se_ahp = baseline(traj, sim_cfg, system, H)   # (A, H, p)
        baseline_se.append(base_se_ahp)
    baseline_mse_time = np.stack(baseline_se, axis=0).mean(axis=0).mean(axis=-1)  # (A, H)

    # 6) Evaluate each model separately and collect results
    mse_time_by_model = []
    for rd in tqdm(model_run_dirs, desc="Evaluating models"):
        model, meta = load_model(rd, device=device)  # loader should NOT call eval()
        model.eval()

        # per-trajectory SE for this model → aggregate to MSE
        name_for_bar = meta.get("type", rd.name)
        se_by_traj = []
        for traj in tqdm(trajs, desc=f"Trajs for {name_for_bar}", leave=False):
            y = traj["measurements"].astype(np.float32)     # (T+1, p)
            u = traj.get("inputs", None)
            if u is not None:
                u = u.astype(np.float32)                    # (T, m)

            se_ahp = eval_model(model, meta, y, u, H, L_test)  # (A, H, p)
            se_by_traj.append(se_ahp)

        mse_time = np.stack(se_by_traj, axis=0).mean(axis=0).mean(axis=-1)  # (A, H)

        mse_time_by_model.append({
            "name": meta["type"],
            "run_dir": str(rd),
            "mse_time_h": mse_time.tolist(),  # (A, H) 
        })

    # 7) Save
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "system": system,
        "H": H,
        "L": (None if L_test is None else int(L_test)),
        "n_traj": n_traj,
        "baseline": {
            "kind": ("KF" if system == "lti" else "EKF"),
            "mse_time_h": baseline_mse_time.tolist(),  # (A, H) 
        },
    }
    for model_res in mse_time_by_model:
        out["model_" + model_res["name"]] = model_res
    out_path = save_dir / f"{system}_test_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"test saved: {out_path}")


def eval_model(model, meta, y_raw, u_raw, H_eval, L_test):
    T = y_raw.shape[0] - 1
    p = y_raw.shape[-1]
    anchors = range(1, T - H_eval + 2)
    A = T - H_eval + 1

    stats = model.norm_stats
    dev = next(model.parameters()).device  # ensure inputs go to same device as the model

    se_ahp = np.empty((A, H_eval, p), dtype=np.float32)

    for idx, t0 in enumerate(anchors):
        if L_test is None:
            # Full context up to t0
            y_hist = y_raw[0:t0, :]
            y_hist_n = (y_hist - stats["mu_y"]) / stats["std_y"]

            u_hist_n = None
            if u_raw is not None:
                u_hist = u_raw[0:t0, :]
                u_hist_n = (u_hist - stats["mu_u"]) / stats["std_u"]

            mask = np.ones(y_hist_n.shape[0], dtype=np.int64)
        else:
            # Fixed L window with left padding
            L = int(L_test)
            start = max(0, t0 - L)
            k = t0 - start          # time reversaL checks

            # --- measurements ---
            y_ctx = y_raw[start:t0, :]                              # (k, p)
            y_ctx_n = (y_ctx - stats["mu_y"]) / stats["std_y"]      # normalize first

            if k < L:
                pad_n = np.zeros((L - k, p), dtype=np.float32)      # zeros in normalized space
                y_hist_n = np.concatenate([pad_n, y_ctx_n], axis=0) # (L, p)
                mask = np.concatenate([np.zeros(L - k, dtype=np.int64),
                                    np.ones(k, dtype=np.int64)], axis=0)
            else:
                y_hist_n = y_ctx_n
                mask = np.ones(L, dtype=np.int64)

            # --- inputs (if any) ---
            u_hist_n = None
            if u_raw is not None:
                m = u_raw.shape[-1]
                u_ctx = u_raw[start:t0, :]                          # (k, m)
                u_ctx_n = (u_ctx - stats["mu_u"]) / stats["std_u"]
                if k < L:
                    upad_n = np.zeros((L - k, m), dtype=np.float32) # zeros in normalized space
                    u_hist_n = np.concatenate([upad_n, u_ctx_n], axis=0)
                else:
                    u_hist_n = u_ctx_n


        # Tensors to model device
        y_t = torch.from_numpy(y_hist_n).float().unsqueeze(0).to(dev)  # (1, L*, p)
        u_t = None if u_hist_n is None else torch.from_numpy(u_hist_n).float().unsqueeze(0).to(dev)
        m_t = torch.from_numpy(mask).long().unsqueeze(0).to(dev)       # (1, L*)

        with torch.no_grad():
            pred = model.backbone(y_t, u_t, attn_mask=m_t)             # (1, L*, H, p)
            pred_last = pred[:, -1, :, :].squeeze(0).cpu().numpy()     # (H, p)
        pred_last = pred_last[:H_eval]

        pred_den = denorm_y(pred_last, stats)                          # (H, p)
        tgt = y_raw[t0:t0 + H_eval, :]                                  # (H, p)
        se_ahp[idx] = (pred_den - tgt) ** 2

    return se_ahp


def baseline(traj, sim_cfg, system, H_eval):
    y = traj["measurements"].astype(np.float32)  # (T+1, p)
    u = traj.get("inputs", None)
    if u is not None:
        u = u.astype(np.float32)

    T = y.shape[0] - 1
    p = y.shape[-1]
    anchors = range(1, T - H_eval + 2)
    A = T - H_eval + 1

    if system == "lti":
        A_mat = np.asarray(sim_cfg["A"], dtype=float)
        B_mat = np.asarray(sim_cfg["B"], dtype=float) if bool(sim_cfg["input_enabled"]) else None
        C_mat = np.asarray(sim_cfg["C"], dtype=float)

        kf = KalmanFilter(
            A=A_mat, C=C_mat,
            initial_state_mean=sim_cfg["initial_state_mean"],
            initial_state_cov=sim_cfg["initial_state_cov"],
            process_noise_mean=sim_cfg["process_noise_mean"],
            process_noise_cov=sim_cfg["process_noise_cov"],
            measurement_noise_mean=sim_cfg["measurement_noise_mean"],
            measurement_noise_cov=sim_cfg["measurement_noise_cov"],
            B=B_mat,
            input_enabled=bool(sim_cfg["input_enabled"]),
        )
        kf.filter(n_steps=T, measurement_sequence=y, input_sequence=u)

        se_ahp = np.empty((A, H_eval, p), dtype=np.float32)
        for idx, t0 in enumerate(anchors):
            '''x_t = kf.estimated_states[t0].copy()
            fut_u = u[t0:t0 + H_eval] if (u is not None) else None
            y_fore, _ = kf.predict_next_H_outputs(H=H_eval, last_state=x_t, future_inputs=fut_u)  # (H, p)
            tgt = y[t0:t0 + H_eval, :]
            se_ahp[idx] = (y_fore - tgt) ** 2'''
            x_t = kf.estimated_states[t0 - 1].copy()
            fut_u = u[t0-1 : t0-1 + H_eval] if (u is not None) else None
            y_fore, _ = kf.predict_next_H_outputs(H=H_eval, last_state=x_t, future_inputs=fut_u)
            tgt = y[t0:t0 + H_eval, :]
            se_ahp[idx] = (y_fore - tgt) ** 2
        return se_ahp

    else:  # "drone"
        ekf = ExtendedKalmanFilter(
            f=drone_state_transition_function,
            h=drone_measurement_function,
            initial_state_mean=sim_cfg["initial_state_mean"],
            initial_state_cov=sim_cfg["initial_state_cov"],
            process_noise_mean=sim_cfg["process_noise_mean"],
            process_noise_cov=sim_cfg["process_noise_cov"],
            measurement_noise_mean=sim_cfg["measurement_noise_mean"],
            measurement_noise_cov=sim_cfg["measurement_noise_cov"],
            input_enabled=bool(sim_cfg["input_enabled"]),
        )

        ekf.filter(n_steps=T, measurement_sequence=y, input_sequence=u)
        se_ahp = np.empty((A, H_eval, p), dtype=np.float32)
        for idx, t0 in enumerate(anchors):
            x_t = ekf.estimated_states[t0 - 1].copy()
            fut_u = u[t0-1 : t0-1 + H_eval] if (u is not None) else None
            fut_u = np.zeros((H_eval, fut_u.shape[-1]), dtype=np.float32) if fut_u is not None else None # delete later
            y_fore, _ = ekf.predict_next_H_outputs(H=H_eval, last_state=x_t, future_inputs=fut_u)
            tgt = y[t0:t0 + H_eval, :]
            se_ahp[idx] = (y_fore - tgt) ** 2
        return se_ahp


if __name__ == "__main__":
    test()
