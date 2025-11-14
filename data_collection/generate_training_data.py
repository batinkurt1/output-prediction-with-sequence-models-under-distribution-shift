import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from dynamical_models.drone import simulate_drone
from dynamical_models.lti_system import simulate_lti_system
from utility_funcs.cfg_loader import load_yaml
from utility_funcs.save_pickle import save_pickle


def simulate_system(system, n_traj):
    if system == "drone":
        cfg_path = "configs/drone.yaml"
        sim_fn = simulate_drone
    elif system == "lti":
        cfg_path = "configs/lti_system.yaml"
        sim_fn = simulate_lti_system
    else:
        raise ValueError("system must be 'drone' or 'lti'")

    cfg = load_yaml(cfg_path)

    traj = []
    for _ in tqdm(range(n_traj), desc=f"Simulating {system}", unit="traj", dynamic_ncols=True):
        sample = sim_fn(cfg)
        traj.append(sample)

    states       = np.stack([t["states"]       for t in traj], axis=0)
    measurements = np.stack([t["measurements"] for t in traj], axis=0)
    inputs0 = traj[0].get("inputs", None)
    inputs  = None if inputs0 is None else np.stack([t["inputs"] for t in traj], axis=0)

    return {
        "system": system,
        "n_traj": n_traj,
        "states": states,
        "measurements": measurements,
        "inputs": inputs,
        "meta": {"config": cfg},
    }

def save_results(results, outputs_dir):
    sys_name = results["system"]
    n_traj   = results["n_traj"]
    n_steps  = int(results["meta"]["config"]["n_steps"])

    input_enabled = results["inputs"] is not None
    cfg = results["meta"]["config"]
    randomized = False if sys_name == "drone" else bool(cfg["randomized_matrices"])

    u_tag = "input" if input_enabled else "no_input"
    r_tag = "random_matrices" if randomized else "fixed_matrices"

    run_dir = Path(outputs_dir) / sys_name / f"N{n_traj}-S{n_steps}-{u_tag}-{r_tag}"
    out_path = run_dir / f"{sys_name}_dataset.pkl"
    saved = save_pickle(results, out_path)
    tqdm.write(f"Saved: {saved}")

def generate_training_data():
    cfg = load_yaml("configs/generate_data.yaml")
    system      = cfg["system"]        # "drone" | "lti" | "both"
    n_traj      = cfg["n_traj"]
    outputs_dir = cfg["outputs_dir"]

    if system in ("drone", "lti"):
        results = simulate_system(system, n_traj)
        save_results(results, outputs_dir)
    elif system == "both":
        for sys_name in ("drone", "lti"):
            results = simulate_system(sys_name, n_traj)
            save_results(results, outputs_dir)
