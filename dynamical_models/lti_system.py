import numpy as np
from utility_funcs.cfg_loader import load_yaml
from utility_funcs.generate_input_sequence import generate_input_sequence
from utility_funcs.create_random_matrices import generate_random_ABC

class LTI_System:
    def __init__(self, initial_state):
        super().__init__()

        self.state = np.asarray(initial_state, dtype=float).reshape(-1)

    def __repr__(self):
        return ", ".join([f"x{i}: {xi:.3f}" for i, xi in enumerate(self.state)])
    
    def _check_shapes(self, A=None, B=None, u=None, process_noise_mean=None, process_noise_cov=None, C=None, measurement_noise_mean=None, measurement_noise_cov=None):
        """
        Ensures shapes are consistent with state dimension.
        """
        n = int(self.state.size)
        if A is not None:
            A = np.asarray(A, dtype=float)
            if A.ndim != 2 or A.shape != (n, n):
                raise ValueError(f"A must be shape ({n},{n}), got {A.shape}")
            
        m = None
        if u is not None:
            u = np.asarray(u, dtype=float).reshape(-1)
            m = int(u.size)

        if B is not None:
            B = np.asarray(B, dtype=float)
            if m is None:
                m = B.shape[1]               # infer from B if u is None
            if B.ndim != 2 or B.shape != (n, m):
                raise ValueError(f"B must be shape ({n},{m}), got {B.shape}")

        if process_noise_mean is not None:
            w_mean = np.asarray(process_noise_mean, dtype=float).reshape(-1)
            if w_mean.shape != (n,):
                raise ValueError(f"process_noise_mean must be ({n},), got {w_mean.shape}")
        if process_noise_cov is not None:
            w_cov = np.asarray(process_noise_cov, dtype=float)
            if w_cov.ndim != 2 or w_cov.shape != (n, n):
                raise ValueError(f"process_noise_cov must be ({n},{n}), got {w_cov.shape}")
            
        p = None
        if C is not None:
            C = np.asarray(C, dtype=float)
            if C.ndim != 2 or C.shape[1] != n:
                raise ValueError(f"C must be shape (p,{n}), got {C.shape}")
            p = int(C.shape[0])

        if measurement_noise_mean is not None:
            v_mean = np.asarray(measurement_noise_mean, dtype=float).reshape(-1)
            if v_mean.shape != (p,):
                raise ValueError(f"measurement_noise_mean must be ({p},), got {v_mean.shape}")
        
        if measurement_noise_cov is not None:
            v_cov = np.asarray(measurement_noise_cov, dtype=float)
            if v_cov.ndim != 2 or v_cov.shape != (p, p):
                raise ValueError(f"measurement_noise_cov must be ({p},{p}), got {v_cov.shape}")

    
    def state_transition(self, A, B, u, process_noise_mean, process_noise_cov):
        """
        A: shape (n,n)
        B: shape (n,m) or None
        u: control vector shape (m,) or None
        process_noise_mean: shape (n,)
        process_noise_cov: shape (n,n)
        """
        self._check_shapes(A=A, B=B, u=u, process_noise_mean=process_noise_mean, process_noise_cov=process_noise_cov)
        
        process_noise = np.random.multivariate_normal(
            mean=np.asarray(process_noise_mean, dtype=float).reshape(-1),
            cov=np.asarray(process_noise_cov, dtype=float))

        if B is not None and u is not None:
            self.state = A @ self.state + B @ u + process_noise
        else:
            self.state = A @ self.state + process_noise
        
    def measurement_function(self, C, measurement_noise_mean, measurement_noise_cov):
        """
        C: measurement matrix (p,n)
        measurement_noise_mean: shape (p,)
        measurement_noise_cov: shape (p,p)
        returns y: shape (p,)
        """

        self._check_shapes(C=C, measurement_noise_mean=measurement_noise_mean, measurement_noise_cov=measurement_noise_cov)
        
        v = np.random.multivariate_normal(
            mean=np.asarray(measurement_noise_mean, dtype=float).reshape(-1),
            cov=np.asarray(measurement_noise_cov, dtype=float))
        
        return C @ self.state + v

def _simulate_lti_system(n_steps, A, B, C, initial_state_mean, initial_state_cov,
                            process_noise_mean, process_noise_cov,
                            measurement_noise_mean, measurement_noise_cov, umin, umax, input_enabled=True):
        """
        Returns a dict:
        {
            "states": (n_steps+1, n),
            "measurements": (n_steps+1, p)
            "inputs": (n_steps, m) or None
        }
        """
        # Initialize system
        initial_state = np.random.multivariate_normal(
            mean=np.asarray(initial_state_mean, dtype=float).reshape(-1),
            cov=np.asarray(initial_state_cov, dtype=float))
                                        
        lti_sys = LTI_System(initial_state=initial_state)
    
        # Preallocate
        n = A.shape[0]
        p = C.shape[0]
        states = np.empty((n_steps + 1, n), dtype=float)
        measurements = np.empty((n_steps + 1, p), dtype=float)

        if input_enabled and B is None:
            raise ValueError("input_enabled=True requires B.")

        if input_enabled:
            m = B.shape[1]
            input_sequence = generate_input_sequence(
                n_steps,
                m,
                umin=umin,
                umax=umax,
                seed=None)

        # t = 0
        states[0] = lti_sys.state
        measurements[0] = lti_sys.measurement_function(
            C, measurement_noise_mean, measurement_noise_cov)
    
        # Steps k = 0...n_steps - 1
        for k in range(n_steps):
            lti_sys.state_transition(A=A, B=B if input_enabled else None,
                u=input_sequence[k] if input_enabled else None,
                process_noise_mean=process_noise_mean,
                process_noise_cov=process_noise_cov)
            states[k + 1] = lti_sys.state.copy()
            measurements[k + 1] = lti_sys.measurement_function(
                C, measurement_noise_mean, measurement_noise_cov)
    
        return {
            "states": states,
            "measurements": measurements,
            "inputs": input_sequence if input_enabled else None
        }

def simulate_lti_system(cfg):
    if cfg["randomized_matrices"]:
        A, B, C = generate_random_ABC(
            n=int(cfg["state_dimension"]),
            m=int(cfg["input_dimension"]),
            p=int(cfg["output_dimension"]),
            dist=cfg.get("matrix_dist", "gaussian"),
            target_rho=float(cfg.get("target_rho", 0.95)),
            seed=cfg.get("random_seed", None),
            max_tries_A=int(cfg.get("max_tries_A", 64)),
            max_tries_BC=int(cfg.get("max_tries_BC", 256)),
        )
    else: A, B, C = (np.asarray(cfg["A"], dtype=float),
                    np.asarray(cfg["B"], dtype=float) if bool(cfg["input_enabled"]) else None,
                    np.asarray(cfg["C"], dtype=float))

    return _simulate_lti_system(
        n_steps=int(cfg["n_steps"]),
        A=A,
        B=B,
        C=C,
        initial_state_mean=cfg["initial_state_mean"],
        initial_state_cov=cfg["initial_state_cov"],
        process_noise_mean=cfg["process_noise_mean"],
        process_noise_cov=cfg["process_noise_cov"],
        measurement_noise_mean=cfg["measurement_noise_mean"],
        measurement_noise_cov=cfg["measurement_noise_cov"],
        umin=float(cfg["umin"]),
        umax=float(cfg["umax"]),
        input_enabled=bool(cfg["input_enabled"]),
    )

