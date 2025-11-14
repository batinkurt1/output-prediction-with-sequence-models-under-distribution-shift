import numpy as np

from utility_funcs.generate_input_sequence import generate_input_sequence

class Drone:
    def __init__(self, initial_state, m, l, J, g, limits, clip_state):
        super().__init__()

        # Constants
        self.m = float(m)
        self.l = float(l)
        self.J = float(J)
        self.g = float(g)
        
        # Limits
        self.limits = np.asarray(limits, dtype=float)

        # Indices for clipping
        self.STATE_ORDER = ("x", "z", "phi", "vx", "vz", "phidot")
        self.CLIP_IDX = np.array([i for i, k in enumerate(self.STATE_ORDER) if k in {"phi","vx","vz","phidot"}], dtype=int)

        self.clip_state = clip_state

        # Initial state
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        if initial_state.size != 6:
            raise ValueError(f"initial_state must have 6 elements, got {initial_state.shape}")
        self.state = np.clip(initial_state, self.limits[0], self.limits[1])

    def __repr__(self):
        x, z, phi, vx, vz, phidot = self.state
        return (f"x: {x:.3f}, z: {z:.3f}, phi: {phi:.3f}, "
                f"vx: {vx:.3f}, vz: {vz:.3f}, phidot: {phidot:.3f}")

    def state_transition(self, u, dt, process_noise_mean, process_noise_cov):
        """
        u: control vector shape (2,) [thrusts u1, u2]
        dt: float (seconds)
        process_noise_mean: shape (6,)
        process_noise_cov: shape (6,6)
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != 2:
            raise ValueError("u must have 2 elements (u1, u2)")
        w = np.random.multivariate_normal(
            mean=np.asarray(process_noise_mean, dtype=float).reshape(-1),
            cov=np.asarray(process_noise_cov, dtype=float)) 

        x, z, phi, vx, vz, phidot = self.state

        # Nonlinear dynamics f
        f = np.array([
            vx * np.cos(phi) - vz * np.sin(phi),            
            vx * np.sin(phi) + vz * np.cos(phi),           
            phidot,                                         
            vz * phidot - self.g * np.sin(phi),             
            -vx * phidot - self.g * np.cos(phi),            
            0.0,                                            
        ], dtype=float)

        # Input matrix B
        B = np.array([
            [0.0,             0.0],     
            [0.0,             0.0],     
            [0.0,             0.0],       
            [0.0,          0.0],        
            [1.0/self.m,  1.0/self.m],
            [ self.l/self.J, -self.l/self.J],
        ], dtype=float) 

        next_state = self.state + (f + B @ u) * float(dt) + w

        next_state[2] = _wrap_angle(next_state[2])  # wrap phi

        if self.clip_state:
            next_state[self.CLIP_IDX] = np.clip(
                next_state[self.CLIP_IDX],
                self.limits[0, self.CLIP_IDX],
                self.limits[1, self.CLIP_IDX],
            )

        self.state = next_state

    def measurement_function(self, C, measurement_noise_mean, measurement_noise_cov):
        """
        C: measurement matrix (p,6)
        returns y: shape (p,)
        """
        C = np.asarray(C, dtype=float)
        v = np.random.multivariate_normal(
            mean=np.asarray(measurement_noise_mean, dtype=float).reshape(-1),
            cov=np.asarray(measurement_noise_cov, dtype=float),
        )
        return C @ self.state + v  

def _wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def _simulate_drone(n_steps, dt, umin, umax, 
                   initial_state_mean, initial_state_cov,
                   process_noise_mean, process_noise_cov,
                   measurement_noise_mean, measurement_noise_cov,
                   C, m, l, J, g, limits, clip_state=False):
    """
    Returns a dict:
      {
        "states": (n_steps+1, 6),        
        "measurements": (n_steps+1, p),
        "inputs": (n_steps, 2) 
      }
    """
    input_sequence = generate_input_sequence(
        n_steps,
        2,
        umin,
        umax,
        seed=None)
    
    initial_state = np.random.multivariate_normal(
        mean=np.asarray(initial_state_mean, dtype=float).reshape(-1),
        cov=np.asarray(initial_state_cov, dtype=float))
                                       
    # Initialize drone
    drone = Drone(
        initial_state=initial_state,
        m=m,
        l=l,
        J=J,
        g=g,
        limits=limits,
        clip_state=clip_state,
    )

    # Preallocate
    p = C.shape[0]
    states = np.empty((n_steps + 1, 6), dtype=float)
    measurements = np.empty((n_steps + 1, p), dtype=float)

    # t = 0
    states[0] = drone.state
    measurements[0] = drone.measurement_function(
        C, measurement_noise_mean, measurement_noise_cov)

    # Steps k = 0...n_steps - 1
    for k in range(n_steps):
        u = input_sequence[k]
        drone.state_transition(u, dt, process_noise_mean, process_noise_cov)
        states[k + 1] = drone.state.copy()
        measurements[k + 1] = drone.measurement_function(
            C, measurement_noise_mean, measurement_noise_cov)

    return {
        "states": states,
        "measurements": measurements,
        "inputs": input_sequence}


def simulate_drone(cfg):
        STATE_ORDER = ["x", "z", "phi", "vx", "vz", "phidot"]          
        mins = np.array([cfg[f"{k}min"] for k in STATE_ORDER], dtype=float)
        maxs = np.array([cfg[f"{k}max"] for k in STATE_ORDER], dtype=float)
        limits = np.vstack([mins, maxs]) 
        return _simulate_drone(
            n_steps=int(cfg["n_steps"]),
            dt=float(cfg["dt"]),
            umin=float(cfg["umin"]),
            umax=float(cfg["umax"]),
            initial_state_mean=cfg["initial_state_mean"],
            initial_state_cov=cfg["initial_state_cov"],
            process_noise_mean=cfg["process_noise_mean"],
            process_noise_cov=cfg["process_noise_cov"],
            measurement_noise_mean=cfg["measurement_noise_mean"],
            measurement_noise_cov=cfg["measurement_noise_cov"],
            C=np.asarray(cfg["C"], dtype=float),
            m=float(cfg["m"]),
            l=float(cfg["l"]),
            J=float(cfg["J"]),
            g=float(cfg["g"]),
            limits=limits,
            clip_state=bool(cfg["clip_state"]))