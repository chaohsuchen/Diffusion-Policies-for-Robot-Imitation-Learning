import numpy as np
import scipy.interpolate
import pickle


class DMP(object):
    """
        Dynamic Movement Primitives wlearned by Locally Weighted Regression (LWR).

    Implementation of P. Pastor, H. Hoffmann, T. Asfour and S. Schaal, "Learning and generalization of
    motor skills by learning from demonstration," 2009 IEEE International Conference on Robotics and
    Automation, 2009, pp. 763-768, doi: 10.1109/ROBOT.2009.5152385.
    """

    def __init__(self, nbasis=30, K_vec=10 * np.ones((6,)), weights=None):
        self.nbasis = nbasis  # Basis function number
        self.K_vec = K_vec

        self.K = np.diag(self.K_vec)  # Spring constant
        self.D = np.diag(2 * np.sqrt(self.K_vec))  # Damping constant, critically damped

        # used to determine the cutoff for s
        self.convergence_rate = 0.01
        self.alpha = -np.log(self.convergence_rate)

        # Creating basis functions and psi_matrix
        # Centers logarithmically distributed between 0.001 and 1
        self.basis_centers = np.logspace(-3, 0, num=self.nbasis)
        self.basis_variances = self.nbasis / (self.basis_centers**2)

        self.weights = weights

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                dict(
                    nbasis=self.nbasis,
                    K_vec=self.K_vec,
                    weights=self.weights,
                ),
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        dmp = cls(nbasis=data["nbasis"], K_vec=data["K_vec"], weights=data["weights"])
        return dmp

    @staticmethod
    def _gradient_nd(x, t):
        g1 = (x[:, 1:, :] - x[:, :-1, :]) / (t[:, 1:, None] - t[:, :-1, None])
        g = np.zeros_like(x)
        g[:, 0, :] = g1[:, 0, :]
        g[:, -1, :] = g1[:, -1, :]
        g[:, 1:-1, :] = (g1[:, 1:, :] + g1[:, :-1, :]) / 2
        return g

    def learn(self, X, T):
        """
        Learn the weights of the DMP using Locally Weighted Regression.

        X: demonstrated trajectories. Has shape [number of demos, number of timesteps,  dofs].
        T: corresponding timings. Has shape [number of demos, number of timesteps].
            It is assumed that trajectories start at t=0
        """
        #
        num_demos = X.shape[0]
        num_timesteps = X.shape[1]
        num_dofs = X.shape[2]

        # Initial position : [num_demos, num_timesteps, num_dofs]
        x0 = np.tile(X[:, 0, :][:, None, :], (1, num_timesteps, 1))
        # Goal position : [num_demos, num_timesteps, num_dofs]
        g = np.tile(X[:, -1, :][:, None, :], (1, num_timesteps, 1))
        # Duration of the demonstrations
        tau = T[:, -1]
        # We know T.shape = (demos, time)
        # : select all demos, -1 select each demo's last time point
        # Note: 
        # X.shape                    # (demos, time, dofs)
        # X[:, 0, :].shape           # (demos, dofs)      → First position
        # X[:, 0, :][:, None, :].shape # (demos, 1, dofs) → Add time dimension

        # TODO: Compute s(t) for each step in the demonstrations
        # Derivation of Phase Variable s(t) from the Canonical System:
        # 1. Start with the canonical differential equation (Eq. 4 in the paper):
        #    tau * (ds / dt) = -alpha * s
        # 2. Separate variables:
        #    (1 / s) * ds = (-alpha / tau) * dt
        # 3. Integrate both sides from t = 0 (where s(0) = 1) to time T:
        #    ln(s) = -alpha * T / tau
        # 4. Exponentiate both sides to solve for s:
        #    s = exp(-alpha * T / tau)
        
        # tau.shape = (demos,) after [:, None] = > (demos, 1)
        # T:               (demos, time)
        # tau[:, None]:    (demos, 1)
        # s:               (demos, time)
        s = np.exp(-self.alpha * T / tau[:, None])

        # TODO: Compute x_dot and x_ddot using numerical differentiation (np.graident)
        # paper: the corresponding velocities and accelerations for all DOF 
        # were computed numerically by differentiating the position signal.
        # X.shape = (demos, time, dofs) X[i].shape = (time, dofs)
        
        # for each gradient：
        # X[i]:   (time, joints)
        # T[i]:   (time,)
        # output：  (time, joints)
        # after stacking：
        # (demos, time, joints)
        
        x_dot = np.stack([
            np.gradient(X[i], T[i], axis=0, edge_order=2)
            for i in range(num_demos)
        ])
        x_ddot = np.stack([
            np.gradient(x_dot[i], T[i], axis=0, edge_order=2)
            for i in range(num_demos)
        ])
        # (demos, time, joints)

        # TODO: Temporal Scaling by tau.
        # paper: In the transformation system equations (such as Eq. 1 or Eq. 6), 
        # the time derivative term on the left side is scaled by $\tau$: $\tau \dot{v} = \dots$
        # v = tau * x_dot
        # x_dot has dimensions like (num_demos, num_timesteps, num_dims) (a 3D array), reshaping tau 
        # into (num_demos, 1, 1) allows NumPy to correctly multiply each demonstration's time-scaling factor 
        # across all of its time steps and spatial dimensions simultaneously.
        tau_3d = tau[:, None, None] # tau3d.shape = (demos, 1, 1)  x_dot.shape = (demos, time, dofs)
        v = tau_3d * x_dot
        v_dot = tau_3d * x_ddot

        # TODO: Compute f_target(s) based on Equation 8.
        # tau_3d: (demos, 1, 1), v_dot: (demos, time, dofs), self.D: (dofs, dofs), K_vec: (dofs,), 
        # g: (demos, time, dofs), X: (demos, time, dofs), x0: (demos, time, dofs).
        # s: (demos, time) → s[..., None]: (demos, time, 1).
        
        # K_vec = 10 * np.ones((6,))   # (dofs,) =>  because np.ones((6,)) creates a 1D array with 6 elemen
        # self.D = np.diag(2 * np.sqrt(K_vec))  # (dofs, dofs) => because np.diag() converts a 1D array into a square diagonal matrix
        
        f_s_target = (
            (tau_3d * v_dot + v @ self.D.T) / self.K_vec
            - (g - X)
            + (g - x0) * s[..., None]
        )

        # TODO: Compute psi(s). Hint: shape should be [num_demos, num_timesteps, nbasis]
        # Paper: Gaussian basis functions, with center ci and width hi, and wi are adjustable weights
        psi = np.exp(
            -self.basis_variances
            * (s[..., None] - self.basis_centers) ** 2
        )

        # TODO: Solve a least squares problem for the weights.
        # Hint: minimize f_target(s) - f_w(s) wrt to w
        # Hint: you can use np.linalg.lstsq
        
        # 1. f_s_target: according to demo, it inferenes how much extra force is needed to do the same traj.
        # 2. Psi (base function): establish 30 Gaussian basis function. Each basis is responsible for different phase in the action.
        #                         e.g. start, middle, end
        # 3. Feature: According to current phase s, ccalculate how much impact should each basis gives.
        # 4. weights: Use LS to find each basis's weight, letting them to combine such that it aligns with f_s_target.
        
        # Normalize bases and build locally weighted regression features
        denominator = np.maximum(
            psi.sum(axis=-1, keepdims=True),
            np.finfo(float).tiny
        )
        
         # f(s) = features @ weights
        features = s[..., None] * psi / denominator

        A = features.reshape(-1, self.nbasis)
        B = f_s_target.reshape(-1, num_dofs)

        self.weights = np.linalg.lstsq(A, B, rcond=None)[0]
        
        

    def execute(self, t, dt, tau, x0, g, x_t, xdot_t):
        """
        Query the DMP at time t, with current position x_t, and velocity xdot_t.
        The parameter tau controls temporal scaling, x0 sets the initial position
        and g sets the goal for the trajectory.

        Returns the next position x_{t + dt} and velocity x_{t + dt}
        """
        if self.weights is None:
            raise ValueError("Cannot execute DMP before parameters are set by DMP.learn()")

        # Calculate s(t) by integrating
        s = np.exp(((-self.alpha / tau) * t)) #this means s is a scalar. not 2d array in learn function (demos, time)

        # basis_centers.shape (nbasis, )
        # s.shape scalar can already broadcase to any shape
        # psi.shape = (nbasis, )
        psi = np.exp(
            -self.basis_variances
            * (s - self.basis_centers) ** 2
        )
        # TODO: Compute f(s). See equation 3.
        f_s = (
            s * (psi @ self.weights)
            / max(psi.sum(), np.finfo(float).tiny)
        )

        # Temporal Scaling
        # tau: total movement duration.
        v_t = tau * xdot_t

        # TODO: Calculate acceleration. Equation 6
        v_dot_t = (
            self.K @ (g - x_t)
            - self.D @ v_t
            - self.K @ (g - x0) * s
            + self.K @ f_s
        ) / tau

        # TODO: Calculate next position and velocity
        xdot_tp1 = xdot_t + dt * v_dot_t / tau

        x_tp1 = x_t + dt * xdot_tp1

        return x_tp1, xdot_tp1

    def rollout(self, dt, tau, x0, g):
    # x0 and g have shape (joints,), e.g. (6,).
    # dt is the time step; tau sets the movement duration.

        time = 0
        # Start from the given joint angle:
        # Shape: (joints, )
        x = x0
        x_dot = np.zeros_like(x0)
        X = [x0]

        while time <= tau:
            x, x_dot = self.execute(t=time, dt=dt, tau=tau, x0=x0, g=g, x_t=x, xdot_t=x_dot)
            time += dt
            X.append(x)

        return np.stack(X)

    @staticmethod
    def interpolate(trajectories, initial_dt):
        """
        Combine the given variable length trajectories into a fixed length array
        by interpolating shorter arrays to the maximum given sequence length.

        trajectories: A list of N arrays of shape (T_i, num_dofs) where T_i is the number
            of time steps in trajectory i
        initial_dt: A scalar corresponding to the duration of each time step.
        """
        # Use the longest demo's point as the common len
        length = max(len(traj) for traj in trajectories)
        # From the first demo, read the number of joints 
        dofs = trajectories[0].shape[1]

        # Allocate space for all resampled joint angles
        # (demos, time, joints) e.g. (87, 30, 6)
        X = np.zeros((len(trajectories), length, dofs))
        T = np.zeros((len(trajectories), length))

        # TODO Your code goes here ...
        for i, traj in enumerate(trajectories):
            # Build the original times from the original spacing. Shape(T_i, )
            # Example: 15 rows -> [0.00, 0.04, ..., 0.56]
            old_t = np.arange(len(traj)) * initial_dt

            # Create length evenly spaced times over the same duration.
            # Shape: (length, )
            # Example: 30 points from 0.00 to 0.56 sec.
            new_t = np.linspace(0.0, old_t[-1], length)

            interpolate_fn = scipy.interpolate.interp1d(
                old_t, traj, axis=0, kind="linear"
            )

            X[i] = interpolate_fn(new_t)
            T[i] = new_t
            # X: (N, length, joints); T: (N, length).
        return X, T
