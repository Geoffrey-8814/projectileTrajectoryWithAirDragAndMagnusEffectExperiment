import numpy as np
from scipy.integrate import solve_ivp


def _drag_and_magnus_accel(vx, vy, omega, mass, radius, area, cd, cl, rho, gravity):
    speed = np.hypot(vx, vy)
    if speed < 1e-10:
        return 0.0, -gravity

    drag = 0.5 * rho * cd * area * speed * speed
    drag_x = -drag * vx / speed
    drag_y = -drag * vy / speed

    magnus_coeff = 0.5 * rho * area * radius * cl
    magnus_x = magnus_coeff * (-omega * vy)
    magnus_y = magnus_coeff * (omega * vx)

    ax = (drag_x + magnus_x) / mass
    ay = (drag_y + magnus_y) / mass - gravity
    return ax, ay


def solve_projectile(vx0, vy0, params, t_span, t_eval=None, hit_ground=False, max_step=0.05):
    mass = params["mass"]
    radius = params["radius"]
    area = np.pi * radius ** 2
    cd = params["cd"]
    cl = params["cl"]
    gravity = params["gravity"]
    rho = params["air_density"]
    v_delta_ratio = params["v_delta_ratio"]

    v0 = float(np.hypot(vx0, vy0))
    omega0 = v_delta_ratio * v0 / (radius)

    # Spin decay from aerodynamic friction torque (no extra tuned parameter):
    #   tau = 0.5 * rho * cd * A * (omega*r)^2 * r
    #   I (solid sphere) = 2/5 * m * r^2
    #   domega/dt = -tau / I * sign(omega)
    #            = -k * |omega| * omega
    # with k computed directly from existing physical constants.
    inertia = (2.0 / 5.0) * mass * radius * radius
    spin_decay_k = 0.5 * rho * cd * area * (radius ** 3) / inertia

    def deriv(t, state):
        x, y, vx, vy, omega = state
        ax, ay = _drag_and_magnus_accel(vx, vy, omega, mass, radius, area, cd, cl, rho, gravity)
        domega = -spin_decay_k * abs(omega) * omega
        return [vx, vy, ax, ay, domega]

    events = None
    if hit_ground:
        def ground_event(t, state):
            if t <= 1e-9:
                return 1.0
            return state[1]

        ground_event.terminal = True
        ground_event.direction = -1
        events = ground_event

    return solve_ivp(
        deriv,
        t_span=t_span,
        y0=[0.0, 0.0, vx0, vy0, omega0],
        t_eval=t_eval,
        events=events,
        max_step=max_step,
        method="RK45",
        atol=1e-9,
        rtol=1e-6,
    )


def solve_projectile_from_polar(v0, theta, params, t_span, t_eval=None, hit_ground=False, max_step=0.05):
    vx0 = float(v0 * np.cos(theta))
    vy0 = float(v0 * np.sin(theta))
    return solve_projectile(vx0, vy0, params, t_span, t_eval=t_eval, hit_ground=hit_ground, max_step=max_step)


def extract_apex(solution):
    y = solution.y[1]
    x = solution.y[0]
    idx = int(np.argmax(y))
    return float(y[idx]), float(x[idx])
