public class TrajectoryCalculator {
    private static final double G = 9.81;
    private static final double DY = 1.2288; // 1.8288 - 0.6
    private static final boolean APEX_BEFORE_TARGET = true;

    // Degree-3 polynomial coefficients for aero correction
    // Features: [1, dx, hmax, dx^2, dx*hmax, hmax^2, dx^3, dx^2*hmax, dx*hmax^2, hmax^3]
    private static final double[] DVX_COEFFS = {
        0.09109675739635631, 0.045855969459507775, -0.11103307821151719,
        0.005641874697656537, -0.019091448976872195, 0.04694949613108499,
        6.453745404126632e-05, -0.0009598421939351932, 0.003329483826038357,
        -0.004969168554923187
    };
    private static final double[] DVY_COEFFS = {
        -0.003698238715882096, -0.0005564562114124564, 0.024345525798852376,
        0.00027678522513076824, 0.0006271422554351257, 0.008815387098165954,
        -6.481589714577357e-05, 0.000275934339010066, -0.0005269732049253877,
        -0.00010821122149957516
    };

    /**
     * Compute launch velocities (vx, vy) for a given horizontal distance and desired apex height.
     *
     * @param dx   horizontal distance to target (m)
     * @param hmax desired apex height (m), must be greater than DY
     * @return double[]{vx, vy} in m/s
     */
    public static double[] solve(double dx, double hmax) {
        // 1. Vacuum baseline from kinematics
        double vyVac = Math.sqrt(2.0 * G * hmax);
        double vxVac;

        if (Math.abs(DY) < 1e-10) {
            vxVac = G * dx / (2.0 * vyVac);
        } else {
            double disc = vyVac * vyVac - 2.0 * G * DY;
            if (disc < 0) {
                return new double[]{0, 0}; // impossible trajectory
            }
            double sqrtDisc = Math.sqrt(disc);
            if (APEX_BEFORE_TARGET) {
                vxVac = dx * (vyVac - sqrtDisc) / (2.0 * DY);
            } else {
                vxVac = dx * (vyVac + sqrtDisc) / (2.0 * DY);
            }
        }

        // 2. Polynomial aero correction: degree-3 features
        double dx2 = dx * dx;
        double dx3 = dx2 * dx;
        double h2 = hmax * hmax;
        double h3 = h2 * hmax;
        double[] features = {
            1, dx, hmax,
            dx2, dx * hmax, h2,
            dx3, dx2 * hmax, dx * h2, h3
        };

        double dvx = dot(DVX_COEFFS, features);
        double dvy = dot(DVY_COEFFS, features);

        // 3. Final velocity = vacuum + aero correction
        return new double[]{vxVac + dvx, vyVac + dvy};
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
