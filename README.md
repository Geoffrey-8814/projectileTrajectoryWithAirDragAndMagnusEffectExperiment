This project includs tools for camera calibration and precise annotation for analyzing projectile motion. 
It also featured different numerical integration for theoretical trajectories.

## Project parameter profiles

Physics and aerodynamic parameters are now centralized in `params/projects.json`.

- Choose a profile name (for example `fuel_2026` or `legacy_ball_150mm`).
- Set `PROJECT_NAME` in `distill.py`, `actualVelocityEstimator.py`, or `traj.py`.
- Shared trajectory integration is implemented once in `ballistics.py`.

This removes duplicated constants/ODE code and keeps all scripts consistent.

![69893904435f6efc9580468ffddddce0](https://github.com/user-attachments/assets/db34710d-c06f-461f-af19-2382c5c99fa5)
