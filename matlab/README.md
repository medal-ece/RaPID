# Steering-direction sampling and analytical array patterns (MATLAB)

MATLAB code for the dataset generation step of the paper (Section III-C): it
places the steering directions and computes analytical array patterns by
pattern multiplication (element pattern x array factor). Base MATLAB only, no
toolboxes.

| file | purpose |
|---|---|
| `setup.m` | array and sampling parameters (frequency, spacing, array size, angular grid, number of steering directions); loads the element pattern |
| `cap_hex_centers.m` | steering directions on a spherical cap as a centroidal Voronoi tessellation: golden-angle initialisation followed by Lloyd relaxation, which gives approximately equal-area, hexagon-like cells |
| `find_phases_for_angle.m` | progressive phases `(dphase_x, dphase_y)` that steer the main beam to a requested `(theta, phi)` |
| `simulate_array.m` | analytical pattern of an M x N array for given progressive phases |
| `ArrayPattern_4x4_dataset.m` | dataset script: steering directions -> phases -> patterns -> one CSV per direction |

## Use

1. Put an element pattern file in this folder (default name `GainPlot_1x1.csv`:
   columns theta, phi, gain in dBi, for example exported from HFSS).
2. Set the array size (`M`, `N`), the number of steering directions
   (`num_batches`) and the cap angle (`cap_thetaMaxDeg`) in `setup.m`.
3. Run `ArrayPattern_4x4_dataset` in MATLAB. The array size is taken from
   `setup.m`; the CSV files are written to `datasets_4x4/`.

The steering phases produced here are the ones applied to the elements in the
full-wave simulations.
