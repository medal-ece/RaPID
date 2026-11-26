# Data

The raw HFSS exports are not stored in this repository because of their size
(the complex sub-block fields alone are about 66 GB). This file documents the
layout the scripts expect, so that the pipeline can be run on your own exports.
`splits.json` is tracked; everything else in this folder is ignored by git.

Set `RAPID_DATA_DIR` to use a different location.

```
data/
  splits.json
  6x6_subblock_fields/      complex far fields of the nine 2x2 sub-blocks of a 6x6 array
  6x6_subblock_magnitude/   the same sub-blocks, magnitude only (dB)
  6x6_array/                whole 6x6 array patterns
  4x4_array/                whole 4x4 array patterns
  NxN_array/                whole NxN array patterns (only for evaluation and the baselines)
```

Every folder holds files `patterns_global_0001.csv`, `patterns_global_0002.csv`, ...
with 50 steering directions per file. A steering direction has the same sample
id (`s00001`, `s00002`, ...) in every folder.

## Antenna and sampling

Copper microstrip patches on FR4-epoxy, 2.4 GHz, half-wavelength (62.5 mm)
spacing on a square lattice. Far field sampled at theta = 0..180 deg (1 deg) and
phi = -179.5..179.5 deg (1 deg): 181 x 360 points per pattern. Rows of the CSV
files run over theta first, then phi (360 blocks of 181 rows).

A steering direction is defined by the progressive phase per element along x
and y (`dphase_x`, `dphase_y`, degrees). Steering directions cover
0 <= theta <= 60 deg and all azimuths, placed with a centroidal Voronoi
tessellation (see `matlab/`).

## File layouts

**Sub-block fields** (`6x6_subblock_fields/`). One 2x2 sub-block is excited at a
time with its steering phases while the other 32 elements stay in place,
unexcited. Sub-blocks are numbered b0..b8, row-major over the 3x3 grid of
sub-blocks.

| row | content |
|---|---|
| 0 | `theta_deg, phi_deg, s00001_b0_reEPhi, s00001_b0_imEPhi, s00001_b0_reETheta, s00001_b0_imETheta, s00001_b1_reEPhi, ...` |
| 1 | `dphase_x` in degrees (repeated for every column of a sample) |
| 2 | `dphase_y` in degrees |
| 3... | 65,160 data rows |

**Magnitude patterns** (`6x6_subblock_magnitude/` with columns `s00001_b0 ... s00001_b8`
per sample, and the whole-array folders with one column `s00001` per sample).
Values are in dB.

| row | content |
|---|---|
| 0 | `theta_deg, phi_deg, s00001, s00002, ...` |
| 1 | `dphase_x` in degrees |
| 2 | `dphase_y` in degrees |
| 3, 4 | peak direction (phi, theta) of each pattern |
| 5... | 65,160 data rows |

## Splits

`splits.json` lists sample ids.

| key | use | count |
|---|---|---:|
| `train`, `val` | cross-scale training task (4x4); also the training pool of the baselines at the target size | 1,400 / 175 |
| `test` | held-out steering directions on which every method is evaluated | 175 |
| `train_6x6`, `val_6x6` | 6x6 training task; contains no steering direction of `val` or `test` | 690 / 63 |
