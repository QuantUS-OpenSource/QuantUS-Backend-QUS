# Vevo F2 RF loader (`vevo_f2_rf`)

Loads FUJIFILM VisualSonics Vevo F2 B-mode RF exports. Select `<base>.rf.bmode` (2D cine) or
`<base>.rf.3d.bmode` (3D sweep); `<base>.rf.xml` must be in the same folder. Frames of either
file are a 2D frame stack (`rf_data` is `[frames, samples, lines]`); for a 3D sweep each frame is
one elevational slice. Pair with the `vevo_f2_config` analysis config loader.

`parser.py` is a Python port of `VsiParseXml.m`, `VsiBModeRfRead.m` and `readF2.m`, verified
sample-for-sample against them for 16- and 32-bit exports. For 8-bit exports the MATLAB reader
returns corrupted values (`fread(..., 'char')` decodes bytes as text); this port reads them as int8.

Loader kwargs: `speed_of_sound` (m/s, default 1540; sets fs from the depth axis as `readF2.m`
does), `ref_frame_idx` (phantom frame, default 0), `dynamic_range` (B-mode dB, default 50).
Phased-array data is scan-converted using the sector geometry from `VsiBModeRfShow.m`.
