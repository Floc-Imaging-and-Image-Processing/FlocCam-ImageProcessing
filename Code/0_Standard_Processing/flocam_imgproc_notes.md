# Pure-Python Image Processing — Implementation Notes

Two files created as a parallel to `1_ImageProcess_v04.py` (no ImageJ required):

## [flocam_imgproc.py](flocam_imgproc.py) — the processing module

Four public functions:

- **`compute_edges(img)`** — Sobel magnitude clipped to uint8, matching ImageJ "Find Edges"
- **`make_binary(img1, img2, watershed=False)`** — difference → Triangle threshold → erode (cross kernel) → fill holes
- **`measure_particles(binary, edges, ...)`** — labels regions, redirects intensity measurements to the `edges` image, matching ImageJ's `redirect=edges`
- **`process_directory(dir_path, ...)`** — processes all sequential image pairs in a subdirectory

## [1_ImageProcess_v04_py.py](1_ImageProcess_v04_py.py) — the parallel main script

Structurally identical to `1_ImageProcess_v04.py` — same path config, same file organization logic, same `ts`/`dir_size`/`rmsrc`/`subdirp` parameters — but replaces the `os.system("java -jar ij.jar ...")` call with `flocam_imgproc.process_directory()`. One new parameter: `watershed = False`.

## Notes

**Difference direction:** `diff = img2 - img1` detects particles that are **brighter** than background (dark-field). If your images use bright-field (dark particles on bright background), flip to `img1 - img2` in `make_binary()` in `flocam_imgproc.py`.

**Output format:** `.txt` files are tab-separated with 17 columns, directly compatible with `2_ParticleDataProcess_v07.py`.

**Dependency:** `pip install scikit-image` if not already installed.
