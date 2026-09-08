# FAQ

## About the code

**Q: Can I run this on Windows?**
Yes. All experiments have been tested on Windows 11 with Anaconda. The multiprocessing scripts use `if __name__ == "__main__": mp.freeze_support()` for Windows compatibility. Wall times reported in `docs/REPRODUCE.md` come from a Ryzen 7 5825U under Windows 11.

**Q: My numbers differ from the paper by 0.001. Is this a bug?**
No. Differences at the 3rd–4th decimal in AUC and similar metrics are within numerical noise and depend on your NumPy/SciPy version. See "Notes on numerical reproducibility" in `docs/REPRODUCE.md`.

**Q: My numbers differ from the paper by 0.05 or more. Is this a bug?**
Probably yes. Open a GitHub issue with your Python version, `pip freeze` output, and OS.

**Q: How do I add a new attractor selection strategy?**
Modify the `extract_attractors()` function in `src/ca_kernel.py`. All downstream code depends only on the returned list of `(i, j)` tuples, so no other changes are needed.

**Q: Can I use my own DEM/study area?**
Yes. Replace `data/DEM_vallebajo.tif` with your DEM (must be UTM-projected) and re-run the CSV construction pipeline documented in `notebooks/02_data_preparation.ipynb`. Then follow `docs/REPRODUCE.md` from Step 0.

**Q: Why not use CUDA / GPU acceleration?**
The bottleneck is Yen's K-shortest paths algorithm on a graph of 12,826 nodes, which does not vectorise trivially. Multiprocessing across parameter configurations gives near-linear speedup on multi-core CPUs and was sufficient for all experiments in the paper.

---

## About the method

**Q: Why 250 m grid and not finer?**
See §3.6 of the paper, "On grid resolution." Briefly: 250 m is a compromise between the SRTM DEM resolution (~30 m, below which downsampling introduces noise), the size of archaeological features (tens of metres to kilometres), and the computational cost of Yen's algorithm at higher resolutions.

**Q: Why min–max normalisation and not z-score?**
Because we want a within-valley ranking, not a cross-valley comparison. See §3.1 and the limitation paragraph in §5. If your goal is to transfer the model between valleys with absolute suitability thresholds, min–max should be replaced.

**Q: Why 24 attractors specifically?**
Not chosen by hand. The number emerges from three constraints: (i) $E$ must exceed the 0.92 quantile, (ii) must be a local maximum in an 11×11 window, and (iii) must be at least 5 cells away from any already-accepted attractor (non-maximum suppression). The resulting count for the lower Cañete Valley is 24. See §3.2.

**Q: The attractors seem to sit near the river. Is this a bias?**
Yes, and it is discussed explicitly in the paper (§3.2, second defensive paragraph). Pre-Hispanic settlement in coastal Peruvian valleys was systematically adjacent to water infrastructure; the model reflects this rather than correcting for it.

**Q: How does this compare to CircuitScape / FETE / classical LCP?**
See §4.5 of the paper for a direct FETE comparison. CircuitScape (electrical circuit theory analogue) is a natural companion baseline and would be a good target for future work. The CA framework here differs from all of these in that it (i) extracts endpoints automatically from an affinity field rather than requiring user input, (ii) uses an ensemble of near-optimal paths rather than a single optimal one, and (iii) includes stochastic perceptual noise and congestion feedback.

---

## About the data

**Q: Can I get the raw MINCUL site coordinates?**
No. The CSV contains only the binary aggregated flag per 250 m cell. Individual site coordinates are subject to Peruvian law regarding cultural patrimony and require separate authorisation from the Ministerio de Cultura.

**Q: How were sites verified in the field?**
See Cornejo Meza (2022) undergraduate thesis, PUCP. Briefly: the 2022 field campaign led by Fernandini validated a subset of MINCUL points and identified new ones through direct surface prospection with GPS.

**Q: Why include the DEM if the CSV already contains the derived variables?**
For full transparency and reproducibility. If a reviewer or future researcher wants to compute slope with a different algorithm, or to derive additional variables (aspect, roughness, curvature), they need the source DEM.

---

## Practical

**Q: I don't want to run the 30-hour phase scan. Can I get the aggregated results directly?**
The current release of this repository does not ship the raw experimental outputs, only the code to produce them. If you would like the phase-scan JSONs or the summary CSVs shared directly (e.g., for meta-analysis), open a GitHub issue and we can arrange transfer.

**Q: How do I cite this?**
See `CITATION.cff`. Once the paper is accepted, the DOI will be added and this file updated.

**Q: I found a bug. What now?**
Open a GitHub issue with:
1. The command you ran
2. The full error trace
3. Your Python version and `pip freeze` output
4. Whether you modified any files

We will investigate as time permits.
