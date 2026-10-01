# Candidate-law discovery

`materia/analysis/discovery.py` provides two Materia-native tools.

* `pi_groups(variables)`: the dimensionless groups of the Buckingham Pi
  theorem as an exact integer null space of the dimension matrix over the SI
  base dimensions. Pendulum: T^2 g / L. Drag: F / (rho v^2 D^2) and
  F rho / mu^2.
* `discover(x, y, terms, term_dimensions, target_dimensions)`: a sparse
  linear combination of library terms. By default the model is chosen on
  held-out data as the sparsest whose held-out error is within 10 percent of
  the best: by exhaustive best-subset search over up to four terms when the
  library has at most 16 terms, and along the path of sequentially
  thresholded least squares (Brunton, Proctor and Kutz 2016) otherwise.
  Sequential thresholding alone failed here on noisy data with strongly
  correlated power laws, keeping r^-8 and r^-14 beside the true terms. Terms whose dimensions, prefactor
  included, differ from the target's are refused before fitting. A held-out
  fraction gives the out-of-sample error and bootstrap resamples give the
  stability of each term, with the whole selection repeated on every resample.

The output is labelled a candidate law with origin `estimated`: it describes
the data within the stated error over the stated range. It is not a theorem.

## Checked

| Case | Result |
| --- | --- |
| Lennard-Jones force from Materia's own evaluations, library r^-1 to r^-14 | selects exactly r^-13 and r^-7; coefficients equal 48 eps sigma^12 and -24 eps sigma^6 to 1e-8; held-out error 2e-10; both terms in every bootstrap resample |
| Same with noise of 1e-3 of the largest force | still selects r^-13 and r^-7, held-out error below 1 percent |

## Limitations

* Linear in the library coefficients; no symbolic search over nonlinear
  forms (no genetic programming), so a law outside the library is not found.
* Dimensions are checked for library terms supplied with their dimensional
  prefactor; nondimensionalising with `pi_groups` first is the cleaner route.
