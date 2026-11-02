# Wine Quality data

Source: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
Dataset DOI: [10.24432/C56S3T](https://doi.org/10.24432/C56S3T).
Creators: Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis (2009).
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The two included files, `winequality-red.csv` and `winequality.names`, are unchanged copies of the official UCI files. The red-wine CSV contains 1599 rows, 11 physicochemical feature columns, and a quality column. No white-wine observations are used.

## Checksums

```text
winequality-red.csv
4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e

winequality.names
15d215e73b39105952380fd487e4ce1bf5a6b83a425abb285c2ec55193d3ca92
```

The loader verifies these bytes before use. No download is required when both supplied files are present.

## Transformation used in the paper

1. Discard the `quality` column; labels are unused.
2. Compute each of the 11 feature RMS values on this public reference table and divide by those scales. Do not center.
3. Prepend the constant coordinate 1 and normalize each augmented row to unit Euclidean norm.
4. For each experimental sample, draw n rows uniformly with replacement from this public bank. Duplicate source rows retain their empirical multiplicities.
5. Multiply each sampled row by sqrt(h), where h = mu * eta, and by an independent fair global sign. The signs make the population mean zero without changing second moments.

The lower dimension is q = 12 and the decision dimension is r = 11. The finite-population second moment is h times the bank's Gram matrix divided by 1599. It is used only to compute population excess risk. The estimator fits its own sampled records and does not receive this population moment.

The reported experiment is covariance calibration using real-world features. It is not a wine-quality prediction benchmark. Preprocessing is fixed using a public reference table before drawing each simulated private sample; this does not justify applying unaccounted preprocessing to genuinely private records.
