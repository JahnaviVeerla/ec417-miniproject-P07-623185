# Data

The project uses the wine quality dataset from the UCI Machine Learning Repository.

## Source
- Red wines: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- White wines: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
- Dataset landing page: https://archive.ics.uci.edu/dataset/186/wine+quality
- Fallback ZIP: https://archive.ics.uci.edu/static/public/186/wine+quality.zip

## Licence
The dataset is distributed by UCI and is used for academic demonstration and benchmarking.

## Local data placement
Run `python scripts/get_data.py` to fetch both CSV files into `data/raw/`.
The raw files are not committed to version control.
