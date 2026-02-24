# Lago do Serpa Water Quality Dataset

## Contents
- `original_dataset/lago_serpa_raw.csv`: original measurements with standardized decimal separator.
- `augmented_dataset/lago_serpa_augmented_x10.csv`: augmented dataset (x10) with Gaussian noise.
- `metadata/data_dictionary.csv`: column definitions.

## Source
Measurements collected in Lago do Serpa (Itacoatiara, AM), Amazon region.
Coordinates are in WGS84 / UTM Zone 21S.

## Augmentation
- Method: repeat + Gaussian noise on input features only.
- Factor: 10 (original + 9 noisy copies).
- Noise std: 0.01 (applied to `x`, `y`, `TEMP`, `COR_A`, `COR_V`).
- Targets (`pH`, `OD`, `TURB`) are unchanged.
- Random seed: 42.

## Format
- CSV with comma delimiter and dot decimal separator.
- Header columns: Pontos, x, y, pH, OD, TEMP, COR_A, TURB, COR_V.

## Usage Note
The augmented dataset is intended for machine learning training stabilization and does not represent additional field measurements.
