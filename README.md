# Submission Data & Code Repository

This repository contains the underlying datasets and custom Python scripts used to generate the main and supplementary figures, as well as the spatial counterfactual random forest models for the study.

## Contents
1. `Data_China_profile_dataset.xlsx`: The raw soil carbon profile and environmental drivers dataset.
2. `code1_boxplots_compare.py`: Code to generate comparative boxplots (SOC/SIC across land uses and depths).
3. `code2_spatial_cv_rf_model.py`: Core script performing 100km spatial block cross-validation with Random Forest and generating counterfactual metrics (Δ changes).
4. `code3_vulnerability_hotspots_map.py`: Spatial mapping script to visualize vulnerability hotspots with custom colormaps and embedded panel elements.
5. `code4_partial_dependence_plots.py`: Script to generate Partial Dependence Plots (PDPs) for interpreting dominant environmental drivers.
6. `map_shp/`: Contains the base polygon shapefiles for China used in the mapping scripts.

## Requirements
- Python 3.9+
- Libraries: pandas, numpy, matplotlib, scikit-learn, geopandas

## Usage
Simply place the dataset in the same relative path or update the `DATA_PATH` variable in the scripts to run the models and output high-quality Nature-style figures.
