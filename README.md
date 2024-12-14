# Causal-Inference-Project

# Causal Inference Project: Investigating the Impact of SO₂ Pollution on Public Health

## Overview

This project examines the causal relationship between sulfur dioxide (SO₂) pollution and general health outcomes. Utilizing causal inference methodologies, we aim to determine whether exposure to SO₂ has a significant effect on public health metrics.

## Repository Structure

- **Data**
  - `data/`: Contains datasets used for analysis, including pollution levels and health outcome metrics.
- **Code**
  - `data_engineering.py`: Scripts for data preprocessing and feature engineering.
  - `data_exploration.py`: Exploratory data analysis to understand data distributions and relationships.
  - `causal_technics.py`: Implementation of causal inference techniques applied in the study.
  - `methods.py`: Additional methods and utilities supporting the analysis.
  - `overlap_analysis.py`: Assesses the overlap in covariate distributions between treatment groups.
  - `time_period_constants.py`: Defines time periods and constants used throughout the analysis.
  - `visUSA.py`: Visualization tools for mapping and plotting data across the United States.
- **Figures**
  - `figs/`: Directory containing visualizations generated during the analysis.
  - `confunders_overlap_figs/`: Visual representations of confounder overlaps.
  - `vis_trends/`: Visualizations depicting trends over time.

## Getting Started

To replicate the analysis:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OmrySh/Causal-Inference-Project.git

2. **Install Dependencies Ensure you have the necessary Python packages installed. You can use the following command:**
   ```bash
   pip install -r requirements.txt