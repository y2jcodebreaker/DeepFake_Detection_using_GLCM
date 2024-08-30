# Project Overview

This repository contains various Python scripts related to deepfake detection, frame extraction, and machine learning model evaluation. Below is a brief description of each script.

## Scripts Description

### 1. **adding_column.py**
   - **Purpose:** This script adds an extra column to the specified CSV file along with concatenating two CSV files.

### 2. **frame_extraction.py**
   - **Purpose:** This script extracts frames from videos and stores them sequentially in the specified directory.

### 3. **glcm.py**
   - **Purpose:** This script calculates the Gray-Level Co-occurrence Matrix (GLCM) along with all its features (e.g., Contrast, Dissimilarity, Homogeneity, etc.) and stores them in CSV files based on each pixel distance.

### 4. **knn.py**
   - **Purpose:** This script implements the K-Nearest Neighbors (KNN) model for binary classification tasks.

### 5. **random_forest.py**
   - **Purpose:** This script implements the Random Forest model for binary classification tasks.

### 6. **lightgbm.py**
   - **Purpose:** This script implements the LightGBM model for binary classification tasks.

### 7. **plotting.py**
   - **Purpose:** This script plots graphs comparing real and fake faces based on each GLCM feature. It first sorts the data before beginning the plotting process.

## Usage

Each script can be run independently, and they are designed to work with specific datasets and configurations. Make sure to review the comments and instructions within each script for further details on usage.

## Requirements

Ensure that you have all the necessary Python libraries installed. You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
