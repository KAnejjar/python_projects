# Development Accounting Analysis using PWT 10.1  

This directory contains a Python-based implementation of **Development Accounting** inspired by Caselli (2003) and Hall & Jones (1999). The analysis decomposes cross-country income differences into contributions from **factors of production** and **efficiency gaps** (TFP) using the **Penn World Table (PWT 10.1)** dataset.  

---

## Features  
- **Data Preparation**: Import and filter PWT 10.1 data for relevant variables.  
- **Descriptive Analysis**: Summarize GDP per worker statistics and compare against the United States.  
- **Graphical Analysis**: Generate scatterplots to examine relationships between income per worker (`y`) and production inputs.  
- **Variance Decomposition**: Measure how much of the income variation is explained by production factors (`k`, `h`, `avh`) versus efficiency (TFP).  
- **Inter-Percentile Analysis**: Compare income dispersion across percentiles.  

---

## Steps  
1. **Data Preparation**  
   - Import PWT 10.1 dataset.  
   - Filter the following variables:  
     - `rgdpo`: Real GDP (Output)  
     - `emp`: Employment  
     - `avh`: Average annual hours worked  
     - `hc`: Human capital index  
     - `labsh`: Labor share  
     - `cn`: Capital stock  

2. **Descriptive Statistics**  
   - Summarize output per worker for the latest year with complete data (**2018**).  
   - Calculate key statistics (mean, std, min, max, percentiles) and compare relative to the United States.  

3. **Graphical Analysis**  
   - Scatterplots of `ln(y)` vs.:  
     - `ln(k)` (Capital per worker)  
     - `ln(h)` (Human capital)  
     - `avh` (Hours worked)  
     - `(1 − α)` (Labor share)  
   - Include regression lines and percentile markers (25th, 50th, 75th).  

4. **Variance Decomposition**  
   - Decompose variance in GDP per worker:  
     - Share explained by factors of production (`k`, `h`, `avh`).  
     - Residual TFP (`Ã`).  
     - PWT-provided TFP (`rtfpna`).  
   - Compute inter-percentile ratios (e.g., 90th vs. 10th percentile).  

5. **Conclusions**  
   - Assess whether income differences are driven by **factor accumulation** or **efficiency gaps**.  
   - Compare results to existing literature.  

---

## Output  
- **Tables**: Descriptive statistics and variance decomposition.  
- **Visualizations**: Scatterplots showing factor relationships.  
- **Insights**: Quantitative analysis of income inequality drivers across countries.  

---

## Data Source  
- **Penn World Table (PWT 10.1)**: (link.csv)

---

## Usage  
To execute the analysis, run the **Jupyter Notebook** in this directory. 

