# Heterogeneous Front-Door Effects: Debiased Estimation with Quasi-Oracle Guarantees

This repository contains supplementary code for the ICLR 2026 submission:

> **Heterogeneous Front-Door Effects: Debiased Estimation with Quasi-Oracle Guarantees**

It provides implementations for:
- **Synthetic experiments** (FD-PI, FD-DR-Learner, FD-R-Learner).
- **Real-world case study** using the *Fatality Analysis Reporting System (FARS)* on primary seat-belt laws.

---

## File Overview

- **`FDCATE.py`**  
  Implements FD-DR-Learner and FD-R-Learner, as described in the paper.  
  Includes:
  - Plug-in estimator (FD-PI) for comparison.
  - Cross-fitting and pseudo-outcome constructions.
  - Synthetic simulation framework.

- **`analyze_fars_2000_fd.py`**  
  Script for real-world case study (state seat-belt laws).  
  Constructs state–year panel data (treatment, mediator, outcome, covariates) from FARS and NHTSA survey tables, and applies FD estimators.

---

## Requirements

- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
