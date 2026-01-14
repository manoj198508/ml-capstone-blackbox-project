# Black-Box Optimisation (BBO) Capstone Project

## Project Overview

This repository contains my work on the **Black-Box Optimisation (BBO) Capstone Project**, where the objective is to optimise **eight unknown functions** under strict query constraints. Each function can only be accessed through input–output evaluations, and no information is available about the underlying functional form, gradients, or noise structure.

The goal of the project is to identify high-performing input configurations using as few queries as possible. This setting closely mirrors real-world machine learning and optimisation problems such as **hyperparameter tuning, experimental design, and system optimisation**, where evaluations are expensive and full system knowledge is unavailable.

From a professional perspective, this project strengthens key data science skills such as **decision-making under uncertainty, iterative experimentation, and clear communication of technical strategies**, all of which are essential in applied machine learning and analytics roles.

## Inputs and Outputs

### Inputs

* Each function accepts an input vector with dimensionality ranging from **2 to 8**
* All input values are constrained to the range **[0, 1]**
* Queries must be submitted in the following format:

```
x1-x2-x3-...-xn
```

where each value:

* starts with `0`
* is specified to **six decimal places**

**Example (4D input):**

```
0.156433-0.285223-0.752660-0.225294
```

### Outputs

* Each query returns a **single scalar value**
* The output represents the function’s performance at the queried point
* No gradients, uncertainty estimates, or structural information are provided

## Challenge Objectives

The objective of the BBO capstone project is to maximise the output of each of the eight unknown functions under the following constraints:

* A limited query budget per function
* Unknown and potentially non-linear response surfaces
* No access to gradients or analytical structure
* Delayed feedback between query submission and receiving results

Given these constraints, each query must be selected carefully, balancing **information gain** with **performance improvement**.

## Strategy Evolution Across Weeks 1–3

This section summarises how my optimisation strategy evolved across the first three weekly query submissions.

### Week 1 – Initial Exploration

The first round focused on broad exploration. Queries were selected to sample different regions of the input space to understand general behaviour and scale without making strong assumptions.

### Week 2 – Identifying Promising Regions

With additional outputs available, the second round focused on identifying promising regions. Queries were refined toward higher-performing areas while maintaining some exploration to reduce uncertainty.

### Week 3 – Focused Refinement with SVM-Inspired Reasoning

By the third round, the strategy became more selective. Exploitation of high-performing regions was prioritised using small perturbations, while unstable functions continued to receive exploratory queries.

At this stage, the reasoning was influenced by **classification-style thinking inspired by Support Vector Machines (SVMs)**. Instead of predicting exact function values, the input space was viewed as regions of **high vs low performance**. Although SVMs were not implemented directly, this conceptual framework informed exploration–exploitation decisions.

## Exploration vs Exploitation

Given the limited query budget, the strategy balances:

* **Exploitation:** refining inputs near known high-performing regions
* **Exploration:** sampling new regions for uncertain or poorly performing functions

This balance is adjusted **per function** depending on observed stability and behaviour across rounds.

## Repository Organisation and Reproducibility

As the project progressed, the repository structure was reviewed and refined to improve clarity, navigability, and reproducibility.

```
.
├── data/        # Input–output pairs from each optimisation round
├── queries/     # Submitted query points per iteration
├── results/     # Returned outputs and performance summaries
├── notebooks/   # Exploratory analysis and visualisation
├── models/      # Surrogate models and optimisation logic
├── docs/        # Reflections, methodology notes, and explanations
├── requirements.txt
└── README.md
```

This structure separates concerns clearly and allows optimisation decisions to be traced across iterations.

## Tools and Libraries

The project primarily uses:

* **NumPy** and **Pandas** for numerical computation and data handling
* **scikit-learn** for lightweight surrogate modelling and exploratory analysis
* **Matplotlib / Seaborn** for visualisation

While deep learning frameworks such as **PyTorch** were explored conceptually during the module, they were not heavily integrated into the main workflow due to limited data availability and the emphasis on strategic optimisation rather than model capacity. This reflects a deliberate trade-off between **flexibility and simplicity**.

## Documentation and Reflections

This README is treated as a **living document** and has been updated iteratively to reflect:

* The evolution of optimisation strategy across rounds
* The reasoning behind exploration–exploitation decisions
* Observed limitations and assumptions

Clear documentation ensures the project communicates not only *what* was done, but *why* decisions were made—an essential requirement for professional machine learning work.

## Limitations and Future Directions

As more data becomes available, several limitations become apparent:

* Heuristic reasoning becomes less effective in higher-dimensional spaces
* Interactions between dimensions are difficult to identify manually
* Some dimensions may be less influential and require formal modelling

Future extensions may include:

* Lightweight surrogate models (classification or regression-based)
* Feature relevance analysis
* More formal Bayesian optimisation strategies

