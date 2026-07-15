# PARTE-MTF
Source code for "PATRE-MTF: Persistence-Aware Topological Resonance Embedding and Topology-Temporal Fusion for Multivariate Time-Series Forecasting"


# Topological Data Analysis (TDA) Pipeline

This part provides a Topological Data Analysis (TDA) pipeline for extracting topological representations from time-series data using persistent homology.
The pipeline integrates:
- **Time-series windowing**
- **Takens delay embedding**
- **Top-Bottom filtration**
- **Persistence diagram generation**
- **Topological feature extraction**

The persistent homology computation is implemented through **PersHomBox**, which serves as an interface to several external TDA backends:
- DIPHA
- Perseus
- Hera Wasserstein Distance
This pipeline is designed to be independent of the downstream machine learning model. Therefore, the same topological feature extraction process can be applied to different tasks, including clustering, classification, and time-series forecasting.
---
# 1. Pipeline Overview
The overall workflow is:
```
Raw Time-Series Data
          |
          v
Sliding Window Sampling
          |
          v
Top-Bottom Filtration
          |
          v
Persistent Homology
          |
          v
Persistence Diagram
          |
          v
Topological Feature Representation
          |
          v
Machine Learning / Forecasting Model
```
---
# 2. System Requirements
## Operating System
Recommended:
```
Ubuntu 20.04 / Ubuntu 22.04
```
---
## Software Requirements
| Software | Version |
|---|---|
| Python | >= 3.8 |
| GCC | >= 9 |
| CMake | >= 3.20 |
| Git | Latest |
---
## Hardware Requirement

The persistent homology computation is CPU-based.
Recommended:

- CPU: 8+ cores
- RAM: 16 GB or higher
GPU is not required for the TDA computation module.
---

# 3. Directory Structure
The recommended directory structure is:

```
your_directory_name/
│
├── tda-toolkit/
│   └── pershombox/
│
├── dipha/
│   └── build/
│       └── dipha
│
├── perseusLin
│
├── hera/
    └── build/
        └── wasserstein/
            └── wasserstein_dist
```
---

# 4. Python Dependencies
Install the required Python packages:
```bash
pip install numpy scipy scikit-learn
```
The main TDA interface:

```
PersHomBox
```
should be placed at:

```
your_directory_name/tda-toolkit
```
---

# 5. Install PersHomBox

PersHomBox provides Python interfaces for persistent homology computation.
Add PersHomBox into the Python path:

```python
import sys
sys.path.append(
    '/your_directory_name/tda-toolkit'
)
import pershombox
```
---

# 6. External TDA Backends

PersHomBox requires external persistent homology software.

The pipeline uses:
1. DIPHA
2. Perseus
3. Hera Wasserstein Distance
---

# 6.1 DIPHA

## Description
DIPHA is used for efficient persistent homology computation.
Repository:
```
https://github.com/DIPHA/dipha
```
---

## Installation
```bash
cd /your_directory_name

git clone https://github.com/DIPHA/dipha.git

cd dipha

mkdir build

cd build

cmake ..

make
```

After compilation:

```
/your_directory_name/dipha/build/dipha
```

Check installation:

```bash
/your_directory_name/dipha/build/dipha --help
```

Set executable permission:

```bash
chmod +x /your_directory_name/dipha/build/dipha
```

---

# 6.2 Perseus

## Description

Perseus is another persistent homology computation backend.

Website:

```
http://people.maths.ox.ac.uk/nanda/perseus/
```

---

## Installation

Compile Perseus:

```bash
cd /your_directory_name/perseus

make
```

The executable should be:

```
/your_directory_name/perseusLin
```

Set permission:

```bash
chmod +x /your_directory_name/perseusLin
```

---

# 6.3 Hera Wasserstein Distance

## Description

Hera is used to calculate Wasserstein distances between persistence diagrams.

Repository:

```
https://github.com/GeometricaLab/hera
```

---

## Installation

```bash
cd /your_directory_name

git clone https://github.com/GeometricaLab/hera.git

cd hera

mkdir build

cd build

cmake ..

make
```

The executable should be:

```
/your_directory_name/hera/build/wasserstein/wasserstein_dist
```

Set permission:

```bash
chmod +x \
/your_directory_name/hera/build/wasserstein/wasserstein_dist
```

---

# 7. Configure Backend Paths

Create the PersHomBox configuration file:

```
pershombox.cfg
```

The file should contain:

```ini
[paths]

# DIPHA executable
dipha=/your_directory_name/dipha/build/dipha

# Perseus executable
perseus=/your_directory_name/perseusLin

# Hera Wasserstein distance executable
hera_wasserstein_dist=/your_directory_name/hera/build/wasserstein/wasserstein_dist
```

Make sure all paths point to valid executable files.

---

# 8. Initialize TDA Environment

Before computing persistence diagrams, initialize the required backends.

Example:

```python
import sys

sys.path.append(
    '/your_directory_name/tda-toolkit'
)

import pershombox

import pershombox._software_backends.resource_handler as rh


# Initialize persistent homology backends

rh.init_backend(
    rh.Backends.dipha
)

rh.init_backend(
    rh.Backends.perseus
)


# Initialize Wasserstein distance backend

rh.init_backend(
    rh.Backends.hera_wasserstein_dist
)
```

---

# 9. Test Installation

Run the following test:

```python
from pershombox.toplex import toplex_persistence_diagrams


dgms = toplex_persistence_diagrams(
    [(1,2)],
    [0]
)


print(dgms)
```

A successful installation should return a persistence diagram:

Example:

```
[array(...)]
```

If an error occurs, check:

- Backend executable paths
- File permissions
- PersHomBox configuration file

---

# 10. Applying the Pipeline to Time-Series Data

The pipeline supports general multivariate time-series.

Assume the original dataset:

```
X ∈ R^(N × d)
```

where:

- N = number of timestamps
- d = number of variables

Example:

```
X.shape = (10000,7)
```

---

## Sliding Window Construction

For forecasting problems, the time-series is divided into overlapping windows.

Example:

```
Look-back window = 48
```

The input becomes:

```
(number_of_samples,48,7)
```

where:

- 48 = historical timestamps
- 7 = variables

Example:

```
Window 1:

t1 ---------------- t48


Window 2:

t2 ---------------- t49

```

Each window is independently transformed into a topological representation.

---

# 11. Takens Embedding

Each time-series window is converted into a point cloud using Takens embedding.

For a time-series:

\[
x(t)
\]

the embedding is:

\[
[x(t),x(t+\tau),...,x(t+(m-1)\tau)]
\]


Parameters:

| Parameter | Meaning |
|---|---|
| m | Embedding dimension |
| τ | Time delay |

Example:

```python
m = 3
tau = 5
```

---

# 12. Persistent Homology Computation

The embedded point cloud is converted into a simplicial complex using:

```
Vietoris-Rips filtration
```

The output is a persistence diagram:

```
(birth, death)
```

Example:

```
[
(0.1,0.8),
(0.3,1.5),
(0.5,2.1)
]
```

Each point represents a topological feature.

---

# 13. Topological Feature Extraction

Persistence diagrams can be transformed into numerical vectors.

Common representations:

- Persistence Landscape
- Persistence Image
- Persistence Entropy
- Wasserstein Distance
- Bottleneck Distance

These features can be used by:

- Support Vector Machines
- Clustering algorithms
- Neural networks
- Time-series forecasting models

---

# 14. Example Pipeline

```python
# Load time-series

data = load_data()


# Generate sliding windows

windows = create_windows(
    data,
    window_size=48
)


# Compute persistence diagrams

diagrams = []

for window in windows:

    pd = toplex_persistence_diagrams(
        window,
        [0]
    )

    diagrams.append(pd)


# Extract topological features

features = extract_features(
    diagrams
)


# Train downstream model

model.fit(
    features,
    labels
)
```

---

# 15. Reusability for Different Time-Series Problems

The persistent homology computation does not depend on:

- Dataset size
- Number of variables
- Forecasting horizon
- Downstream model

For example:

Original data:

```
(N,7)
```

with:

```
look_back = 48
```

becomes:

```
(samples,48,7)
```

Only the input preparation changes.

The internal persistent homology computation remains unchanged.

---

# 16. Troubleshooting

## Problem 1: Cannot import PersHomBox

Error:

```
ModuleNotFoundError: No module named 'pershombox'
```

Solution:

```python
sys.path.append(
'/root/project12/tda-toolkit'
)
```

---

## Problem 2: Backend initialization failure

Example:

```
Cannot find dipha executable
```

Check:

```bash
ls /root/project12/dipha/build/dipha
```

Update:

```
pershombox.cfg
```

---

## Problem 3: Permission denied

Solution:

```bash
chmod +x executable_path
```

---

# 17. Citation

If you use this pipeline, please cite the relevant works:

- Persistent Homology
- PersHomBox
- DIPHA
- Perseus
- Hera Wasserstein Distance

---

# 18. License

This project is developed for academic research purposes.
