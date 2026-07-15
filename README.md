# PARTE-MTF

Source code for:

**"PARTE-MTF: Persistence-Aware Topological Resonance Embedding and Topology-Temporal Fusion for Multivariate Time-Series Forecasting"**

This repository provides the implementation of PARTE-MTF, a topology-aware framework for multivariate time-series forecasting. The model integrates persistent homology-based topological representations with temporal feature learning to improve long-term forecasting performance.

---

# 1. Topological Data Analysis (TDA) Requirements

The persistent homology computation module in PARTE-MTF is implemented using **PersHomBox**, which provides a unified interface to several external Topological Data Analysis (TDA) backends.

The following TDA backends are required:

- **DIPHA**: Persistent homology computation
- **Perseus**: Persistent homology computation
- **Hera Wasserstein Distance**: Persistence diagram distance computation
**Can Download the three libaries and tda-tooltik at" https://drive.google.com/drive/folders/1ZbffEmAmvwCP9cFeBEV91PDyMn4AZZYP
---

# 1.1 Python Dependencies

Install the required Python packages:

```bash
pip install numpy scipy scikit-learn
```

The main TDA interface, **PersHomBox**, should be placed inside the project directory:

```
your_directory_name/tda-toolkit
```

The recommended directory structure is:

```
PARTE-MTF/

├── tda-toolkit/
│   └── pershombox/
│
├── models/
├── data/
├── scripts/
└── ...
```

---

# 1.2 Install PersHomBox

PersHomBox provides Python interfaces for persistent homology computation and connects the pipeline with external TDA backends.

Add PersHomBox to the Python path:

```python
import sys

sys.path.append(
    '/your_directory_name/tda-toolkit'
)

import pershombox
```

---

# 2. External TDA Backend Installation

PersHomBox requires external software backends for persistent homology computation and persistence diagram comparison.

The PARTE-MTF framework uses:

1. DIPHA
2. Perseus
3. Hera Wasserstein Distance

---

# 2.1 DIPHA

## Description

DIPHA is used for efficient persistent homology computation.

Repository:

```
https://github.com/DIPHA/dipha
```

---

## Installation

Clone and compile DIPHA:

```bash
cd /your_directory_name

git clone https://github.com/DIPHA/dipha.git

cd dipha

mkdir build

cd build

cmake ..

make
```

After successful compilation, the executable should be located at:

```
/your_directory_name/dipha/build/dipha
```

Verify the installation:

```bash
/your_directory_name/dipha/build/dipha --help
```

Grant executable permission:

```bash
chmod +x /your_directory_name/dipha/build/dipha
```

---

# 2.2 Perseus

## Description

Perseus is another persistent homology computation backend used by PersHomBox.

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

Grant executable permission:

```bash
chmod +x /your_directory_name/perseusLin
```

---

# 2.3 Hera Wasserstein Distance

## Description

Hera is used to calculate Wasserstein distances between persistence diagrams.

Repository:

```
https://github.com/GeometricaLab/hera
```

---

## Installation

Clone and compile Hera:

```bash
cd /your_directory_name

git clone https://github.com/GeometricaLab/hera.git

cd hera

mkdir build

cd build

cmake ..

make
```

After compilation, the executable should be located at:

```
/your_directory_name/hera/build/wasserstein/wasserstein_dist
```

Grant executable permission:

```bash
chmod +x /your_directory_name/hera/build/wasserstein/wasserstein_dist
```

---

# 3. Configure TDA Backend Paths

PersHomBox requires a configuration file specifying the locations of all external TDA backends.

In:

```
../tda-toolkit/pershombox/_software_backends/software_backends.cfg
```

with the following content:

```ini
[paths]

# DIPHA executable
dipha=/your_directory_name/dipha/build/dipha

# Perseus executable
perseus=/your_directory_name/perseusLin

# Hera Wasserstein distance executable
hera_wasserstein_dist=/your_directory_name/hera/build/wasserstein/wasserstein_dist
```

Ensure that all paths correspond to valid executable files.

---

# 4. Initialize TDA Environment

Before computing persistence diagrams, initialize all required TDA backends.

Example:

```python
import sys

sys.path.append(
    '/your_directory_name/tda-toolkit'
)

import pershombox

import pershombox._software_backends.resource_handler as rh


# Initialize persistent homology computation backends

rh.init_backend(
    rh.Backends.dipha
)

rh.init_backend(
    rh.Backends.perseus
)


# Initialize Wasserstein distance computation backend

rh.init_backend(
    rh.Backends.hera_wasserstein_dist
)
```

---

# 5. Verify Installation

Run the following test script:

```python
from pershombox.toplex import toplex_persistence_diagrams


dgms = toplex_persistence_diagrams(
    [(1,2)],
    [0]
)


print(dgms)
```

A successful installation should return a persistence diagram object:

```
[array(...)]
```

If an error occurs, check:

- Whether all backend executables exist
- Whether executable permissions are correctly assigned
- Whether the paths in `cfg` are correct
- Whether PersHomBox is correctly added to the Python path

---
