# ME.250.790 ML w/ Python for Biomedical Informatics: Starter Files

This directory contains the files to get you started.

## 🚀 Quick Start Guide

Follow these steps to get your environment set up. This works on both **Mac** and **Windows**.

### Step 1: Install Prerequisites

You need:
- **Anaconda or Miniconda** - [Download Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for data science)
- **VS Code** - [Download here](https://code.visualstudio.com/)
- **Git** (optional, for cloning the repository)

> **Why Conda?** Conda is the standard package manager for data science because it handles complex scientific library dependencies (NumPy, SciPy, scikit-learn, etc.) much better than pip.

### Step 2: Set Up Python Environment

Open VS Code's integrated terminal (Command Palette → Create New Terminal) and run the following commands in order:

**Using Conda (Recommended for Data Science):**
```bash
cd me_250_790
conda env create -f environment.yml
conda init powershell # This line on windows only
conda activate me_250_790
pip install -e .
```

> **Tip:** You should see `(me_250_790)` at the start of your terminal prompt when the environment is activated.

> **Note:** The `pip install -e .` command installs the local utility library so you can use `from util.common import *` in your notebooks.


### Step 3: Configure Your Submission Token

To submit assignments, you need to set up your personal token:

1. If you don't have one already, **create an Informatics Classroom Token** here: https://bids-class.azurewebsites.net/classes/ClinicalMachineLearning/

1. **Rename `env.example` to `.env`** and make sure it is in the root folder of this project (same folder as this README)

2. **Add your token** to the `.env` file:
   ```
   SUBMISSION_TOKEN=your_token_here
   ```
   
   Replace `your_token_here` with the actual token provided by your instructor.

3. **Save the file** - That's it! The notebooks will automatically load your token.

> **Important:** Never share your token with others and don't commit it to version control. Keeping the token in a separate `.env` file, with an entry in `.gitignore`, is good practice to ensure this won't happen on accident.


### Step 4: Verify Everything Works

1. Open `setup.ipynb`
2. Select the Python kernel: click the kernel picker in the top-right corner of the notebook
3. Choose the kernel that shows `me_250_790` (your conda environment)
4. Run the cells - if it runs without errors, you're all set! 🎉

---

## 📚 Using the Utility Library

This course includes a utility library (`util/common.py`) with helpful functions you can use in any notebook.

### Importing Utilities

At the top of any notebook, add:
```python
from util.common import *
```

This gives you access to:
- **Enhanced pandas functions**: `df.value_counts_pct()` - show counts and percentages together
- **Visualization helpers**: `confusion_matrix_chart(y_test, y_pred)` - quickly plot confusion matrices
- **ROC/PR curves**: `roc_curve_chart()`, `pr_curve_chart()` - performance visualization
- **Pre-configured plotting**: Altair backend automatically enabled for pandas
- **The informatics classroom answer submission function**

### Example Usage

```python
from util.common import *
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'A', 'B']})

# Use the enhanced value_counts with percentages
df['category'].value_counts_pct()
```

Output:
```
         count  percent
A          3     50.00%
B          2     33.33%
C          1     16.67%
```

---

## 🔧 Troubleshooting

### "Module not found" errors
- Make sure your conda environment is activated (you should see `(me_250_790)` in your terminal)
- Try: `conda activate me_250_790` then `pip install -e .`
- In VS Code, select the correct kernel (the one that says `me_250_790`)
- If using venv, activate it and run `pip install -e .`

### Token not loading
- Check that your `.env` file is in the root folder (same location as `README.md`)
- Verify the file is named exactly `.env` (not `.env.txt`)
- Make sure there are no extra spaces in the line: `SUBMISSION_TOKEN=your_token`
- Restart the notebook kernel after creating/editing the `.env` file

### Conda environment not activating
- Try: `conda activate me_250_790`
- If conda isn't recognized, restart your terminal or run: `conda init` then restart terminal
- List available environments: `conda env list`

### Packages not installing
- **With conda**: `conda install package-name` or `conda install -c conda-forge package-name`
- **With pip** (in conda env): `pip install package-name`
- Update conda: `conda update conda`

---

## 📂 Repository Structure

```
me_250_790/
├── README.md                          # This file
├── .env                               # Your personal token (you create this)
├── environment.yml                    # Conda environment configuration
├── pyproject.toml                     # Local utility library configuration
├── util/                              # Utility library
│   ├── __init__.py
│   └── common.py                      # Helpful functions for notebooks
├── data/                              # Place datasets here
├── Week 1/                            # Place week 1 materials here
└── .../
```

---
