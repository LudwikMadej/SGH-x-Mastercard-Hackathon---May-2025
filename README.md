## Directory Structure

```text
├── Data/                 # Root data directory (contains all sub‑folders below)
│   ├── Downloaded_data/   # Raw data pulled from external sources
│   ├── Preprocessed_data/ # Staged/interim data sets (after light cleaning)
│   ├── Cleaned_data/      # Fully cleaned data ready for feature engineering
│   ├── MergedDataGen.py   # One‑off script that builds aggregated data sets
│   └── toolkit.py         # Shared helpers 
│     
├── toolkit.py             # utilities
│
├── EDA/                   # Exploratory notebooks & helpers
│   ├── EDA.ipynb          # Main exploratory analysis notebook
│   └── toolkit.py         # Plotting shortcuts used in the notebook
│
├── Preprocessing/         # Feature‑engineering 
│   ├── dictionaries/      # lookup tables
│   ├── Merging.py         # Merging transformers
│   ├── MetricsHelper.py   # Calcualting cosine and eucalidian metric
│   ├── Preprocessor.py    # Running merger class
│   ├── Transformers.py    # Custom transformers
│   └── toolkit.py         # Shared helpers 
│
├── Models/                # Training/evaluation artefacts
│   ├── savedModels/       # Saved hyperparameters and selected features
│   ├── FinalModel.ipynb   # Notebook that assembles final model
│   ├── GridSearch.py      # Hyper‑parameter search 
│   ├── Modelling.ipynb    # Creating first models
│   ├── Visuals.py         # Static plot generation (ROC, f1, etc.)
│   └── toolkit.py         # Shared helpers 
