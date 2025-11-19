# LoanApproval
ATLAS FA25 Team #1 김세연 &amp; 서지현


FILE STRUCTURE:

LoanApproval/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned and transformed data
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration
│   ├── 02_analysis.ipynb       # Main analysis
│   └── 03_modeling.ipynb       # Model development (if applicable)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data cleaning and transformation functions
│   ├── visualization.py        # Custom plotting functions
│   ├── modeling.py             # Model training/loading functions
│   └── utils.py                # Helper utilities
│
├── pages/                      # Streamlit multi-page support
│   ├── 1_📊_Overview.py
│   ├── 2_📈_Analysis.py
│   └── 3_🔮_Predictions.py
│
├── assets/
│   ├── images/                 # Logos, screenshots, diagrams
│   ├── styles.css              # Custom CSS styling
│   └── data_samples/           # Sample data for demos
│
├── models/                     # Saved ML models (if applicable)
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py
│
└── .gitignore
