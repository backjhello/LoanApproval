<!-- Copilot / AI Assistant instructions tailored for the LoanApproval repo -->
# Copilot guidelines — LoanApproval

This file is a short, practical guide for AI coding agents (like Copilot / assistants) to be productive quickly in this repository.

1) Quick project summary
- Streamlit multi-page dashboard driven from `app.py` and the `pages/` folder.
- Main logic lives in `src/` (helpers for loading, EDA, plotting, preprocessing, modeling).
- Data is expected under `data/processed/` and models under `models/`.

2) How to run locally (what humans do)
- Install dependencies: `python -m pip install -r requirements.txt`
- Start the app: `streamlit run app.py` (this runs the multi-page Streamlit app)

3) Important conventions and examples to follow
- Presentation / UI code belongs in `pages/*.py` (use `src` helpers). Example: `pages/2_EDA.py` calls `src.loader.load_processed_data()` and `src.viz.plot_distribution()`.
- Reusable functions (pure transformation or plotting) live in `src/` and return data objects or matplotlib Figures. Example: `src.viz.plot_distribution()` returns a matplotlib Figure consumed by `st.pyplot` in pages.
- Cache Streamlit data loads with `@st.cache_data` (see `src/loader.py`). Prefer `src.loader` helper functions over loading files inline in page code.
- Model artifacts saved to `models/` — `src/model.py` expects `models/rf_model.pkl` by default.

4) Known issues and concrete checks the assistant should be mindful of
- Canonical processed data file: `data/processed/transactions_cleaned.csv`. Use `src.loader.load_processed_data()` from pages and helpers instead of direct `pd.read_csv(...)` to ensure consistent filenames and caching across the app.
- Empty/placeholder pages: `pages/4_Clustering.py` and `pages/5_Credit_Limit_Model.py` are empty — treat these as TODOs and add content only after confirming intended behavior.
- Minimal tests: `tests/` contains only `__init__.py`. Add unit tests in `tests/` when changing logic in `src/`.

5) Safe edits the assistant can perform right away
 - Fix file path inconsistencies by calling `src.loader.load_processed_data()` (it loads `data/processed/transactions_cleaned.csv`) and update any stray `pd.read_csv("data/processed/cleaned_data.csv")` usages.
- Implement missing pages using existing helpers (`src.viz`, `src.eda`, `src.preprocessing`, `src.model`). Keep UI code in `pages/` and logic in `src/`.
- Add basic unit tests for small pure functions in `src/` (e.g., `src/eda.anova_by_group`, `src/preprocessing.compute_age`).

6) Avoid making large, subjective changes without confirmation
- Don't add or modify large datasets in-place — ask a human if you need to change data sources or formats.
- Avoid long-running training loops inside Streamlit pages. If adding training code, create runnable scripts (e.g., `scripts/train.py`) and persist artifacts to `models/`.

7) Helpful quick tasks / PR scaffolding examples
- When implementing a new feature, include:
  - small unit tests in `tests/test_<module>.py`
  - a short update to `README.md` explaining the change
  - if you add model files, include a small `models/.gitignore` policy if large files are expected

8) Next recommended repo improvements (ask before changing):
- Add a `CONTRIBUTING.md` and basic CI that runs `pip install -r requirements.txt` and `pytest`.
- Add sample subset data in `data/sample/` so the app can run without large datasets.

If anything here looks unclear or incomplete, say what additional examples or constraints you want and I will update this file.
