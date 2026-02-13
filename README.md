# LoanApproval
ATLAS FA25 Team #1 김세연 &amp; 서지현

## Streamlit Cloud Deployment

This project is configured for [share.streamlit.io](https://share.streamlit.io):

- App entrypoint: `app.py`
- Python runtime: `runtime.txt` (`python-3.11`)
- Dependencies: `requirements.txt`
- Dataset path: `data/processed/customer_features.csv`

If the local CSV is missing, `src/loader.py` falls back to:
`https://raw.githubusercontent.com/backjhello/LoanApproval/main/data/processed/customer_features.csv`

## Deploy Steps

1. Push this folder to `https://github.com/backjhello/LoanApproval`.
2. Go to Streamlit Community Cloud and create a new app.
3. Select repository `backjhello/LoanApproval`, branch `main`, file path `app.py`.
4. Deploy.

## File Structure

<img width="769" height="762" alt="image" src="https://github.com/user-attachments/assets/2c86ae6b-d0de-49a2-9dda-6e226415369d" />
