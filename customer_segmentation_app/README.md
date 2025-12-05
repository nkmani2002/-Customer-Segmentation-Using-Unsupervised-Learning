# Customer Segmentation AI - Streamlit App

This package contains a Streamlit application for customer segmentation using KMeans clustering and PCA.

## Files
- `app.py` - Main Streamlit application.
- `requirements.txt` - Python dependencies.
- `users.json` - User store (initially empty).

## How to run
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

Notes:
- `users.json` will be created/updated in the working directory when users register.
- For production, consider replacing JSON-based user storage with a secure database and improve password hashing/salting.
