# AarogyaVeda

AarogyaVeda is a Streamlit app for chest X-ray analysis and report generation.
It helps clinicians review X-ray predictions, generate long narrative reports, export PDF files, and store reports in Google Drive.

## Current Features

- Chest X-ray classification: `NORMAL` or `PNEUMONIA`
- Grad-CAM heatmap visualization
- Image-assisted long-form AI clinical report generation
- Professional PDF report export
- Google Drive report upload and link history
- Insights dashboard and archive CSV download

## Architecture

1. Upload a chest X-ray image (`jpg/jpeg/png`).
2. Preprocess the image (`224x224`, RGB) and run the CNN classifier (`ResNet50` transfer pipeline).
3. Generate the prediction (`NORMAL` or `PNEUMONIA`) with confidence.
4. Build a Grad-CAM heatmap for explainability and region localization.
5. Generate a detailed clinical report from the image-assisted text pipeline using the uploaded X-ray plus Grad-CAM context.
6. Extract the three report sections from the same generated response: findings, impression, and precautions.
7. Render a hospital-style PDF report and make it available for download.
8. Save the report to Google Drive and store the Drive link in history.
9. Show adaptive analytics in Insights, which switches between day/week/month aggregation as history grows.

## Main Files

- `app.py`: main Streamlit application
- `cv_model.py`: training/loading and image prediction
- `medical_report_generator.py`: image-assisted report and PDF generation
- `google_drive_manager.py`: Google Drive integration with personal OAuth
- `app_utils.py`: common helpers, history utilities, and timezone handling
- `generate_refresh_token.py`: one-time script to generate personal Google OAuth refresh token

## Setup

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Hugging Face token to `.env`.

```env
HF_TOKEN=your_hf_token_here
```

## Google Drive Setup (Required for Cloud Report Saving)

To enable cloud report saving to your personal Google Drive:

1. Create OAuth Desktop credentials in Google Cloud Console.
2. Download the OAuth credentials JSON and save as `client_secret.json` in project root.
3. Run the refresh token generation script:

```bash
python generate_refresh_token.py
```

4. Copy the three values returned:
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GOOGLE_REFRESH_TOKEN`

5. **For Local Development:** Add these to `.env` file:

```env
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REFRESH_TOKEN=your_refresh_token
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id (optional, creates reports folder if omitted)
```

6. **For Streamlit Cloud Deployment:** Add the same values as secrets in Streamlit Cloud dashboard under "Settings > Secrets".

## Run the App

```bash
streamlit run app.py
```

## Dataset Structure

The app auto-detects either of these structures:

- `chest_xray/train`, `chest_xray/val`, `chest_xray/test`
- `chest_xray/chest_xray/train`, `chest_xray/chest_xray/val`, `chest_xray/chest_xray/test`

Each split should contain class folders like `NORMAL` and `PNEUMONIA`.

## Downloading the Training Dataset

To retrain the model or re-evaluate on the full dataset:

1. Download the chest X-ray dataset (ZIP file) from this link:
   - **[Chest X-Ray Dataset (Google Drive)](https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_LINK_HERE)** (Size: ~2.4 GB)
   - Or get it directly from [Kaggle: Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. Extract the ZIP file into the project root directory.
3. Ensure the folder structure matches either format above.
4. The app will auto-detect the `chest_xray/` folder on startup.

**Note:** Pre-trained model weights are included in the repo, so you can use the app immediately without downloading the dataset. Download only if you want to retrain or validate on the full dataset.

## Deployment Notes

### Streamlit Cloud

The app is configured for Streamlit Cloud deployment with personal Google OAuth refresh token authentication.

**Prerequisites:**
- GitHub repository with all source files
- Streamlit Cloud account linked to GitHub
- OAuth refresh token generated via `generate_refresh_token.py`

**Setup:**
1. Push all code to GitHub
2. Create Streamlit Cloud app from GitHub repo
3. Add the following secrets in Streamlit Cloud dashboard:
   - `GOOGLE_CLIENT_ID`
   - `GOOGLE_CLIENT_SECRET`
   - `GOOGLE_REFRESH_TOKEN`
   - `GOOGLE_DRIVE_FOLDER_ID` (optional)
   - `APP_TIMEZONE` (e.g., `Asia/Kolkata`, optional, defaults to Asia/Kolkata)

4. Deploy and access your app

**Features:**
- No OAuth popup on subsequent visits (uses refresh token)
- Reports saved to personal Google Drive
- Local timezone awareness for timestamps
- Automatic session history tracking

- Keep secrets private (`HF_TOKEN`, OAuth files, and tokens).
- If Google Drive credentials are not present, the app still runs with local features.
- History is stored in `history/prediction_history.csv`.

## Disclaimer

This tool supports clinical documentation workflows. It does not replace expert medical judgment.
