# 🚀 24/7 Cloud Deployment Guide (Render & Vercel)

This guide shows you how to get a **permanent public web link** for your FloraVision AI Flower Species Classification project that works **24/7 on any phone or laptop**, even when your computer is shut down.

---

## 🌟 Method 1: Deploy to Render.com (Recommended for PyTorch & YOLO)

Render provides a 100% free cloud server that natively supports Python, PyTorch, OpenCV, and YOLOv8 without serverless bundle size limits.

### Step 1: Create a GitHub Repository
1. Log in to [GitHub](https://github.com).
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. Name it: `flower-species-classification`
4. Set it to **Public** (or Private).
5. **Do NOT** check "Add a README" (we already have one).
6. Click **Create repository**.

### Step 2: Push Your Code to GitHub
Open VS Code terminal (`Ctrl + ~`) in `c:\Users\SHARAN\Downloads\flower` and run:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/flower-species-classification.git
git push -u origin main
```
*(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username).*

---

### Step 3: Launch on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and Sign In using your **GitHub account**.
2. Click the blue **New +** button in the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository**, then click **Next**.
4. Connect and select your `flower-species-classification` repository.
5. Fill in these settings (or Render will auto-fill them from `render.yaml`):
   - **Name**: `floravision-ai` *(or any name you like)*
   - **Language / Runtime**: `Python 3`
   - **Region**: Choose closest to you (e.g., Singapore or Frankfurt)
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Instance Type**: Select **Free**
6. Click **Create Web Service**.

Within 2 to 3 minutes, your deployment will finish and you will see your live public URL at the top:
👉 **`https://floravision-ai.onrender.com`**

You can send this link to your friend or open it on any phone or laptop in the world without your laptop being on!

---

## ⚡ Method 2: Deploy to Vercel

If you prefer Vercel:
1. Push your code to GitHub (as shown in Step 1 & 2 above).
2. Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
3. Click **Add New...** -> **Project**.
4. Import your `flower-species-classification` repository.
5. Vercel will detect `vercel.json` automatically.
6. Click **Deploy**.

> **Note**: Vercel Serverless Functions have a 250MB uncompressed limit. If PyTorch exceeds serverless RAM on Vercel, use **Render.com** which provides full VM instances with unconstrained ML dependencies.
