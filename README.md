# 🎬 CineKNN Barebone — Free-to-Host Hybrid Movie Recommender

> **A minimal, lightweight, zero-bloat recreation of the Dual-Branch Hybrid Movie Recommender System.**
> Built from scratch with pure Python, Flask, NumPy, and Pandas. Designed to be **barebone and straightforward in UI**, yet preserving **100% of the mathematical and functional power** of the original project. Can be **hosted 100% free** on Render, Railway, or PythonAnywhere.

---

## 🌟 What This Re-Creation Does

- **Dual-Branch Hybrid KNN Engine:** Mathematically predicts star ratings ($0.5 - 5.0$) for unwatched titles by dynamically blending Item-Item Collaborative Filtering with Content-Based Genre Cosine Distance:
  $$\hat{r}_{\text{Hybrid}} = \alpha \cdot \hat{r}_{\text{CF}} + (1 - \alpha) \cdot \hat{r}_{\text{CB}}$$
- **Bayesian Shrinkage Mean Prior:** Mitigates sparse rating bias by shrinking item averages toward the dataset mean:
  $$\mu_i = \frac{\sum_{u \in U_i} r_{u,i} + 10 \cdot \bar{r}}{|U_i| + 10}$$
- **Full 600-Title Bollywood Dataset:** Authentic 600 Bollywood titles (1970–2024) and 23,853 interaction logs across 400 user profiles.
- **4 Dynamic Shelves:**
  1. 🌟 *Top Recommended Movies For You* (Highest predicted star ratings)
  2. 🎯 *Because You Rated Highly* (Cosine similarity to user's favorite anchor title)
  3. 🎭 *Top in Favorite Genres* (Personalized genre filtering)
  4. 🏆 *Top Rated Overall* (Bayesian smoothed item baseline)
- **🔬 Algorithm Lab & Real-Time Calculation Breakdown (`/algorithm`):** Inspect the exact mathematical trace for any movie and profile, including Bayesian numbers, deviation calculations, and top influential neighbors.
- **Barebone, Fast UI:** Clean cards, simple tables, zero heavy JavaScript frameworks, fast page loads, and mobile responsiveness.
- **User Accounts & Instant Demo Profiles:** SQLite persistence for personal ratings and watchlists, plus 1-click demo profiles (User #101, #205, #350).
- **Free Cloud Deployment:** Ready for 1-click deployment on Render.com or any standard Python free tier.

---

## 📊 Offline Benchmark Evaluation

Evaluated across 3,000 holdout test ratings (`python evaluate.py`):

| Model Configuration | RMSE | MAE | Status |
| :--- | :---: | :---: | :--- |
| **1. Global Mean Baseline** | 0.9634 | 0.7888 | Benchmark baseline |
| **2. Bayesian Item Mean Baseline** | 0.8317 | 0.6596 | Popularity prior |
| **3. Content-Based Genre KNN ($\alpha=0$)** | 0.5578 | 0.4415 | Cold-start stability |
| **4. Collaborative Item-Item KNN ($\alpha=1$)** | 0.2551 | 0.1921 | Peer interaction wisdom |
| **5. Hybrid Model ($\alpha^* = 1.0$)** | **0.2551** | **0.1921** | **73.5% Error Reduction** |

---

## 📁 Repository Structure

```
d:\hybrid2\
├── app.py                     # Flask web server, user context, and REST API
├── recommender.py             # Dual-Branch Hybrid KNN Engine (~180 lines)
├── database.py                # SQLite persistence (users, ratings, watchlist CRUD)
├── evaluate.py                # Offline model evaluation benchmark (RMSE / MAE)
├── test_suite.py              # Automated unit tests covering all components
├── requirements.txt           # Python dependencies
├── Procfile                   # Gunicorn process definition for cloud deployment
├── render.yaml                # 1-click Render.com free deployment config
├── .gitignore                 # Git ignore rules
├── data/
│   ├── movies.csv             # 600 curated Bollywood movies with metadata
│   └── ratings.csv            # 23,853 interaction logs across 400 user profiles
├── static/
│   └── style.css              # Barebone minimalist CSS (~250 lines)
└── templates/
    ├── base.html              # Clean navbar and layout wrapper
    ├── index.html             # Dashboard with shelves, watchlist, and controls
    ├── algorithm.html         # Live mathematical derivation breakdown lab
    ├── search.html            # Search catalog results with live predictions
    └── login.html             # Demo profile launcher and user auth
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd hybrid2

pip install -r requirements.txt
```

### 2. Run Offline Benchmark
```bash
python evaluate.py
```

### 3. Run Automated Test Suite
```bash
python test_suite.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser!

---

## 🌐 Free Cloud Hosting Guide

### Option 1: Render.com (Recommended — 100% Free Web Service)
1. Push your code to your GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of barebone hybrid recommender"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. Sign in to [Render.com](https://render.com) (free).
3. Click **New +** -> **Web Service**.
4. Select your GitHub repository.
5. Render detects `render.yaml` automatically, or configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
   - **Plan:** Free
6. Click **Deploy Web Service** — your app is live on a free `.onrender.com` URL!

### Option 2: PythonAnywhere (100% Free Forever)
1. Create a free account at [PythonAnywhere](https://www.pythonanywhere.com/).
2. Open a Bash console and clone your repo:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   pip install --user -r requirements.txt
   ```
3. Go to the **Web** tab -> **Add a new web app** -> **Manual configuration (Python 3.10+)**.
4. In the WSGI configuration file, point to `from app import app as application`.
5. Reload the app — it runs 24/7 on `<username>.pythonanywhere.com` for free!

### Option 3: Hugging Face Spaces (Free Docker/Streamlit/Flask)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) -> **Create new Space**.
2. Select **Docker** or **Static/Gradio**.
3. Push your repository — it boots instantly on free CPU hardware.
