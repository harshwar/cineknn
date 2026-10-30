# 📐 CineKNN: Complete Mathematical Architecture & Algorithmic Derivation

> **Document Type:** Mathematical Specification & Calculation Reference  
> **Target System:** CineKNN Dual-Branch Hybrid Movie Recommender  
> **Scope:** Full mathematical derivations, order of operations, execution flowcharts, and step-by-step arithmetic breakdowns of all core formulas.

---

## 📑 Table of Contents

1. [Mathematical Notation Glossary](#1-mathematical-notation-glossary)
2. [Master Algorithmic Flow & Order of Operations](#2-master-algorithmic-flow--order-of-operations)
3. [Stage 1: Bayesian Shrinkage Prior Smoothing](#3-stage-1-bayesian-shrinkage-prior-smoothing)
4. [Stage 2: Dual Vector Space Modeling & Unit Normalization](#4-stage-2-dual-vector-space-modeling--unit-normalization)
5. [Stage 3: Vectorized Item-Item Collaborative KNN Deviation](#5-stage-3-vectorized-item-item-collaborative-knn-deviation)
6. [Stage 4: Content-Based Genre Cosine Distance Branch](#6-stage-4-content-based-genre-cosine-distance-branch)
7. [Stage 5: Dynamic Hybrid Convex Combination & Clamping](#7-stage-5-dynamic-hybrid-convex-combination--clamping)
8. [Stage 6: Hyperparameter Optimization (Data-Driven Alpha Tuning)](#8-stage-6-hyperparameter-optimization-data-driven-alpha-tuning)
9. [Stage 7: Causal Anchoring ("Because You Liked" Rationale)](#9-stage-7-causal-anchoring-because-you-liked-rationale)
10. [Stage 8: Empirical Holdout Evaluation Metrics (RMSE & MAE)](#10-stage-8-empirical-holdout-evaluation-metrics-rmse--mae)
11. [Master Equation Reference Table](#11-master-equation-reference-table)

---

## 1. Mathematical Notation Glossary

| Symbol | Mathematical Definition | Domain / Dimensions | Dataset Value in CineKNN |
| :--- | :--- | :--- | :--- |
| **U** | Set of all active user profiles | card(U) = 400 | 400 users (IDs 101 to 500) |
| **I** | Set of all movie titles in catalog | card(I) = 600 | 600 Bollywood titles (1970–2024) |
| **R** | Observed user-item rating matrix | Matrix of size 400 × 600 | 23,853 observed ratings (~9.94% dense) |
| **r(u, i)** | Explicit star rating by user u on movie i | [0.5, 5.0] in discrete steps of 0.5 | 0.5★ to 5.0★ |
| **U_i** | Set of users who rated movie i | Subset of U | card(U_i) varies from 1 to 185 |
| **I_u** | Set of movies rated by user u | Subset of I | card(I_u) varies from 30 to 75 (avg ~60) |
| **r̄ (r_bar)** | Global dataset mean across all ratings | Real scalar | **3.8124 stars** |
| **C** | Bayesian shrinkage pseudo-count constant | Real constant | **10.0 pseudo-observations** |
| **μ_i (mu_i)** | Bayesian smoothed baseline rating for movie i | [0.5, 5.0] | Derived per movie (e.g., 4.41★ for 3 Idiots) |
| **v_cf(i)** | Unit-normalized collaborative vector for movie i | Vector in R^400 | Length ||v_cf(i)||_2 = 1.0 |
| **v_cb(i)** | Unit-normalized multi-hot genre vector for movie i | Vector in R^20 | Length ||v_cb(i)||_2 = 1.0 |
| **N_k(c, u)** | k nearest rated neighbors to candidate c by user u | Subset of I_u | k = 15 |
| **S_cj^(cf)** | Collaborative cosine similarity between c and j | Range [-1.0, 1.0] | Dot product: v_cf(c) · v_cf(j)^T |
| **S_cj^(cb)** | Content genre cosine similarity between c and j | Range [0.0, 1.0] | Overlap: card(G_c ∩ G_j) / √(card(G_c) × card(G_j)) |
| **α (alpha)** | Dynamic hybrid blending parameter | Range [0.0, 1.0] | Optimal tuned: α = 1.0 (Collab dominant) |
| **r̂_hybrid** | Predicted star rating for user u on candidate c | Range [0.5, 5.0] | Final clamped predicted rating |

---

## 2. Master Algorithmic Flow & Order of Operations

### 2.1 Visual Pipeline Flowchart

```
========================================================================================
                       STAGE 1: OFFLINE PREPROCESSING & VECTORIZATION
========================================================================================
  [Raw Ratings: 23,853 cells]                  [Catalog: 600 Movies, 20 Genres]
              │                                              │
              ▼                                              ▼
  Compute Global Mean: r̄ = 3.8124              Encode Multi-Hot Genre Matrix G (600 × 20)
              │                                              │
              ▼                                              │
  Compute Bayesian Baseline per Movie:                       │
  μ_i = (Σ r + 10·r̄) / (N_i + 10)                           │
              │                                              │
              ▼                                              │
  Center Ratings: R̃_ui = r_ui - μ_i                          │
              │                                              │
              ▼                                              ▼
  Unit L2 Normalize CF Vectors:                Unit L2 Normalize CB Vectors:
  v_cf(i) = R̃_i / ||R̃_i||_2                   v_cb(i) = G_i / ||G_i||_2
              │                                              │
========================================================================================
                       STAGE 2: OFFLINE HYPERPARAMETER TUNING
========================================================================================
              │                                              │
              └──────────────────────┬───────────────────────┘
                                     ▼
                   Sample 1,000 Holdout Validation Ratings
                                     ▼
                   Grid Search α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
                   Find α* minimizing Validation RMSE (α* = 1.0)
                                     │
========================================================================================
                   STAGE 3: ONLINE REAL-TIME INFERENCE (User u, Candidate c)
========================================================================================
                                     │
                   User Profile I_u  AND  Candidate Movie c
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        BRANCH A: COLLABORATIVE               BRANCH B: CONTENT GENRE
        ───────────────────────               ───────────────────────
     Cosine: S_cj = v_cf(c)·v_cf(j)        Cosine: S_cj = v_cb(c)·v_cb(j)
     Retrieve Top-15 Rated Neighbors       Retrieve Top-15 Genre Neighbors
     Weighted Shift:                       Weighted Shift:
     Δ_cf = Σ S·(r - μ) / Σ|S|             Δ_cb = Σ S·(r - μ) / Σ|S|
     CF Score:                             CB Score:
     r̂_cf = μ_c + Δ_cf                     r̂_cb = μ_c + Δ_cb
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                     STAGE 4: SYNTHESIS & BOUNDARY CLAMPING
                     ──────────────────────────────────────
                     Raw Blend: r̂_raw = α·r̂_cf + (1 - α)·r̂_cb
                     Clamping:  r̂_hybrid = min(5.0, max(0.5, r̂_raw))
                                     │
                                     ▼
                     Deliver Ranked Trays & Explainable Anchors
========================================================================================
```

---

## 3. Stage 1: Bayesian Shrinkage Prior Smoothing

### 3.1 The Fundamental Flaw of Arithmetic Means
Let $R_i$ be the raw arithmetic average:

$$
R_i = \frac{1}{|U_i|} \sum_{u \in U_i} r_{u,i}
$$

- If Movie $A$ has **1 rating of 5.0★**, its sample mean is $R_A = 5.0★$ (extreme variance, low confidence).
- If Movie $B$ has **150 ratings averaging 4.35★**, its sample mean is $R_B = 4.35★$ (high statistical confidence).
- Unsmoothed rankings place $A > B$, polluting recommendations with unvetted noise.

### 3.2 The Bayesian Shrinkage Formula
We model item quality under a prior centered at the global community average $\bar{r}$, weighted by $C = 10.0$ pseudo-observations:

$$
\mu_i = \frac{\sum_{u \in U_i} r_{u,i} + C \cdot \bar{r}}{|U_i| + C}
$$

Where:
- $\sum_{u \in U_i} r_{u,i}$ is the empirical sum of all observed ratings for movie $i$.
- $\bar{r} = 3.8124$ stars (global dataset average across all 23,853 ratings).
- $|U_i|$ is the count of users who rated movie $i$.
- $C = 10.0$ is the shrinkage prior weight.

### 3.3 Asymptotic Limits

1. **Cold-Start Limit ($|U_i| \to 0$):**

$$
\lim_{|U_i| \to 0} \mu_i = \frac{0 + 10 \cdot 3.8124}{0 + 10} = \mathbf{3.8124} = \bar{r}
$$

*Outcome:* An unrated title regresses safely to the global dataset mean.

2. **Large-Sample Limit ($|U_i| \to \infty$):**

$$
\lim_{|U_i| \to \infty} \mu_i = \lim_{|U_i| \to \infty} \left[ \frac{\sum r_{u,i}}{|U_i|} \cdot \frac{|U_i|}{|U_i| + C} + \bar{r} \cdot \frac{C}{|U_i| + C} \right] = R_i
$$

*Outcome:* As empirical ratings accumulate, the prior fades and the true average dominates.

---

### 3.4 Step-by-Step Calculation Breakdowns

#### Calculation 1: Universal Blockbuster (*3 Idiots*, Movie #1)
- Observed rating count: $|U_1| = 68$
- Observed rating sum: $\sum r_{u,1} = 305.5$
- Raw sample average: $R_1 = \frac{305.5}{68} = 4.4926★$

Calculation:

$$
\mu_1 = \frac{305.5 + (10.0 \times 3.8124)}{68 + 10} = \frac{305.5 + 38.124}{78} = \frac{343.624}{78} = \mathbf{4.4054★}
$$

*(Tempered downward by 0.087★ to reflect population uncertainty).*

#### Calculation 2: Single-Vote Rare Title
- Observed rating count: $|U_{\text{rare}}| = 1$
- Observed rating sum: $\sum r = 5.0$
- Raw sample average: $R_{\text{rare}} = 5.0★$

Calculation:

$$
\mu_{\text{rare}} = \frac{5.0 + (10.0 \times 3.8124)}{1 + 10} = \frac{5.0 + 38.124}{11} = \frac{43.124}{11} = \mathbf{3.9204★}
$$

*(Corrected downward by 1.08★! Prevents false 5-star inflation).*

---

## 4. Stage 2: Dual Vector Space Modeling & Unit Normalization

Every movie item $i$ is projected into two orthogonal metric vector spaces:

```
Movie Item i
   ├── Collaborative Vector: v_cf(i) in R^400   (400 User Interaction Dimensions)
   └── Content Genre Vector: v_cb(i) in R^20    (20 Multi-Hot Genre Dimensions)
```

### 4.1 Branch A: Collaborative Interaction Vector ($\mathbf{v}_{cf}$)
Let $\mathbf{r}_i \in \mathbb{R}^{400}$ be the raw ratings vector for movie $i$ across all 400 community users.

#### Step 1: Baseline Centering
We mean-center observed ratings by subtracting the movie's Bayesian baseline $\mu_i$:

$$
\tilde{r}_{u, i} = \begin{cases} r_{u, i} - \mu_i & \text{if } (u, i) \in \mathcal{R} \\ 0 & \text{if } (u, i) \notin \mathcal{R} \end{cases}
$$

- A user who gave $5.0★$ to a movie with baseline $4.2★$ produces a positive coordinate: $\tilde{r} = +0.8$.
- A user who gave $3.0★$ produces a negative coordinate: $\tilde{r} = -1.2$.
- Unobserved entries remain neutral $0$.

#### Step 2: Unit L2 Normalization

$$
\mathbf{v}_{cf}(i) = \frac{\tilde{\mathbf{r}}_i}{\max(\|\tilde{\mathbf{r}}_i\|_2, \, 10^{-9})}
$$

Where:

$$
\|\tilde{\mathbf{r}}_i\|_2 = \sqrt{\sum_{u=1}^{400} \tilde{r}_{u,i}^2}
$$

---

### 4.2 Branch B: Content Genre Multi-Hot Vector ($\mathbf{v}_{cb}$)
Let $\mathcal{G}_i \subseteq \{1, \dots, 20\}$ be the active genre set for movie $i$.

$$
\mathbf{g}_i = [g_{i,1}, g_{i,2}, \dots, g_{i,20}]^T \quad \text{where } g_{i,k} \in \{0, 1\}
$$

Unit L2 Normalization:

$$
\mathbf{v}_{cb}(i) = \frac{\mathbf{g}_i}{\|\mathbf{g}_i\|_2} = \frac{\mathbf{g}_i}{\sqrt{|\mathcal{G}_i|}}
$$

---

### 4.3 Why Unit Normalization Accelerates Cosine Similarity
By definition, cosine similarity is:

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}^T}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}
$$

Because all item vectors in CineKNN are pre-normalized during initialization such that $\|\mathbf{v}(i)\|_2 = 1.0$:

$$
S_{ij} \equiv \mathbf{v}(i) \cdot \mathbf{v}(j)^T = \sum_{d=1}^{D} v_d(i) \cdot v_d(j)
$$

*Efficiency Benefit:* Cosine similarity across all catalog titles reduces to a single vectorized BLAS dot product, executing in under **0.4 milliseconds**.

---

## 5. Stage 3: Vectorized Item-Item Collaborative KNN Deviation

When active user $u$ evaluates unwatched candidate movie $c$:

### 5.1 The Master Deviation Formula

$$
\hat{r}_{u, c}^{(cf)} = \mu_c + \frac{\sum_{j \in \mathcal{N}_k(c, u)} S_{cj}^{(cf)} \cdot (r_{u, j} - \mu_j)}{\max\left(\sum_{j \in \mathcal{N}_k(c, u)} |S_{cj}^{(cf)}|, \, 10^{-9}\right)}
$$

### 5.2 Step-by-Step Mathematical Decomposition

1. **Candidate Anchor Baseline ($\mu_c$):** The candidate starts at its Bayesian prior baseline.
2. **Neighborhood Selection ($\mathcal{N}_k(c, u)$):**
   - Compute similarity vector: $\mathbf{S} = \mathbf{v}_{cf}(c) \cdot \mathbf{V}_{cf}[I_u]^T$.
   - Retrieve top $k = 15$ nearest neighbors using `np.argpartition`.
3. **Neighbor Rating Deviation ($(r_{u,j} - \mu_j)$):**
   - Measures how much user $u$ liked neighbor $j$ relative to its expected quality $\mu_j$.
4. **Weighted Net Shift ($\Delta_{\text{net}}$):**

$$
\Delta_{\text{net}} = \frac{\sum_j S_{cj}^{(cf)} \cdot (r_{u,j} - \mu_j)}{\sum_j |S_{cj}^{(cf)}|}
$$

---

### 5.3 Step-by-Step Arithmetic Trace: User #101 on *3 Idiots*

**Setup Parameters:**
- Target Candidate $c$: **3 Idiots** ($\mu_c = 4.41$)
- Active User: **User #101** (Classic Drama & Biography profile)
- Top 3 Rated Neighbors ($k=3$ trace):

```
Target: 3 Idiots (μ_c = 4.41)
  ├── Neighbor 1: Taare Zameen Par  (μ_1 = 4.38, r_u,1 = 5.0, S_c1 = 0.84)
  ├── Neighbor 2: PK                (μ_2 = 4.12, r_u,2 = 4.5, S_c2 = 0.79)
  └── Neighbor 3: Dangal            (μ_3 = 4.35, r_u,3 = 5.0, S_c3 = 0.72)
```

**Step 1: Compute User Deviations:**
- $\Delta_1 = 5.0 - 4.38 = \mathbf{+0.62}$
- $\Delta_2 = 4.5 - 4.12 = \mathbf{+0.38}$
- $\Delta_3 = 5.0 - 4.35 = \mathbf{+0.65}$

**Step 2: Compute Weighted Products ($S_{cj} \cdot \Delta_j$):**
- Item 1: $0.84 \times (+0.62) = \mathbf{+0.5208}$
- Item 2: $0.79 \times (+0.38) = \mathbf{+0.3002}$
- Item 3: $0.72 \times (+0.65) = \mathbf{+0.4680}$
- Numerator Sum: $0.5208 + 0.3002 + 0.4680 = \mathbf{+1.2890}$

**Step 3: Compute Denominator (Absolute Similarity Sum):**
- Denominator Sum: $0.84 + 0.79 + 0.72 = \mathbf{2.3500}$

**Step 4: Compute Net Neighborhood Deviation:**

$$
\Delta_{\text{net}} = \frac{+1.2890}{2.3500} = \mathbf{+0.5485 \text{ stars}}
$$

**Step 5: Compute Final Collaborative Prediction:**

$$
\hat{r}_{101, \text{3 Idiots}}^{(cf)} = \mu_c + \Delta_{\text{net}} = 4.41 + 0.55 = \mathbf{4.96★}
$$

---

## 6. Stage 4: Content-Based Genre Cosine Distance Branch

### 6.1 Genre Overlap Cosine Formula
Let $\mathcal{G}_c$ and $\mathcal{G}_j$ be the genre sets for candidate movie $c$ and rated movie $j$:

$$
S_{cj}^{(cb)} = \frac{\mathbf{g}_c \cdot \mathbf{g}_j^T}{\|\mathbf{g}_c\|_2 \|\mathbf{g}_j\|_2} = \frac{|\mathcal{G}_c \cap \mathcal{G}_j|}{\sqrt{|\mathcal{G}_c| \cdot |\mathcal{G}_j|}}
$$

#### Arithmetic Example:
- Movie $c$ (*3 Idiots*): `{Comedy, Drama}` $\implies |\mathcal{G}_c| = 2$
- Movie $j$ (*PK*): `{Comedy, Drama, Fantasy}` $\implies |\mathcal{G}_j| = 3$
- Shared Genres: `{Comedy, Drama}` $\implies |\mathcal{G}_c \cap \mathcal{G}_j| = 2$

Calculation:

$$
S_{cj}^{(cb)} = \frac{2}{\sqrt{2 \times 3}} = \frac{2}{\sqrt{6}} = \frac{2}{2.4495} = \mathbf{0.8165}
$$

### 6.2 Content KNN Prediction Formula

$$
\hat{r}_{u, c}^{(cb)} = \mu_c + \frac{\sum_{j \in \mathcal{N}_k^{(cb)}(c, u)} S_{cj}^{(cb)} \cdot (r_{u, j} - \mu_j)}{\max\left(\sum_{j \in \mathcal{N}_k^{(cb)}(c, u)} |S_{cj}^{(cb)}|, \, 10^{-9}\right)}
$$

---

## 7. Stage 5: Dynamic Hybrid Convex Combination & Clamping

### 7.1 Master Synthesis Equation

$$
\hat{r}_{u, c}^{\text{raw}} = \alpha \cdot \hat{r}_{u, c}^{(cf)} + (1 - \alpha) \cdot \hat{r}_{u, c}^{(cb)}
$$

### 7.2 Strict Boundary Clamping Function

$$
\hat{r}_{u, c}^{\text{hybrid}} = \operatorname{clamp}(\hat{r}_{u, c}^{\text{raw}}, \, 0.5, \, 5.0) \equiv \min(5.0, \, \max(0.5, \, \hat{r}_{u, c}^{\text{raw}}))
$$

### 7.3 Theoretical Boundary Guarantee
For all users and candidate movies:

$$
0.5 \le \hat{r}_{u, c}^{\text{hybrid}} \le 5.0
$$

- If positive deviations yield $\hat{r}^{\text{raw}} = 5.24$, it snaps strictly to **5.0★**.
- If negative deviations yield $\hat{r}^{\text{raw}} = 0.21$, it snaps strictly to **0.5★**.

---

## 8. Stage 6: Hyperparameter Optimization (Data-Driven Alpha Tuning)

The parameter $\alpha$ is tuned on a holdout validation sample $\mathcal{D}_{\text{val}}$ ($N_{\text{val}} = 1,000$ ratings).

### 8.1 Empirical Optimization Objective

$$
\alpha^* = \arg\min_{\alpha \in \mathcal{A}} \operatorname{RMSE}(\alpha)
$$

Where:

$$
\mathcal{A} = \{0.00, \, 0.25, \, 0.50, \, 0.75, \, 1.00\}
$$

$$
\operatorname{RMSE}(\alpha) = \sqrt{\frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{(u,i) \in \mathcal{D}_{\text{val}}} \left( r_{u,i} - \operatorname{clamp}\left(\alpha \cdot \hat{r}_{u,i}^{(cf)} + (1 - \alpha) \cdot \hat{r}_{u,i}^{(cb)}, \, 0.5, \, 5.0\right) \right)^2}
$$

### 8.2 Empirical Result
- When dense interaction logs exist ($\sim 40$ reviews/movie), collaborative peer wisdom dominates, yielding **$\alpha^* = 1.0$**.
- In sparse, cold-start settings, the optimizer balances between $\alpha \in [0.25, 0.50]$.

---

## 9. Stage 7: Causal Anchoring ("Because You Liked" Rationale)

For each recommended movie $c$, CineKNN identifies the causal anchor item from the user's favorite history:

$$
\text{Anchor}(c) = \arg\max_{j \in \mathcal{U}_{\text{liked}}} S_{\text{blended}}(c, j) \quad \text{where } r_{u,j} \ge 4.0★
$$

Where:

$$
S_{\text{blended}}(c, j) = \alpha \cdot \left(\mathbf{v}_{cf}(c) \cdot \mathbf{v}_{cf}(j)^T\right) + (1 - \alpha) \cdot \left(\mathbf{v}_{cb}(c) \cdot \mathbf{v}_{cb}(j)^T\right)
$$

*Example:* Candidate *Swades* links to *Lagaan* ($S_{\text{blended}} = 0.89$, User Rating $5.0★$) $\implies$ **"Because you rated 'Lagaan' highly (5.0★)"**.

---

## 10. Stage 8: Empirical Holdout Evaluation Metrics (RMSE & MAE)

### 10.1 Evaluation Formulas
Evaluated on $N_{\text{test}} = 3,000$ strictly isolated test ratings with fixed seed (`1337`):

1. **Root Mean Squared Error:**

$$
\text{RMSE} = \sqrt{\frac{1}{N_{\text{test}}} \sum_{k=1}^{N_{\text{test}}} \left( y_k - \hat{y}_k \right)^2}
$$

2. **Mean Absolute Error:**

$$
\text{MAE} = \frac{1}{N_{\text{test}}} \sum_{k=1}^{N_{\text{test}}} |y_k - \hat{y}_k|
$$

3. **Percentage Error Reduction:**

$$
\text{Gain} = \frac{\text{RMSE}_{\text{Global Mean}} - \text{RMSE}_{\text{CineKNN}}}{\text{RMSE}_{\text{Global Mean}}} \times 100\%
$$

---

### 10.2 Empirical Offline Benchmark Results

| Model Configuration | Mathematical Formula | RMSE | MAE | Accuracy Role |
| :--- | :--- | :---: | :---: | :--- |
| **1. Global Mean Baseline** | y_hat = 3.8124 | **0.9634** | **0.7888** | Naïve non-personalized benchmark |
| **2. Bayesian Item Mean** | y_hat = mu_i | **0.8317** | **0.6596** | Popularity prior smoothing |
| **3. Content Genre KNN (alpha=0)** | y_hat = mu_i + Delta_genre | **0.5578** | **0.4415** | Cold-start metadata shield |
| **4. Collaborative KNN (alpha=1)** | y_hat = mu_i + Delta_collab | **0.2551** | **0.1921** | Dense interaction peer patterns |
| **5. CineKNN Hybrid (alpha=1.0)** | y_hat = clamp(alpha*CF + (1-alpha)*CB) | **0.2551** | **0.1921** | **WINNER: 73.5% Error Reduction** |

$$
\text{Gain} = \frac{0.9634 - 0.2551}{0.9634} \times 100\% = \frac{0.7083}{0.9634} \times 100\% = \mathbf{73.52\%}
$$

---

## 11. Master Equation Reference Table

| Concept | Mathematical Formula | Operational Description |
| :--- | :--- | :--- |
| **Global Dataset Mean** | r_bar = (1 / N_ratings) * sum(r_ui) = 3.8124 | Center of mass across all 23,853 ratings |
| **Bayesian Shrinkage Prior** | mu_i = (sum(r_ui) + 10 * r_bar) / (count_i + 10) | Prevents small-sample rating inflation |
| **Collaborative Unit Vector** | v_cf(i) = (r_i - mu_i) / ||r_i - mu_i||_2 | Centered relative user deviation in R^400 |
| **Content Multi-Hot Vector** | v_cb(i) = g_i / ||g_i||_2 = g_i / sqrt(card(G_i)) | Invariant normalized genre tags in R^20 |
| **Unit Cosine Dot Product** | S_ij = v(i) · v(j)^T | Computes similarity in under 0.4 milliseconds |
| **KNN Deviation Formula** | r_hat = mu_c + sum(S * (r - mu)) / sum(abs(S)) | Shifts candidate baseline by user neighbor taste |
| **Hybrid Convex Blend** | r_raw = alpha * r_cf + (1 - alpha) * r_cb | Blends interaction logs and genre metadata |
| **Boundary Clamping** | r_final = min(5.0, max(0.5, r_raw)) | Strictly bounds scores to 0.5★ - 5.0★ |
| **Causal Explanation Anchor** | Anchor(c) = argmax_{j: r>=4} S_blended(c, j) | Justifies "Because you liked [Movie]" |
| **Holdout Objective Loss** | RMSE = sqrt( (1/N) * sum((y - y_hat)^2) ) | Loss metric minimized during validation |
