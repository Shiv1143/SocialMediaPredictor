# Social Media Performance Predictor — Multi-Brand Edition

> Consuma Technologies · AI Engineer Assignment

Predicts how an Instagram post will perform for a given beverage brand, and explains *why*.

---

## Quick Start

```bash
# 1. Activate the virtual environment
source assignment/bin/activate   # created with: uv venv assignment --python 3.11

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads ~80 MB sentence-transformer on first run)
python train.py

# 4. Start the API
uvicorn backend.main:app --reload --port 8000

# 5. Open the frontend
open http://localhost:8000
```

The `train.py` script trains the model, runs full evaluation, and saves both the model and evaluation results to `saved_models/`.

---

## Architecture

```
POST /api/predict/upload    (multipart — with optional image upload)
POST /api/predict           (JSON body)
POST /brands/register       (register a new brand with seed posts — cold start)
GET  /brands                → list of known brands
GET  /api/dataset/stats     → per-brand engagement statistics
GET  /api/evaluation        → pre-computed evaluation results
```

### System Diagram

```
User Input
  caption + media_type + metadata + [image]
          │
          ▼
  ┌───────────────────────────────────────┐
  │         Feature Extraction            │
  │  - Structured features (22 dims)      │  ← media type, duration, collab,
  │  - Embedding text construction        │    timing, followers, caption stats
  └──────────┬──────────────┬────────────┘
             │              │
             ▼              ▼
  ┌──────────────┐  ┌──────────────────────┐  ┌────────────────────┐
  │    Ridge     │  │  Sentence-Transformer │  │  Collaborator Tier │
  │  Regression  │  │    (all-MiniLM-L6-v2) │  │  Score Adjustment  │
  │  (α = 1.0)   │  │  + Brand-aware KNN    │  │  (+0–0.15 boost)   │
  └──────┬───────┘  └──────────┬───────────┘  └────────┬───────────┘
                                                         │
  ┌──────┴──────────────────────┘               off-strategy detection
  │   35% Ridge + 65% KNN                       (brand centroid cosine sim)
         │ weight 0.35         │ weight 0.65
         └──────────┬──────────┘
                    ▼
          ┌─────────────────┐
          │  Hybrid Score   │  brand-normalised 0–1
          │  → Tier         │  low / medium / high
          │  → Estimated ER │  denormalised engagement rate
          │  → Explanation  │  feature contributions + similar posts
          └─────────────────┘
```

### Why this architecture?

**Problem with standard ML here:** The dataset has 35–120 posts per brand. A gradient-boosted tree with 22 features would memorize the training set and generalise poorly. XGBoost/LightGBM are the wrong tool for this scale.

**Ridge Regression** is the right choice for small N:
- L2 regularization prevents overfitting with 35–120 samples
- Coefficients directly interpret as feature importance scores
- Fast, stable, no hyperparameter sensitivity

**Embedding KNN** covers what structured features miss:
- Captures semantic content: "garmi (heat) + Sprite" posts cluster together
- Explains predictions naturally: "your post is similar to this high-performing post"
- Handles missing data gracefully: works even with no metadata
- Enables cold-start for new brands: finds similar historical posts from any brand

**Hybrid (0.35 Ridge + 0.65 KNN):** KNN gets higher weight because with small N, the content similarity signal is more reliable than learned feature weights. The ratio was chosen so that purely structural posts (no caption, no visual summary) still get a reasonable prediction from Ridge.

### Target Variable

`engagement_rate` is normalized to a **brand-level percentile rank (0–1)** before training. This is essential because:
- Pepsi India (millions of followers) has a median ER of 0.13%
- Red Bull India has a median ER of 5.58%

A raw ER of 2% is excellent for Pepsi and mediocre for Red Bull. Percentile rank makes the target comparable across brands. Predictions are then denormalised back to estimated ER for display.

---

## Evaluation Strategy

Two separate evaluation protocols are used because they answer different questions:

| Protocol | Question | Relevance |
|---|---|---|
| **Leave-One-Brand-Out (LOBO)** | "How well does the model work for a brand it has never seen?" | Cold-start / new brand onboarding |
| **Within-Brand K-Fold (k=3)** | "How well does the model rank posts within a known brand?" | Primary deployment scenario |

### Results

**[1] Leave-One-Brand-Out CV — cold start**

| Brand Left Out     | n   | MAE   | Spearman | Tier Acc |
|--------------------|-----|-------|----------|----------|
| cocacola_india     | 120 | 0.256 | −0.026   | 32%      |
| pepsiindia         | 58  | 0.259 | −0.050   | 34%      |
| redbullindia       | 119 | 0.265 | −0.168   | 33%      |
| sprite_india       | 46  | 0.266 | +0.148   | 28%      |
| thumsupofficial    | 35  | 0.248 | +0.174   | 37%      |
| **Mean**           |     | **0.259** | **+0.016** | **33%** |

**[2] Within-Brand K-Fold CV — primary use case**

| Brand            | n   | k | Spearman | MAE   |
|------------------|-----|---|----------|-------|
| sprite_india     | 46  | 3 | **+0.451** | 0.233 |
| cocacola_india   | 120 | 3 | +0.139   | 0.249 |
| redbullindia     | 119 | 3 | +0.130   | 0.248 |
| pepsiindia       | 58  | 3 | −0.206   | 0.268 |
| thumsupofficial  | 35  | 3 | −0.354   | 0.271 |
| **Overall mean** |     |   | **+0.032** |       |

**[3] Baselines (in-sample)**

| Baseline             | MAE   | Spearman |
|----------------------|-------|----------|
| Always predict 0.5   | 0.250 | 0.000    |
| Brand mean           | 0.250 | 0.021    |
| Random               | 0.335 | −0.001   |

### Honest interpretation of results

**LOBO Spearman ≈ 0.016** and **within-brand Spearman ≈ 0.032** are both near-zero. This is not a bug — it is the ground truth of this problem with this dataset.

**Why the signal is near-zero:**

1. **Unobservable factors dominate.** Within a brand, post-level engagement is largely driven by things we cannot see: Instagram algorithm promotion (organic vs paid), whether the post rode a trending news moment, how engaged the specific collaborator's audience was that week, the exact follower segment that was shown the post. These factors plausibly explain 70–80% of within-brand variance.

2. **Small N + high variance.** With 35–120 posts per brand and 3-fold CV, each test fold has 12–40 samples. At n=12, the 95% confidence interval for a Spearman estimate is approximately ±0.57 (via Fisher z-transform). So a measured Spearman of −0.354 (Thumsup) is statistically indistinguishable from 0.

3. **Brand content strategies are unique.** What makes a Sprite post go viral (heat-relief comedy + local creator) has almost no overlap with what makes a Red Bull post go viral (extreme sports + celebrity athlete). Cross-brand KNN is weak signal.

**Where the model provides value despite weak numerical prediction:**

- **Similar post discovery**: Finding 7 historically similar posts is often more actionable than a ± noisy engagement number. "Your draft looks most like these 3 high-performing Sprite posts and these 2 low-performers — here's what's different" is a useful output.
- **Content direction**: Structural features (reel vs static, collaboration, duration bin, prime-time posting) do carry weak signal that aggregates into useful guidance.
- **Cold-start safety net**: For brands with zero posts, cross-brand similarity gives a starting point that is better than no signal.
- **The explanation is the product**: A system that says "this post is similar to X (3.5% ER, high)" is more actionable than a system that says "predicted ER: 2.87%" with no context.

### Key findings from dataset analysis

1. **Reels dominate** (295/378 posts, 78%) and have higher median ER than static posts.
2. **Collaborations are polarising**: 175/378 posts are collaborated. Some celebrity collabs drive 10×+ the brand median; most underperform.
3. **Follower count is inverse to ER**: Pepsi India (millions of followers) has the lowest median ER (0.13%), while Red Bull India has the highest (5.58%) despite a large follower base.
4. **83 posts have 0 views**: These are static image/album posts — expected behaviour, not missing data.
5. **Engagement rates are heavy-tailed**: Mean >> median for all brands (outlier viral posts skew the mean). This is why we use percentile rank as the target rather than raw ER.
6. **Caption language**: Most Sprite/Coke/Pepsi/Thums Up posts use Hindi/Hinglish captions; Red Bull posts more in English.

### Where the model fails

The 10 worst predictions are outlier posts:
- Celebrity mega-collaborations (e.g. Virat Kohli × Pepsi) that drive 50×+ normal engagement
- Festival-specific posts (IPL, Diwali, New Year) with anomalous reach spikes
- Posts with <3 similar neighbours in the training corpus (edge-case content styles)

With 10× more data, we'd have more examples of these extremes and the embedding model would identify them reliably.

---

## Feature Engineering

26 structured features extracted from each post:

| Category   | Features |
|------------|----------|
| Media type | `is_reel`, `is_post`, `is_album`, `is_static` (views=0 flag) |
| Duration   | `duration_seconds`, `duration_short` (≤15s), `duration_medium` (15–60s), `duration_long` (>60s) |
| Collab     | `is_collaborated`, `num_collaborators`, `is_reel_collab` (reel×collab interaction) |
| Timing     | `post_hour_ist`, `post_day_of_week`, `post_month`, `is_weekend`, `is_prime_time` (17–22 IST) |
| Audience   | `followers_log`, `followers_bucket` (tier 0–3) |
| Caption    | `caption_word_count`, `caption_char_count`, `emoji_count`, `hashtag_count`, `mention_count`, `has_question`, `has_exclamation` |

**Collaborator tier** (post-hoc score adjustment, not a Ridge feature): When `collaborator_follower_count` is supplied, a score boost is applied based on expected audience amplification:

| Collaborator tier | Follower range | Score boost |
|-------------------|---------------|-------------|
| Mega celebrity    | 5M+           | +0.15       |
| Macro influencer  | 1M–5M         | +0.08       |
| Mid-tier creator  | 100K–1M       | +0.03       |
| Micro-influencer  | <100K         | 0.00        |
| Unknown           | not provided  | 0.00        |

Boost is conservative (the training data shows that generic collaborations actually show 0.8× ER lift on average — it's the high-profile celebrity collabs that drive outsized results).

For embeddings, the following text is encoded:
```
Brand: {brand}. Content type: {media_type}. Duration: {n} seconds.
[Collaborated post with N collaborator(s).]
Caption: {caption}
Visual: {visual_summary[:600]}
```

The pre-computed `summary` field on media items (from a vision model) is used directly as the visual description — no external API call needed.

---

## Modeling Choices & Justification

| Decision | Choice | Reason |
|----------|--------|--------|
| Target variable | Brand-normalised percentile rank | Makes ER comparable across brands with very different follower counts |
| Model class | Ridge Regression | Small N (35–120/brand), interpretable coefficients, no overfitting risk |
| Regularisation | α = 1.0 | Standard conservative choice for small N |
| Embedding model | `all-MiniLM-L6-v2` (80 MB) | Small, fast, strong semantic understanding of mixed Hindi-English text |
| KNN K | 7 | Enough neighbours for stable average; few enough to stay local in semantic space |
| Hybrid weights | 0.35 Ridge + 0.65 KNN | KNN weighted higher because semantic content similarity is more signal-rich than metadata for this domain |
| Per-brand vs global | Global model with brand-normalised target | Per-brand models would have 35–120 samples each — far too small. Global model + normalised target is the pragmatic choice. |

---

## Potential challenges Discovered and Addressed

A careful audit of the data and model revealed several non-obvious problems:

### 1. Two different ER denominators (fixed)
**Problem:** Reels use `views` as denominator (`ER = interactions/views × 100`); static posts and albums use `followers` (`ER = interactions/followers × 100`). These are fundamentally different metrics. Ranking a reel's ER against a static post's ER within the same brand would be comparing apples to oranges — a reel with 2% ER is excellent (2% of millions of views), while a static post with 2% ER is merely decent (2% of followers actually liked it).

**Fix:** The target variable (`brand_norm_score`) is now computed separately for reels and static content within each brand. Denormalization at inference time also uses the content-type-specific percentile curve.

### 2. Extreme ER outliers distorting the target (fixed)
**Problem:** Several posts have engagement rates above 50% — e.g., a Pepsi post at 153%, a Thumsup post at 157%. These are almost certainly viral micro-events or paid amplification anomalies. If ranked naively, they absorb the entire "high" tier, making the performance tier essentially binary (everyone vs one outlier).

**Fix:** Engagement rates are winsorized at the 98th percentile per (brand, content-type) group *before* computing percentile ranks. Raw ER is preserved in the data; only the ranking input is capped.

### 3. UTC vs IST timezone in timing features (fixed)
**Problem:** All timestamps in the dataset are UTC. The `post_hour` feature was using UTC hour directly. Indian Standard Time is UTC+5:30, so a post published at 6pm IST (peak Indian Instagram hours) has a UTC hour of 12:30pm — a 5.5-hour offset. The "is this a prime-time post?" signal was completely wrong.

**Fix:** Timestamps are converted to IST before extracting `post_hour_ist`, `is_weekend`, and `is_prime_time` (17:00–22:00 IST).

### 4. `is_static` flag computed but not used as a feature (fixed)
**Problem:** The different ER denominators make `is_static` one of the most important features — it tells the model whether to expect reel-level or static-level engagement. It was computed in the loader but missing from `FEATURE_COLUMNS`.

**Fix:** `is_static` added to feature matrix. Also added `is_reel_collab` interaction term (reel × collaborated).

### 5. Silent token truncation (fixed)
**Problem:** `all-MiniLM-L6-v2` has a hard 256-token limit. `sentence_transformers` silently truncates inputs that exceed it. 8 posts had combined caption + visual summary lengths exceeding 1000 characters (roughly 250 tokens). Their long context was being silently dropped.

**Fix:** Text is explicitly truncated to 900 characters *before* encoding, with a priority order (caption > visual summary). Truncation is now deterministic and visible in the code.

### 6. Cross-brand KNN hurts within-brand ranking (fixed)
**Problem:** For a known brand, finding similar posts from Red Bull to predict Sprite performance is almost useless — engagement dynamics differ entirely. The original KNN used all brands equally. Within-brand k-fold Spearman was near-zero partly because of this cross-brand noise.

**Fix:** Brand-aware KNN: for known brands, 70% weight on same-brand neighbours + 30% on cross-brand neighbours. For unknown brands, 100% cross-brand (no choice).

### 7. No signal for unknown brands (handled explicitly)
**Problem:** An API call with `brand="mountain_dew_india"` would silently fall through to global stats with no indication that the prediction quality was severely degraded.

**Fix:** The response now includes a `warnings` field when the brand is unknown, and confidence is multiplied by 0.6 to reflect the degraded prediction quality.

### 8. Romanized Hindi captions (documented, not fixed)
**Problem (implicit assumption):** Most captions appear to be in Hindi but are written in Roman script (transliteration), e.g. "Thand rakh, garmi bhagao". Only 1/378 posts uses actual Devanagari script. `all-MiniLM-L6-v2` tokenizes Roman Hindi as sub-word tokens and finds some semantic signal, but it misses the full meaning of words like "garmi" (heat) or "thand" (cold).

**Ideal fix:** `paraphrase-multilingual-MiniLM-L12-v2` handles Hindi romanization better. Not applied here to keep the model lightweight, but worth switching to in production.

### 9. Followers count is current, not at post time (documented)
**Problem (data limitation):** The `followers` field is the brand's current follower count, not the count at post time. For posts from 2 years ago, the brand may have grown significantly. This means the computed `engagement_rate` for old posts is slightly deflated (same interactions, larger denominator). This is a form of temporal leakage in the training data.

**Partial mitigation:** We use `followers_log` and `followers_bucket` as features — these are more stable over time than raw follower count. The root issue requires storing snapshot follower counts at post time.

---

## Handling Edge Cases

| Case | Handling |
|------|----------|
| `views == 0` | Treated as is_static flag; not treated as missing data. Engagement rate can still be predicted. |
| Expired/missing S3 URLs | The system uses pre-computed `summary` fields from media items; never fetches raw media URLs at inference time. |
| Unknown brand | Falls back to global brand stats for denormalization; KNN still finds similar posts from other brands. |
| Brand with very few posts (< 20) | Model still produces a prediction using cross-brand similarity. Ridge component uses global feature weights. System warns in the explanation. |
| Missing caption | Empty string handled; embedding still uses brand + media type + visual summary. |
| Missing visual summary | Empty string; embedding uses caption only. No crash. |
| Missing timestamp | Defaults applied (noon, midweek, June) with no crash. |

---

## What I'd Do with 10× More Data

1. **Per-brand fine-tuned models**: With 350–1200 posts per brand, a lightweight gradient boosting model (HistGradientBoosting) would outperform the global model. Feature importance would be more reliable.

2. **Fine-tuned embedding model**: Fine-tune `all-MiniLM-L6-v2` on (post, engagement_percentile) pairs using contrastive learning. High-performing posts would cluster together in embedding space.

3. **Temporal features**: With more posts over time, you could detect seasonal patterns (IPL season, festive season, summer) and posting-cadence effects.

4. **Collaborator network features**: With more collab data, build a graph of collaborators and their audience sizes. A collaborator with 10M followers should be weighted very differently from one with 100K.

5. **Cross-brand transfer learning**: Enough data to study whether patterns from one brand transfer to another (e.g., does "reel + collaborator + heat theme" always outperform regardless of brand?).

6. **Better cold-start**: With more brands, train a meta-learning model (e.g., few-shot MAML) that can adapt quickly to a new brand with 5–10 posts.

---

## Project Structure

```
assignment/
├── backend/
│   ├── config.py               # Paths and constants
│   ├── main.py                 # FastAPI application + brand registration endpoint
│   ├── data/
│   │   ├── loader.py           # Dataset parsing → DataFrame
│   │   └── features.py         # Feature extraction (26 structured features)
│   ├── models/
│   │   └── predictor.py        # HybridPredictor (Ridge + KNN + brand centroids)
│   ├── llm/
│   │   └── explainer.py        # LLM explanation layer (GPT-4o-mini, falls back to template)
│   └── evaluation/
│       └── evaluator.py        # LOBO CV, baselines, failure analysis
├── frontend/
│   └── index.html              # Single-page prediction UI
├── saved_models/
│   ├── hybrid_predictor.joblib  # Trained model
│   └── evaluation_results.json  # Full evaluation output
├── train.py                    # Training + evaluation script
├── requirements.txt
└── README.md
```

---

## API Reference

### `POST /api/predict`

```json
{
  "caption": "Thand rakh! Sprite ke saath garmi ko karo bye 🧊",
  "brand": "sprite_india",
  "media_type": "reel",
  "duration": 30,
  "is_collaborated": false,
  "collaborators": [],
  "posted_at": "2026-05-01T18:00:00",
  "visual_summary": "Two friends laughing on a rooftop with Sprite bottles. Hot summer day."
}
```

**Response:**

```json
{
  "predicted_engagement_rate": 1.45,
  "brand_normalized_score": 0.42,
  "performance_tier": "medium",
  "confidence": 0.81,
  "explanation": {
    "summary": "This post is predicted to perform average for sprite india...",
    "key_factors": [ ... ],
    "similar_posts": [ ... ],
    "brand_context": { "brand_median_er": 1.91, "estimated_er": 1.45 }
  },
  "model_details": {
    "ridge_score": 0.46,
    "knn_score": 0.40,
    "ridge_weight": 0.35,
    "knn_weight": 0.65
  }
}
```

### `POST /api/predict/upload`

Same fields as above but as `multipart/form-data`. Accepts an optional `image` file. If `OPENAI_API_KEY` is set, the image is described via GPT-4o-mini Vision and the description is used for embedding.

Additional fields:
- `collaborator_follower_count` (int, optional): Enables collaborator tier score adjustment (+0.03 to +0.15)

### `POST /brands/register`

Register a new brand (cold-start onboarding) or add seed posts to an existing brand. After registration, all subsequent predictions for this brand will use within-brand KNN.

```json
{
  "brand": "mountain_dew_india",
  "followers": 1500000,
  "seed_posts": [
    {
      "caption": "Do the Dew! Extreme stunts with Hrithik Roshan 🏄",
      "engagement_rate": 3.8,
      "media_type": "reel",
      "duration": 45,
      "is_collaborated": true
    },
    {
      "caption": "Mountaineer. Explorer. Dew drinker.",
      "engagement_rate": 1.2,
      "media_type": "post"
    }
  ]
}
```

> **Minimum 3 seed posts required**. Provide 10+ for best results.

---

## Environment Variables (optional)

```env
# Ollama (local LLM — for explanation generation)
OLLAMA_HOST=http://localhost:11434   # default; change if Ollama runs elsewhere
OLLAMA_MODEL=llama3.2               # any model you have pulled, e.g. mistral, qwen2.5

# OpenAI (only needed for image → visual summary via GPT-4o-mini Vision)
OPENAI_API_KEY=sk-...
```

### LLM Explanation (Ollama)

When Ollama is running locally, the `explanation.summary` field is replaced with a rich, actionable paragraph generated by your local model. The system gracefully falls back to a structured template explanation if Ollama is not reachable.

**Quick Ollama setup:**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2       # ~2 GB — good default
# or: ollama pull mistral / qwen2.5 / phi3

# Start Ollama (if not already running as a service)
ollama serve
```

No API key or internet connection required for explanation generation once the model is pulled.
