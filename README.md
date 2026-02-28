# Mental State Engine

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A behavioral fusion system that integrates explicit self-report, implicit text features, and interaction patterns to assess mental state consistency and risk.

---

## What This Project Is

Mental State Engine is a **prototype backend** for a mental health assessment system. It combines structured questionnaire responses (explicit signals) with conversational text analysis (implicit signals) and behavioral metadata (interaction patterns) to compute:

1. **Consistency scores** — how well implicit signals align with explicit self-report
2. **Risk assessments** — relative severity indicators based on multi-modal fusion

This is **not** a clinical diagnostic tool. It is a research prototype exploring whether consistency between self-report and conversational language can provide signal about mental state validity and severity.

**Current stage:** Phase 1 prototype. Infrastructure complete, awaiting real session data for calibration and training.

---

## Why Consistency Matters

Traditional mental health systems use either structured questionnaires or language modeling.

This engine introduces an explicit **consistency variable**:

```
C = ||x_explicit - x̂_explicit||
```

This variable models disagreement between self-report and conversational signals, treating **inconsistency itself as an informative feature**.

When someone's words don't match their self-report, that mismatch may indicate:
- Minimization or denial
- Alexithymia (difficulty identifying emotions)
- Lack of insight into mental state
- Social desirability bias

Most systems concatenate modalities. This system **measures their alignment** as a separate diagnostic signal.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Session                            │
│  ┌──────────────────┐        ┌────────────────────────┐     │
│  │  Explicit Phase  │        │   Implicit Phase       │     │
│  │  (Structured FSM)│        │   (Conversational)     │     │
│  │                  │        │                        │     │
│  │  • Mood: 1-5     │        │  "I feel completely    │     │
│  │  • Sleep: 1-5    │        │   broken and useless"  │     │
│  │  • Stress: 1-5   │        │                        │     │
│  └──────────────────┘        └────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────────┐
        │        Feature Extraction                 │
        │                                           │
        │  Text Features (768-dim embedding)        │
        │  + Sentiment, Self-reference, Distortion  │
        │                                           │
        │  Behavior Features (latency, variance)    │
        └──────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────────┐
        │         Projection & Fusion               │
        │                                           │
        │  Implicit → Explicit projection           │
        │  Consistency = ||x_exp - x̂_exp||         │
        │  Risk = α₁·x̄_exp + α₂·C + α₃·x̄_behav     │
        └──────────────────────────────────────────┘
```

---

## Mathematical Framework

### Formal Session Representation

A session is modeled as a finite state machine:

```
M = (S, s₀, Σ, δ, F)
```

Where:
- `S` — set of FSM states (mood, sleep, stress questions)
- `s₀` — initial state
- `Σ` — input alphabet (user responses)
- `δ` — transition function
- `F` — final accepting states

Each interaction generates a log entry:

```
ℓ_t = (s_t, r_t, λ_t, τ_t)
```

Where:
- `s_t` — FSM state at time `t`
- `r_t` — user response
- `λ_t` — response latency
- `τ_t` — absolute timestamp

### Feature Spaces

The system operates on three feature spaces:

**Explicit Space** (from structured FSM):
```
x^explicit ∈ ℝ^m       (e.g., [mood, sleep, stress])
```

**Implicit Space** (from conversational text):
```
x^implicit = [e_text, Sent, SR, CD] ∈ ℝ^d

where:
  e_text    — DistilBERT CLS embedding (768-dim)
  Sent      — sentiment polarity ∈ [-1, 1]
  SR        — self-reference ratio
  CD        — cognitive distortion ratio
```

**Behavior Space** (from interaction logs):
```
x^behavior = [λ̄, σ_λ, D_session, Q] ∈ ℝ^4

where:
  λ̄         — mean response latency
  σ_λ       — latency standard deviation
  D_session — total session duration
  Q         — number of answered questions
```

---

## Core Algorithms

### 1. Text Feature Extraction

**DistilBERT Embedding:**
```
H = DistilBERT(text)       # shape: (seq_len, 768)
e_text = H[CLS]            # extract CLS token embedding
```

**Sentiment Analysis:**
```
Sent = VADER(text)         # polarity ∈ [-1, 1]
```

**Self-Reference Ratio:**
```
SR = count(first_person_pronouns) / N_words
```

**Cognitive Distortion Detection:**
```
distortion_keywords = ["always", "never", "nothing", "completely", ...]
CD = count(distortions) / N_words
```

**Combined implicit feature:**
```
x^implicit = [e_text, Sent, SR, CD]     # shape: (772,)
```

---

### 2. Behavioral Feature Aggregation

From FSM interaction logs:

**Response Latency:**
```
λ_t = τ_t^response - τ_t^display
```

**Session Statistics:**
```
λ̄ = (1/T) Σ λ_t                        # mean latency

σ_λ = sqrt((1/T) Σ (λ_t - λ̄)²)        # latency variance

D_session = τ_end - τ_start             # total duration

Q = count(answered_questions)           # completion count
```

**Combined behavior feature:**
```
x^behavior = [λ̄, σ_λ, D_session, Q]    # shape: (4,)
```

---

### 3. Implicit-to-Explicit Projection

**Current implementation (Phase 1):** Heuristic projection using weighted linear combination.

**Mathematical form:**
```
x̂^explicit = W · x^implicit + b
```

Where `W` and `b` are **currently hand-tuned** based on domain knowledge.

Example heuristic mapping for mood dimension:
```
x̂_mood = w₁·Sent + w₂·SR + w₃·CD + b
```

**Phase 2 plan:** Replace with learned projection trained on real session pairs:
```
Loss = ||x^explicit - x̂^explicit||₂²
```

Minimize this loss over 200+ sessions to learn optimal `W` and `b`.

---

### 4. Consistency Score

Measures alignment between what the user explicitly reports and what their text implicitly reveals:

```
C = ||x^explicit - x̂^explicit||₂
```

**Interpretation:**
- `C ≈ 0` → strong alignment (user's words match their self-report)
- `C >> 0` → mismatch (possible minimization, alexithymia, or inaccurate self-assessment)

This is the **core novel contribution** of this system — most mental health tools use explicit OR implicit data, not the consistency between them.

---

### 5. Risk Fusion Model

**Current implementation (Phase 1):** Linear weighted fusion.

```
risk = α₁·x̄^explicit + α₂·C + α₃·x̄^behavior
```

Where:
- `x̄^explicit` — mean of explicit dimensions (averaged severity)
- `C` — consistency score (alignment penalty)
- `x̄^behavior` — mean of normalized behavior features
- `α₁, α₂, α₃` — fusion weights (currently `α₁=0.5`, `α₂=0.3`, `α₃=0.2`)

**Phase 2 plan:** Replace with learned risk model (logistic regression or neural network):

```
For classification:    p(risk=1) = σ(w^T · x_fused + b)
For regression:        ŷ = w^T · x_fused + b          (e.g., PHQ-9 score)
```

Train on labeled sessions with ground-truth clinical assessments.

---

## Project Structure

```
mental_state_engine/
│
├── data/
│   ├── raw_sessions/           # Raw session logs (JSON)
│   ├── processed/              # Preprocessed features (CSV)
│   └── dictionaries/           # Lexical resources (distortion keywords)
│
├── src/
│   ├── config.py               # Hyperparameters and paths
│   ├── main.py                 # End-to-end pipeline runner
│
│   ├── ingestion/
│   │   ├── session_loader.py   # Load and parse session logs
│   │   └── schema.py           # Define session data schema
│
│   ├── text_features/
│   │   ├── embedding.py        # DistilBERT CLS extraction
│   │   ├── sentiment.py        # VADER polarity
│   │   ├── self_reference.py   # First-person pronoun counting
│   │   ├── distortion.py       # Cognitive distortion detection
│   │   └── feature_builder.py  # Concatenate all text features
│
│   ├── behavior/
│   │   ├── latency.py          # Compute λ̄, σ_λ from logs
│   │   └── behavior_features.py # Build x^behavior vector
│
│   ├── projection/
│   │   ├── heuristic_projection.py   # Hand-tuned W, b (Phase 1)
│   │   ├── learned_projection.py     # Trained projection (Phase 2)
│   │   └── consistency.py            # Compute ||x - x̂||
│
│   ├── fusion/
│   │   ├── risk_model.py       # Logistic regression / NN risk model
│   │   └── loss.py             # Loss functions for training
│
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, AUC, MSE
│   │   └── correlation.py      # Feature correlation analysis
│
│   └── utils/
│       ├── logging.py          # Structured logging
│       └── helpers.py          # Utility functions
│
├── notebooks/
│   └── exploration.ipynb       # Exploratory data analysis
│
├── requirements.txt
└── README.md
```

---

## Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
spacy>=3.5.0
scipy>=1.10.0
nltk>=3.8.0
vaderSentiment>=3.3.2
```

**Installation:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

### Phase 1: Test with Dummy Data

```python
from src.main import run_pipeline

# Simulate a session
explicit_vector = [4, 4, 2]  # [mood, sleep, stress]
text_input = "I feel completely broken and nothing ever goes right"
latencies = [1200, 1500, 1800, 2000]  # milliseconds

risk_score = run_pipeline(
    explicit=explicit_vector,
    text=text_input,
    latencies=latencies
)

print(f"Risk Score: {risk_score:.2f}")
```

**Output:**
```
Extracting text features...
Computing behavioral features...
Projecting implicit → explicit...
Consistency: 2.14
Risk Score: 8.79
```

---

### Phase 2: Train on Real Sessions (After Data Collection)

```python
from src.projection.learned_projection import train_projection
from src.fusion.risk_model import train_risk_model

# Load 200+ labeled sessions
sessions = load_sessions("data/raw_sessions/")

# Train projection
W, b = train_projection(sessions)

# Train risk model
model = train_risk_model(sessions, labels="PHQ9_score")

# Evaluate
metrics = evaluate_model(model, test_sessions)
print(f"AUC: {metrics['auc']:.3f}")
print(f"MSE: {metrics['mse']:.3f}")
```

---

## Key Design Decisions

### Why DistilBERT Instead of BERT?

- **6× faster inference** (~40ms vs 250ms per sentence)
- **50% smaller** (66M vs 110M parameters)
- **Retains 97% of BERT's performance** on text understanding tasks

For real-time mental health assessment, latency matters. DistilBERT provides the best speed-quality tradeoff.

### Why Not Use PCA or XGBoost Yet?

**Phase 1 principle:** Build deterministic infrastructure first.

- PCA requires knowing the feature distribution — we don't have 200+ sessions yet
- XGBoost requires labeled training data — we're in data collection phase
- Linear models + heuristics let us validate the pipeline before adding complexity

**Phase 2:** Once real data is collected, we'll add:
- Learned projection (`W`, `b` trained on sessions)
- Ensemble risk models (XGBoost, neural networks)
- Dimensionality reduction if needed

### Why CSV/SQLite Instead of a Database?

**Prototype stage** → Simple flat files are easier to inspect and debug.

**Production stage** → Migrate to PostgreSQL or MongoDB when:
- Session count > 10,000
- Need for concurrent writes
- Deployment to cloud infrastructure

Right now, premature optimization would slow development.

---

## Current Limitations

This is an honest engineering project. Here are the current constraints:

### 1. No calibrated risk thresholds

The risk score `8.79` is **relative**, not absolute. Without a reference distribution from real sessions, we cannot say:
- What is "normal" risk?
- What is "elevated" risk?
- What threshold should trigger clinical referral?

**Resolution:** After collecting 200+ sessions with clinical ground truth (e.g., PHQ-9 scores), we can calibrate thresholds via ROC analysis.

### 2. Heuristic projection weights

The mapping `x̂^explicit = W·x^implicit + b` uses hand-tuned weights. These are educated guesses based on domain knowledge, not learned from data.

**Resolution:** Train projection on real session pairs once data is available.

### 3. No validation set

Cannot measure generalization yet — no held-out test data.

**Resolution:** After data collection, split sessions into train/val/test (70/15/15).

### 4. Dummy explicit vectors

Currently testing with `[4,4,2]` placeholders. Real system will receive these from the Flutter FSM frontend.

**Resolution:** Integration with Flutter app (next milestone).

### 5. No interpretability layer

Risk score is a scalar — doesn't explain *why* the score is high.

**Planned addition:** Feature attribution (SHAP values) to show which signals contributed most to risk.

---

## Integration with Flutter Frontend

This backend is designed to integrate with a Flutter mobile app via REST API.

### Planned API Endpoints

```
POST /api/session/start
→ Initialize session, return session_id

POST /api/session/log
→ Log FSM transition with timestamp
→ Payload: {state, response, latency, timestamp}

POST /api/session/text
→ Submit conversational text for implicit analysis
→ Payload: {session_id, text}

GET /api/session/{session_id}/risk
→ Compute and return risk score + consistency

POST /api/session/end
→ Finalize session, store to database
```

**Technology:** FastAPI (lightweight Python REST framework).

**Deployment:** Planned for Phase 2 after validation on dummy data.

---

## What Makes This System Novel

Most mental health assessment tools use **either** self-report questionnaires **or** language analysis. This system uses **both and measures their agreement**.

**Clinical intuition:** When someone's words don't match their self-report, that mismatch itself is a signal:
- High consistency + low severity → likely stable
- High consistency + high severity → clear distress signal
- Low consistency → possible minimization, alexithymia, or insight gaps

**Example:**

**Case 1:** User reports:
```
Explicit: [mood=5, sleep=5, stress=1]   # "I'm fine"
Text:    "Everything is going great, excited about the week"
→ Consistency: LOW, risk: LOW
```

**Case 2:** User reports:
```
Explicit: [mood=5, sleep=5, stress=1]   # "I'm fine"
Text:    "I feel hopeless and nothing matters anymore"
→ Consistency: HIGH, risk: ELEVATED
```

The mismatch in Case 2 is clinically meaningful — the system flags it.

---

## Research Context

This project is inspired by multimodal mental health modeling research that combines structured assessments with conversational or behavioral signals. 

Unlike many prior systems that concatenate modalities, this engine explicitly models **consistency between self-report and language signals** as a separate diagnostic feature.

The current implementation is a deterministic prototype and does not reproduce any specific neural architecture from prior literature.

---

## Evaluation Metrics (Phase 2)

Once training data is available, we will measure:

### Classification (Risk Thresholding)

```
Accuracy   = (TP + TN) / (TP + TN + FP + FN)
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1-score   = 2 · (Precision · Recall) / (Precision + Recall)
AUC-ROC    = Area under ROC curve
```

### Regression (Severity Prediction)

```
MSE        = (1/N) Σ (y - ŷ)²
MAE        = (1/N) Σ |y - ŷ|
R²         = 1 - (SS_res / SS_tot)
```

### Consistency Analysis

```
Pearson correlation between:
  - Explicit severity scores
  - Implicit feature projections
  - Ground-truth clinical assessments (PHQ-9, GAD-7)
```

---

## Development Roadmap

**Phase 1: Infrastructure (Current)**
- ✅ Text feature extraction (DistilBERT, sentiment, distortion)
- ✅ Behavioral feature aggregation (latency, variance)
- ✅ Heuristic projection
- ✅ Consistency scoring
- ✅ Basic fusion model
- ⏳ Integration with Flutter FSM (in progress)

**Phase 2: Training & Calibration**
- Collect 200+ real sessions with clinical labels
- Train learned projection (`W`, `b`)
- Train risk classification model
- Establish risk thresholds via ROC analysis
- Add evaluation metrics

**Phase 3: Production Deployment**
- Build FastAPI REST endpoints
- Deploy to cloud (AWS/GCP)
- Add user authentication & session storage
- Implement real-time inference pipeline
- Clinical validation study

**Phase 4: Advanced Features**
- Add interpretability (SHAP values)
- Multi-session longitudinal tracking
- Personalized risk models (user-specific baselines)
- Voice prosody analysis (if audio data available)

---

## References

1. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
2. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* NeurIPS Workshop. [[paper]](https://arxiv.org/abs/1910.01108)
3. Hutto, C.J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis.* ICWSM. [[paper]](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
4. Pennebaker, J.W. (2011). *The Secret Life of Pronouns: What Our Words Say About Us.* Bloomsbury Press.
5. Beck, A.T., et al. (1979). *Cognitive Therapy of Depression.* Guilford Press. — Cognitive distortion patterns
6. Kroenke, K., et al. (2001). *The PHQ-9: Validity of a brief depression severity measure.* Journal of General Internal Medicine. [[paper]](https://pubmed.ncbi.nlm.nih.gov/11556941/)
7. Spitzer, R.L., et al. (2006). *A brief measure for assessing generalized anxiety disorder: the GAD-7.* Archives of Internal Medicine. [[paper]](https://pubmed.ncbi.nlm.nih.gov/16717171/)
8. Lundberg, S.M., & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. [[paper]](https://arxiv.org/abs/1705.07874) — SHAP interpretability


## License

MIT License. See `LICENSE` for details.

---

## Disclaimer

**This is a research prototype, not a clinical diagnostic tool.**

The system produces relative risk indicators based on text and behavioral patterns, but:
- It is **not** trained on clinical populations yet
- It is **not** validated against gold-standard assessments
- It should **not** be used for medical decision-making

Always consult qualified mental health professionals for diagnosis and treatment.
