<div align="center">

# 🌱 SEED Project

## AI-Based Automatic Sperm Detection and Integrated Morphology · Motility Analysis System

<br>

**Team T.O.P** — *Technology Of Prognosis*

`PM Kim Min-ji` · `CM Ji Seung-hyun` · `QA Seo Hyeon-jun` · `ENG1 Kim Tae-kyoung` · `ENG2 Kim Hye-hyeon`

Faculty Advisor **Prof. Song Gi-won**  |  2026-1 Convergence Capstone Design I Final Presentation  |  `Ver 1.0.0`

<br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<br>

**🌐 Language:** [한국어](README.md) | **English**

</div>

> 📑 This README **mirrors the structure of the final presentation deck** ([`deliverables/[T.O.P]Seed_최종발표_v1.0.0.pptx`](deliverables/)).
> It follows the table of contents and flow of the presentation slides so the whole project can be understood at a glance.

---

## 📋 Table of Contents

| | Chapter | Contents |
|---|---|---|
| **01** | [Introduction](#01-introduction) | Company · Org chart · Project overview · Methodology |
| **02** | [Problem](#02-problem) | Problem awareness · Problem definition · Solution |
| **03** | [How](#03-how) | Actor definition · Dev environment · System Architecture · Use Case |
| **04** | [Outcome](#04-outcome) | Dataset · Model flow · Model performance · UI/UX · Demo · Impact |
| **05** | [Artifacts](#05-artifacts) | MS Project · Artifact status · Roadmap · Traceability · GitHub |
| **06** | [Reference](#06-reference) | References |
| **+** | [Appendix](#appendix) | Kinematic metrics · WHO criteria · Model details · Datasets · Limitations |
| **+** | [Technology Q&A](#-technology-choice-qa--why-it-was-built-this-way) | Condensed rationale for key technology choices |
| **+** | [Quick Start](#-quick-start) | How to run · Project structure |

---

# 01. Introduction

## 🏢 Company

> ### Realizing disease prediction with AI

**T.O.P** stands for *"Technology Of Prognosis"* — carrying the meaning of **predictive technology**.

## 👥 Org Chart

> ### Distributed team structure

| Role | Name | Responsibilities | Development part |
|---|---|---|---|
| **PM** | Kim Min-ji (김민지) | Project oversight · Schedule · Progress management | Motility analysis model |
| **CM** | Ji Seung-hyun (지승현) | Dev-environment standardization · Artifact configuration · Source integration | Data collection & preprocessing |
| **QA** | Seo Hyeon-jun (서현준) | Artifact quality · System risk management · Model performance monitoring | Object detection model |
| **ENG1** | Kim Tae-kyoung (김태경) | System architecture design/build · Integration · Performance improvement | Morphology analysis model |
| **ENG2** | Kim Hye-hyeon (김혜현) | System architecture design/build · Integration · Performance improvement | Web page implementation |

## 🌱 Project Overview

> ### An AI-based automatic sperm detection and motility analysis system

**SEED** stands for *"Sperm Evaluation and Embryo Development."*

The system automatically detects and tracks sperm in a video and **quantitatively evaluates** their state.

## 🔄 Methodology

> ### A Waterfall methodology with staged artifacts and verification steps

```
Proposal  →  Analysis  →  Design  →  Implementation  →  Testing  →  Completion
```

At each stage, artifacts are produced and pass a verification step before proceeding.

---

# 02. Problem

## 🔍 Problem Awareness

> ### Male factors account for ~50% of all infertility, and diagnoses are rising faster

- About **50%** of all infertility causes are male or combined factors — infertility is not solely a female issue.
- In 2024, domestic male infertility diagnoses **surpassed 100,000**, rising faster than for women.
- Change in domestic infertility diagnoses (2020 → 2024): **Men +36.9%** · Women +28.5%

<br>

> ### Manual reading has limits in result consistency and quantitative analysis

An examiner observes each sperm in the microscopy video manually and records its state.

| # | Limitation | Detail | Result |
|---|---|---|---|
| 1 | **Examiner subjectivity** | Results vary with skill/condition; the same sample can yield different results | → hard to achieve consistent evaluation |
| 2 | **Long analysis time** | Manually observing hundreds\~thousands of sperm takes 30 min\~1 hr | → low examination efficiency |
| 3 | **Quantification limits** | Motility/trajectory hard to quantify; morphology relies on subjective criteria | → hard to secure objective metrics |

<br>

> ### CASA automated this, but integrated analysis and accessibility remain limited

**CASA** (*Computer-Assisted Sperm Analysis*) — a system that automates semen analysis using microscopy video and image-analysis algorithms.

| # | Limitation | Detail | Result |
|---|---|---|---|
| 1 | **Morphology limits** | Morphology reading still relies on manual, subjective expert judgment | → time cost and consistency issues |
| 2 | **Separate morphology/motion** | Morphology and motility analyzed separately, no per-same-sperm result | → integrated evaluation is hard |
| 3 | **Low accessibility** | Expensive equipment (USD 30–40k) · specialists needed | → barriers to adoption in general clinics |

<br>

> ### Consumer self-test products cannot provide precise diagnosis

| Product type | Representative | Key limitation |
|---|---|---|
| Domestic instant kit | TENGA Men's Loupe | Mostly sperm concentration · cannot measure morphology/volume/pH |
| Overseas instant kit | YO Home Sperm Test (FDA-cleared) | Cannot measure the key metric, sperm **morphology** |
| Mail-in kit | Fellow Semen Analysis | Wait time for results · morphology and other key parameters missing |

## 🎯 Problem Definition

> ### Absence of an objective, integrated, and accessible AI sperm analysis system

| # | Problem | Detail | Result |
|---|---|---|---|
| 1 | **Lack of result objectivity** | Manual-reading variance; CASA morphology depends on expert subjectivity | → time cost and consistency issues |
| 2 | **No integrated morphology/motility analysis** | Separate analysis; no per-sperm-object integrated analysis | → fragmented analysis limits holistic evaluation |
| 3 | **Low accessibility** | Expensive equipment/specialists required | → constraints for general clinics and individuals |

## ✅ Solution

> ### An AI-based automatic sperm detection and morphology·motility analysis system

| # | Contribution | Core technology | Target performance |
|---|---|---|---|
| 1 | **Quantified motility analysis** | YOLO11 detection + ByteTrack tracking + VCL·VSL etc. kinematics* | mAP@50 ≥ 0.65 · MAE ≤ 7.0 |
| 2 | **Integrated morphology/motility** | EfficientNet-B3 morphology + motility·morphology module integration | Per-part mean AUC ≥ 0.72 |
| 3 | **Improved accessibility** | Works from a standard microscopy video · web-based results without costly equipment | — |

> *Kinematics: characteristics of movement — quantifying how fast, along what path, how straight, and how widely a sperm moves.

---

# 03. How

## 🎭 Actor Definition

> ### Providing an analysis/reference aid centered on male users and medical staff

| Actor | Situation | Goal |
|---|---|---|
| **Male user** | A patient planning pregnancy who wants to check infertility in advance | Self-check without dedicated equipment/personnel |
| **Medical staff** | Clinicians needing a reference aid for semen analysis | Objective, quantitative numbers to assist reading |

## 🛠 Development Environment

> ### A stable, open-source-based AI development stack

| Category | Technology |
|---|---|
| Language | **Python** 3.10 |
| Dev tools | **JupyterLab** · Cursor IDE |
| AI framework | **PyTorch** · Ultralytics (YOLO11) · scikit-learn |
| Web · API server | **Flask** · Gunicorn |
| External deploy | **ngrok** · Render |
| Version control | **Git · GitHub** |

## 🏗 System Architecture

> ### A 4-layer AI analysis pipeline from input to output

<p align="center"><img src="docs/images/architecture.png" width="840" alt="System Architecture"></p>

**Data-flow view** — module mapping:

```
┌─────────────────────────────────────────────────────────────────┐
│  [Input layer]   microscopy video (.mp4/.avi) → quality · normalize │
│                  quality.py · normalizer.py  →  640×480 / 50fps     │
├─────────────────────────────────────────────────────────────────┤
│  [Detect·track]  YOLO11 detection  →  ByteTrack ID·trajectory       │
│                  detector.py · tracker.py                          │
├─────────────────────────────────────────────────────────────────┤
│  [Analysis]      kinematics · motility regression · morphology      │
│                  casa_features.py · analyzer.py · morphology.py     │
├─────────────────────────────────────────────────────────────────┤
│  [Interpret·out] WHO verdict · confidence · annotated video · report │
│                  interpreter.py · annotator.py · webapp/           │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Use Case

> ### A user-centered service flow from video upload to result delivery

<p align="center"><img src="docs/images/usecase.png" width="840" alt="Use Case"></p>

When a user uploads a microscopy video on the web, the server runs the analysis asynchronously
and returns a result screen with motility, morphology, and kinematic metrics plus a confidence score.

---

# 04. Outcome

## 🗂 Dataset Description

> ### Using authoritative public datasets as sources — 3 by purpose

Total raw scale: **105 videos** (VISEM-Tracking 20 · VISEM 85), ~**29,000** annotated frames, **1,540** morphology crop images.

| Purpose | Dataset | Provider | Data type |
|---|---|---|---|
| Detection·tracking | **VISEM-Tracking** | SimulaMet · OsloMet (2023) | Microscopy video (640×480, 50fps) + bounding boxes·tracking IDs |
| Motility analysis | **VISEM** (original) | SimulaMet · OsloMet (2019) | Microscopy video + motility measurements (CSV), 85 participants |
| Morphology analysis | **MHSMA** | Javadi & Mirroshandel (2019) | Sperm crop images (128×128) 1,540 + labels for 4 parts |

- **VISEM family** — published in *Nature Scientific Data* · CC BY 4.0. Adopted as an international standard (MediaEval) → comparable under the same conditions as world-leading research.
- **MHSMA** — a standard benchmark for sperm morphology analysis. Its unstained microscopy condition matches real analysis videos.

## ⚙️ Model Flow

> ### A sperm analysis pipeline of 5 core algorithms

<p align="center"><img src="docs/images/pipeline.png" width="840" alt="Model flow"></p>

| # | Stage | Model / Algorithm | Role |
|---|---|---|---|
| ① | Object detection | **YOLO11** | Per-frame sperm detection |
| ② | Object tracking | **ByteTrack** | Restore same-sperm IDs·trajectories |
| ③ | Kinematics | CASA metric computation | Kinematic metrics (VCL/VSL/ALH etc.) |
| ④ | Motility analysis | **Ridge + RandomForest ensemble** | Predict progressive·non-progressive·immotile ratios |
| ⑤ | Morphology analysis | **EfficientNet-B3** | Classify head·acrosome·vacuole·tail (4 parts) |

## 📊 Model Performance

> ### Exceeded targets on every metric — motility·detection·morphology ✅

| Item | Metric | Result | Target | Meaning |
|---|---|---|---|---|
| **Motility** | MAE | **6.90 %p** | ≤ 7.0 | Mean difference between predicted motility (%) and measured value |
| **Detection** | mAP@50 | **0.677** | ≥ 0.65 | Rate of correctly detecting sperm at the right location |
| **Morphology** | Mean AUC | **0.727** | ≥ 0.72 | Ability to distinguish normal/abnormal per sperm part |

- Motility MAE 6.90 %p — a **0.41 %p reduction vs. the prior best under the same conditions, motilitAI (7.31 %p)** (5-Fold CV).
- Both the detection and morphology models exceeded their targets.

<p align="center"><img src="docs/images/performance_motility.png" width="780" alt="Motility performance — MAE comparison"></p>
<p align="center"><img src="docs/images/performance_detection_morphology.png" width="840" alt="Detection mAP@50 · Morphology AUC"></p>

> 📈 For per-version comparison, cross-validation, and domain-adaptation experiments, see [`docs/performance.md`](docs/performance.md).

## 🖥 Application UI/UX

> ### A user-centered service flow from video upload to result delivery

<p align="center"><img src="docs/images/app_ui.png" width="900" alt="Application UI/UX — upload·analysis·result screens"></p>

**① Upload screen** → **② Analysis screen** (async progress) → **③ Result screen** (metrics·annotated video)

Implemented as a Flask-based web application (`webapp/` · `app.py`).

## 🎬 Demo Video

The full analysis process from video upload to result delivery is demonstrated in the web demo.

<p align="center">
  <a href="docs/images/demo.mp4">
    <img src="docs/images/demo_poster.png" width="780" alt="Demo video — click to play">
  </a>
</p>

> ▶ **Click the image to play the demo video (`docs/images/demo.mp4`).** (See [Quick Start](#-quick-start) below for how to run the demo.)

## 🌟 Impact

> ### Accurate, integrated — enabling analysis access for everyone

| # | Effect | Detail |
|---|---|---|
| 1 | **Objectivity** | Consistent results independent of examiner skill/condition · objective, quantitative reading aid |
| 2 | **Integrated analysis** | Simultaneous per-same-sperm morphology·motility evaluation → holistic evaluation previously impossible |
| 3 | **Accessibility** | Analysis from a standard microscopy video without costly equipment → extends even to self-check settings |

---

# 05. Artifacts

## 📅 MS Project

> ### Project progress 100%

Waterfall stage schedules were managed with MS Project, completed at **100%** progress.
Original file: [`deliverables/[PM]T.O.P_Ms_Project_v1.0.0.mpp`](deliverables/)

## 📦 Artifact Status

> ### 21 artifacts total

| Stage | Artifact | Author |
|---|---|---|
| **Proposal** | Team formation · Logo selection · Project proposal | PM Kim Min-ji |
| | Artifact management plan | CM Ji Seung-hyun |
| **Analysis** | Configuration management plan | CM Ji Seung-hyun |
| | Risk management plan | ENG1·ENG2 |
| | Quality management plan · Requirements review | QA Seo Hyeon-jun |
| | Initial project development plan | PM Kim Min-ji |
| | Requirements analysis | ENG1·ENG2 |
| **Design** | Basic design · Detailed design | ENG1·ENG2 |
| **Implementation** | Source-code documentation | CM Ji Seung-hyun |
| **Testing** | Unit test plan·result · Integration test plan·result | QA Seo Hyeon-jun |
| **Completion** | Development completion report | PM Kim Min-ji |
| | User manual | ENG1·ENG2 |
| **Other** | Final artifact template | CM Ji Seung-hyun |
| | MS Project | PM Kim Min-ji |

> 📄 The final integrated artifacts are preserved in [`deliverables/[CM]T.O.P_통합산출물_v1.0.0.hwp`](deliverables/) (PDF included).

## 🚀 Roadmap

> ### Faster processing · low-concentration handling · WHO 6th Ed. parameter expansion

| # | Direction | Detail |
|---|---|---|
| 1 | **Faster processing** | Parallelize video preprocessing · real-time processing → shorter wait time |
| 2 | **Low-concentration handling** | Compensate accuracy drop when sperm count < 10 · refine confidence penalties · improve re-capture guidance |
| 3 | **Parameter expansion** | Add currently-unmeasured items (concentration·total count·vitality) → higher WHO 6th Ed. completeness |

## 🔗 Teams Traceability

The whole project's schedule, artifacts, and issues were shared and tracked via **MS Teams**, securing configuration management and collaboration traceability.

## 💻 GitHub

> Source code and system configuration management: **https://github.com/MoriochoRadio/seed-project**

---

# 06. Reference

<details>
<summary><b>View all references (20)</b></summary>

1. Leslie, S. W., et al. (2024). *Male infertility.* StatPearls.
2. National Health Insurance Service. (2024). *Infertility diagnoses over the past 5 years.* National audit data.
3. Liang, Y., et al. (2025). *Global, regional, and national prevalence and trends of infertility (1990–2021).* Human Reproduction, 40(3), 529–544.
4. Bieniek, J. M., et al. (2021). *A Novel Approach to Improving the Reliability of Manual Semen Analysis.* PMC.
5. Siddharth, K., et al. (2023). *Interobserver Variability in Semen Analysis.* Cureus, 15(10), e46388.
6. Barcena, P., et al. (2025). *AI approaches to male infertility in IVF: a mapping review.* PMC.
7. Chawre, S., et al. (2024). *A Review of Semen Analysis: Updates From the WHO Sixth Edition Manual.* Cureus, 16(6), e63485.
8. Mortimer, D., & Mortimer, S. T. (2015). *The future of computer-aided sperm analysis.* Asian Journal of Andrology, 17(4), 545–553.
9. Finelli, R., et al. (2021). *The validity and reliability of computer-aided semen analyzers: a systematic review.* Translational Andrology and Urology, 10(7), 3069–3079.
10. Gonzalez, D., et al. (2021). *Clinical Update on Home Testing for Male Fertility.* World Journal of Men's Health, 39(4), 615–625.
11. Kobori, Y., et al. (2016). *Novel device for male infertility screening with single-ball lens microscope and smartphone.* Fertility and Sterility, 106(3), 574–578.
12. U.S. FDA. (2016). *K161493: YO Home Sperm Test (510(k)).*
13. Samplaski, M. K., et al. (2021). *Development and validation of a novel mail-in semen analysis system.* Fertility and Sterility, 115(4), 922–929.
14. Thambawita, V., et al. (2023). *VISEM-Tracking, a human spermatozoa tracking dataset.* Data in Brief, 47, 108944.
15. Hicks, S. A., et al. (2019). *Machine learning-based analysis of sperm videos and participant data for male fertility prediction.* Scientific Reports, 9, 16770.
16. Javadi, S., & Mirroshandel, S. A. (2019). *A novel deep learning method for automatic assessment of human sperm morphology.* Computers in Biology and Medicine, 109, 182–194.
17. World Health Organization. (2021). *WHO laboratory manual for the examination and processing of human semen* (6th ed.).
18. Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
19. Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019, PMLR 97, 6105–6114.
20. Miahi, E., et al. (2019). *Genetic Neural Architecture Search for automatic assessment of human sperm images.* arXiv:1909.09432.

</details>

---

# Appendix

<details>
<summary><b>📐 Kinematic metrics ① — Velocity</b> (how fast and straight a sperm moves)</summary>

<br>

| Metric | Name | Definition | Computation |
|---|---|---|---|
| **VCL** | Curvilinear velocity | Speed along the actual winding path | actual path length ÷ time |
| **VSL** | Straight-line velocity | Speed based on start → end straight distance | straight displacement ÷ time |
| **VAP** | Average-path velocity | Speed along a smoothed average path | smoothed path ÷ time |

> Computed right after tracking (ByteTrack) · unit µm/s · based on 50 fps

</details>

<details>
<summary><b>📐 Kinematic metrics ② — Linearity·oscillation</b> (what pattern it moves in)</summary>

<br>

| Metric | Name | Meaning | Computation |
|---|---|---|---|
| **LIN** | Linearity | How straight the path is | VSL ÷ VCL |
| **STR** | Straightness | Straight ratio vs. average path | VSL ÷ VAP |
| **WOB** | Wobble | Average path vs. actual path | VAP ÷ VCL |
| **ALH** | Amplitude of lateral head displacement | Side-to-side swing width (µm) | lateral deviation vs. average path |
| **BCF** | Beat-cross frequency | Tail crosses per second (Hz) | lateral-signal zero crossings ÷ time |
| **PAW** | Mean lateral width | Mean side-to-side path width (µm) | mean of lateral deviation |

</details>

<details>
<summary><b>🏃 Motility grades & WHO criteria</b></summary>

<br>

| Grade | Name | Description |
|---|---|---|
| **PR** | Progressive motility | Moves straight forward well — most important for fertilization |
| **NP** | Non-progressive motility | Moves but inefficiently (in place, circular, etc.) |
| **IM** | Immotile | Barely moves |

> **WHO normal criteria** — total motility (PR+NP) ≥ 40% · progressive (PR) ≥ 32%

</details>

<details>
<summary><b>🤖 The 4 AI models & training</b></summary>

<br>

| # | Role | Model | Description | Training |
|---|---|---|---|---|
| 1 | Object detection | **YOLO11** | Quickly finds and boxes small sperm | VISEM-Tracking videos split into frames → supervised box learning |
| 2 | Object tracking | **ByteTrack** | Same ID across frames → trajectory restoration | No separate training (rule-based box linking) |
| 3 | Motility analysis | **Ridge + RF ensemble** | Predicts motility % from trajectory features; averaged for stability | Kinematic features of trajectories → regression on VISEM clinical motility (CSV) |
| 4 | Morphology analysis | **EfficientNet-B3** | Judges 4-part normality from the sperm image | MHSMA 1,540 crops, 4-part classification + VisemStyleAugment augmentation |

</details>

<details>
<summary><b>📏 Meaning of the metrics</b> (mAP · MAE · AUC)</summary>

<br>

| Metric | Meaning | Direction | This system |
|---|---|---|---|
| **mAP@50** | How accurately sperm are found (detection accuracy) | higher is better ↑ | detection **0.677** |
| **MAE** | Mean difference between predicted motility% and actual | lower is better ↓ | motility **6.9 %p** |
| **AUC** | Normal/abnormal discrimination (0.5 = chance, 1.0 = perfect) | higher is better ↑ | morphology **0.727** |

</details>

<details>
<summary><b>🔬 Morphology — the 4 sperm parts</b></summary>

<br>

| Part | English | Description |
|---|---|---|
| **머리** | head | Shape/size normality |
| **첨체** | acrosome | Enzyme sac at the front of the head — needed to penetrate the egg |
| **공포** | vacuole | Empty space inside the head — a defect if present |
| **꼬리** | tail | The part responsible for movement |

> **Normal morphology rate** — proportion of sperm normal in all 4 parts · WHO ≥ 4% normal · reference-only due to domain gap

</details>

<details>
<summary><b>📊 Confidence score · Video normalization · 5-second analysis</b></summary>

<br>

**Confidence score (0–100)** — how much the analysis result can be trusted
- Purpose: quantify result reliability on a 0–100 scale
- Penalties: few sperm (N<10) · unstable detection · short tracking
- Use: below 60 outputs a **'re-capture recommendation'**
- Example: a video with only 2 detected sperm → confidence 50 → re-capture recommended

**Video normalization** — automatically converts videos of various resolutions/frame rates to the training domain (640×480·50fps) so the model behaves consistently

**5-second (250-frame) analysis** — motility·CASA statistics converge stably over short spans. Clinical CASA also analyzes short fields → ensures speed·consistency

</details>

<details>
<summary><b>⚠️ System limitations</b></summary>

<br>

| Item | Limitation |
|---|---|
| **Morphology** | Reference-only numbers due to the training↔application domain gap (lower confidence than motility) |
| **Low quality·concentration** | Few sperm or blurry video can cause error → mitigated by the confidence score |
| **Purpose** | An assistive/reference tool; does not replace medical diagnosis |

</details>

---

## 💬 Technology Choice Q&A — Why It Was Built This Way

> A condensed, answer-ready summary of the technology-choice rationale scattered across the main text and the Appendix. See each link for the detailed grounds.

**Q1. Why the YOLO11 + ByteTrack combination for detection and tracking?**
YOLO11 rapidly detects small sperm frame by frame, and ByteTrack recovers each sperm's identity and trajectory purely through detection-box association rules, with no extra training. Those trajectories are what make kinematic metrics such as VCL and VSL computable. → [Appendix — 4 AI models & training](#appendix)

**Q2. Why a Ridge + RandomForest ensemble for motility prediction?**
The task is regressing kinematic features from the tracked trajectories onto VISEM's clinical motility values, and averaging the two models stabilizes the prediction. The result was an MAE of 6.90 %p, lower than motilitAI (7.31 %p), the prior study under identical conditions. → [Model Performance](#-model-performance)

**Q3. Why choose the VISEM family and MHSMA as datasets?**
The VISEM family is a standard dataset published in *Nature Scientific Data* and adopted by MediaEval, enabling comparison with the world's best research under identical conditions; MHSMA is the standard benchmark for morphology analysis whose unstained, microscope conditions match our actual analysis videos. → [Dataset Description](#-dataset-description)

**Q4. Why analyze only 5 seconds (250 frames) instead of the whole video?**
Motility and CASA statistics converge stably even over short segments, and clinical CASA likewise analyzes short fields, so this choice secures both speed and consistency. → [Appendix — Confidence score · Video normalization · 5-second analysis](#appendix)

**Q5. Why are the morphology numbers limited to "reference only"?**
Because of the domain gap between the training data (MHSMA) and the applied videos, morphology is less reliable than motility, so it is explicitly marked reference-only; errors on low-quality or low-concentration videos are mitigated by the confidence score (re-recording recommended below 60 points). → [Appendix — System limitations](#appendix)

---

# 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/MoriochoRadio/seed-project.git
cd seed-project

# 2. Create a virtual environment and install dependencies
conda create -n seed python=3.10
conda activate seed
pip install -r requirements.txt
# Install PyTorch separately for your environment (see the comment at the top of requirements.txt)

# 3. Prepare model weights
#   - Motility/morphology models are included under models/
#   - YOLO11 weights (best.pt) auto-download from Google Drive
#     (after setting GDRIVE_* env vars) python setup_models.py

# 4. Run the web demo
python app.py          # http://localhost:5000
```

> Deployment is configured for Render.com via [`render.yaml`](render.yaml).

## 📂 Project Structure

```
seed-project/
├── app.py            ← Flask entry point
├── src/              ← core analysis modules
│   ├── detector.py       (YOLO11 sperm detection)
│   ├── tracker.py        (ByteTrack tracking + motility features)
│   ├── casa_features.py  (CASA kinematic metrics)
│   ├── analyzer.py       (motility ensemble regression)
│   ├── morphology.py     (EfficientNet-B3 morphology analysis)
│   ├── interpreter.py    (WHO-based verdict · confidence)
│   ├── normalizer.py     (video normalization) · quality.py (quality assessment)
│   ├── annotator.py      (annotated video generation) · pipeline.py (integrated pipeline)
│   └── ...
├── webapp/           ← Flask web application (routes · templates · static)
├── models/           ← trained model weights (motility ensemble, morphology v3)
├── notebooks/        ← experiment·training notebooks (01~16)
├── docs/             ← project docs (architecture · performance · guides)
├── deliverables/     ← final presentation materials & official artifacts (v1.0.0)
└── data/             ← datasets (gitignored)
```

## 📚 More

- **System design** — [`docs/architecture.md`](docs/architecture.md)
- **Detailed performance** — [`docs/performance.md`](docs/performance.md)
- **User / Admin guides** — [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) · [`docs/ADMIN_GUIDE.md`](docs/ADMIN_GUIDE.md)
- **Final presentation · artifacts** — [`deliverables/`](deliverables/)

---

## 📄 License

This project is an **academic capstone project** and is **not a clinical diagnostic tool**.
The distribution license is [`LICENSE`](LICENSE) (MIT).

<div align="center">

**Team T.O.P** · *Technology Of Prognosis* · 2026

</div>
