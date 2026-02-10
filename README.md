# kuznetsov_nndl_hw2
## Prompt 

> You are a senior full-stack ML engineer specializing in browser-based machine learning with TensorFlow.js. Generate a complete, production-ready web application that trains a shallow binary classifier on the Kaggle Titanic dataset directly in the browser (no backend). The project must be deployable to GitHub Pages.
>
> Output exactly TWO files:
>
> index.html – HTML layout + UI + minimal CSS  
> app.js – all JavaScript / TensorFlow.js logic
>
> Do NOT embed JavaScript inside index.html except for linking app.js.
>
> Use these CDNs:
>
> TensorFlow.js  
> https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest
>
> tfjs-vis  
> https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest
>
> index.html must load app.js explicitly.
>
> All code comments must be in English.
>
> The app must be interactive and reusable for similar CSV datasets (clearly mark schema swap points in comments).

---

## Overall Goal

Build a shallow neural network binary classifier (single hidden layer) trained entirely in the browser using TensorFlow.js on the Titanic dataset.

Everything runs client-side:
- No backend
- No API
- No server
- GitHub Pages compatible

---

## User Workflow

The UI guides the user through the following steps:

1. Loading CSV files  
2. Inspecting the dataset  
3. Preprocessing features  
4. Training the model  
5. Evaluating performance  
6. Tuning classification threshold  
7. Generating predictions  
8. Exporting results  
9. Downloading the trained model  

---

## index.html — Requirements

### Layout Sections

1. Data Load
   - File inputs for train.csv and test.csv
   - Load button
   - Status messages

2. Data Inspection
   - Preview table (first rows)
   - Dataset shape
   - Missing value percentages
   - Bar charts:
     - Survival by Sex
     - Survival by Pclass  
     *(tfjs-vis)*

3. Preprocessing

4. Model

5. Training
   - Train button

6. Metrics
   - ROC curve plot
   - Threshold slider (0–1)
   - Confusion matrix
   - Precision / Recall / F1 (live updates)

7. Prediction

8. Export
   - Download submission.csv
   - Download probabilities.csv
   - Download trained model

### Styling

- Basic CSS
- Responsive layout
- Mobile-friendly

### Deployment Note

> Create a public GitHub repository, commit index.html and app.js, enable GitHub Pages (main/root), then open the provided URL.

---

## app.js — Functional Requirements

### CSV Loading

- Support both fetch() and file input
- Fix CSV comma escape / quoted field issues
- Handle malformed rows gracefully
- Show alerts for invalid or missing files

---

### Data Schema

Target
- Survived (0 / 1)

Features
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

Identifier (excluded from training)
- PassengerId

Optional Derived Features (UI toggle)
- FamilySize = SibSp + Parch + 1
- IsAlone = FamilySize === 1

> Schema swap points must be clearly commented for reuse with other datasets.

---

### Data Inspection

After loading:
- Display preview table
- Show number of rows and columns
- Calculate missing percentage per column
- Plot Survival vs Sex and Survival vs Pclass using tfjs-vis bar charts

---

### Preprocessing

- Age → median imputation
- Embarked → mode imputation
- Standardize Age and Fare
- One-hot encode:
  - Sex
  - Pclass
  - Embarked
- Print final tensor shapes and feature names

---

### Model Architecture

Using tf.sequential():

- Dense(16, activation = relu)
- Dense(1, activation = sigmoid)

Compile
- Optimizer: adam
- Loss: binaryCrossentropy
- Metrics: accuracy

Print model summary.

---

### Training

- 80 / 20 stratified split
- Epochs: 50
- Batch size: 32
- Early stopping on val_loss (patience = 5)
- Live loss & accuracy plots via tfjs-vis.fitCallbacks

---

### Metrics & Evaluation

After training:
- Compute ROC curve and AUC manually
- Plot ROC curve

#### Threshold Slider (0–1)

Dynamically recompute:
- Confusion matrix
- Precision
- Recall
- F1 score

> IMPORTANT: Fix evaluation table rendering — ensure DOM elements are properly created and updated.

---

### Feature Importance (Sigmoid Gate)

- Add a trainable sigmoid feature gate before the Dense layer
- Gate learns per-feature importance
- Multiply gate output with input tensor

After training:
- Extract gate values
- Display feature importance:
  - Table
  - Bar chart

---

### Inference & Export

On test.csv:
- Generate probabilities
- Apply selected threshold

Create:
- submission.csv  
  (PassengerId, Survived)
- probabilities.csv  
  (PassengerId, Probability)

Allow model download:

---

### Buttons Required

- Load Data
- Train Model
- Evaluate
- Predict
- Export

Add clear error handling for invalid steps.

---

## Code Summary Requirement

At the end of app.js, include a multiline comment explaining:
- Data flow
- Preprocessing pipeline
- Model architecture
- Training logic
- Evaluation logic
- Export logic

---

## Final Output

Return ONLY:
- index.html
- app.js

No explanations outside code.
