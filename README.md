# 🌿 Plant Disease Detection using Deep Learning (ResNet50 + Transfer Learning)

> I’ve always been fascinated by how technology can solve real-world problems. Agriculture is one area where a little bit of AI can make a huge difference. I built this project to explore how deep learning can help detect plant diseases from leaf images — and honestly, the results blew me away. After fine-tuning ResNet50 on the PlantVillage dataset, I achieved **98.78% validation accuracy** across 38 disease classes, deployed as an interactive Gradio web app anyone can use.

---

## 📌 Table of Contents

- [Why I Built This](#why-i-built-this)
- [Demo](#demo)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [My Approach — Step by Step](#my-approach--step-by-step)
  - [Step 1 — Data Loading & Preprocessing](#step-1--data-loading--preprocessing)
  - [Step 2 — Dealing with Class Imbalance](#step-2--dealing-with-class-imbalance)
  - [Step 3 — Building the Model with ResNet50](#step-3--building-the-model-with-resnet50)
  - [Step 4 — Phase 1: Training the Head](#step-4--phase-1-training-the-head)
  - [Step 5 — Phase 2: Fine-Tuning](#step-5--phase-2-fine-tuning)
  - [Step 6 — Evaluating the Model](#step-6--evaluating-the-model)
  - [Step 7 — Visualizations & Explainability](#step-7--visualizations--explainability)
  - [Step 8 — Deploying with Gradio](#step-8--deploying-with-gradio)
- [Results](#results)
- [Sample Visualizations](#sample-visualizations)
- [How to Run This Yourself](#how-to-run-this-yourself)
- [File Structure](#file-structure)
- [What I Learned](#what-i-learned)
- [What I'd Do Differently Next Time](#what-id-do-differently-next-time)
- [Connect With Me](#connect-with-me)

---

## Why I Built This

When I first came across the PlantVillage dataset, I felt this was a genuinely impactful problem worth solving. Plant diseases cost farmers billions every year, and most detection today still relies on manual inspection by experts — which is slow, expensive, and often unavailable in rural areas.

I wanted to see if I could build something that's not just accurate on paper, but that a real person could actually use — upload a photo, get an answer. That's what this project is about.

I also wanted to go beyond just training a model. I wanted to understand *why* it makes certain predictions, so I added Grad-CAM visualizations and t-SNE plots to peek inside the model's reasoning.

---

## Demo

> Web Interface
I also built a simple **Gradio-based UI** where users can upload a leaf image and get predictions in real time.

This made the project feel more like a complete application rather than just a model.

The app will tell you what disease (if any) it detects, along with a confidence percentage. No setup needed.

<!-- PLACEHOLDER: Add a screenshot of the Gradio web app here -->
> <img width="1920" height="920" alt="Plant Disease Detection_gradio_blank" src="https://github.com/user-attachments/assets/ab08cd3d-4af3-4307-a3d9-5642cadb1481" />
> <img width="1920" height="920" alt="Plant Disease Detection_gradio_leaf" src="https://github.com/user-attachments/assets/cfdfdd54-802a-4761-ad05-3785e271d284" />



---

## Dataset

I used the **PlantVillage** dataset, loaded directly via TensorFlow Datasets. It's a well-known benchmark dataset in plant pathology research.

| Property          | Value          |
|------------------|----------------|
| Total Samples     | 54,303         |
| Number of Classes | 38             |
| Image Size Used   | 96 × 96 pixels |
| Train Split       | 80% (43,442)   |
| Validation Split  | 20% (10,861)   |

It covers diseases and healthy states across crops like **tomato, potato, apple, corn, grape**, and several others.

One thing I noticed pretty early on was that the dataset is **significantly imbalanced** — some disease classes have thousands of samples while others have just a few dozen. I knew if I ignored this, the model would just learn to be good at common diseases and terrible at rare ones, which defeats the whole purpose. More on how I handled that below.

---

## Tech Stack

| Tool | Why I Used It |
|------|--------------|
| Python 3.12 | Language of choice for ML work |
| TensorFlow / Keras | My go-to deep learning framework — clean, well-documented |
| ResNet50 | Powerful pre-trained backbone; no need to train from scratch |
| TensorFlow Datasets | Convenient dataset loading — handles downloading automatically |
| Scikit-learn | Metrics, class weight computation, t-SNE |
| Matplotlib / Seaborn | All the visualizations |
| Gradio | Honestly, the easiest way I've found to demo an ML model |
| Kaggle GPU (Tesla T4 × 2) | Free GPU access that made training feasible |

---

## Project Architecture

Here's the full pipeline from raw data to deployed app:

```
PlantVillage Dataset (TFDS)
         │
         ▼
  Data Preprocessing
  (Resize → 96×96, ResNet Normalize)
         │
         ▼
  Class Weight Computation
  (Handle Imbalance)
         │
         ▼
  ┌─────────────────────────┐
  │  ResNet50 (ImageNet)    │
  │  ─────────────────────  │
  │  Frozen Convolutional   │
  │  Base Layers            │
  └────────────┬────────────┘
               │
               ▼
     GlobalAveragePooling2D
               │
               ▼
         Dense(128, ReLU)
               │
               ▼
          Dropout(0.4)
               │
               ▼
        Dense(38, Softmax)
               │
               ▼
       Phase 1: Train Head Only
       (8 epochs, Adam lr=1e-3)
               │
               ▼
       Phase 2: Fine-Tune Top 30 Layers
       (8 epochs, Adam lr=1e-5)
               │
               ▼
    Final Validation Accuracy: 98.78%
               │
               ▼
     Gradio Web App Deployment
```

---

## My Approach — Step by Step

### Step 1 — Data Loading & Preprocessing

Loading the data with TensorFlow Datasets was straightforward. But one choice I had to make early was the **image size**. The original images are much larger, but I settled on **96×96 pixels** — small enough to train fast on a free Kaggle GPU, large enough to retain meaningful leaf texture and spot patterns.

I also made sure to use ResNet50's built-in `preprocess_input` rather than just dividing by 255. This matters because ResNet was trained with a specific normalization, and using the wrong one would hurt performance.

```python
dataset, info = tfds.load("plant_village", split="train", as_supervised=True, with_info=True)

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    return image, label
```

I also used `cache()`, `shuffle(10000)`, and `prefetch(AUTOTUNE)` to keep the GPU fed and avoid data pipeline bottlenecks.

---

### Step 2 — Dealing with Class Imbalance

When I plotted the class distribution, I was immediately struck by how uneven it was. Some classes had over a thousand samples; others had fewer than 100. If I had just trained on raw counts, the model would essentially "cheat" by ignoring rare diseases.

My fix was **class weights** — I computed a weight for each class so that rarer classes have more influence on the loss during training. It's a simple but very effective approach.

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
```

---

### Step 3 — Building the Model with ResNet50

I chose **ResNet50** because I felt it struck the right balance — deep enough to learn complex visual features, but not so huge that it's impractical to fine-tune on a free GPU. It's also been battle-tested on image classification tasks across many domains.

Rather than training from scratch (which would need far more data and compute), I used it as a **feature extractor** with the weights frozen. Then I added my own small classifier on top:

- `GlobalAveragePooling2D` — pools the spatial features into a single compact vector
- `Dense(128, relu)` — a hidden layer to learn disease-specific patterns
- `Dropout(0.4)` — to prevent overfitting, especially given the small head
- `Dense(38, softmax)` — one output neuron per class

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False
```

---

### Step 4 — Phase 1: Training the Head

I trained just the top layers for **8 epochs** at a learning rate of **1e-3**. The frozen ResNet base acted as a fixed feature extractor while my new layers learned to classify plant diseases.

What surprised me was how quickly the validation accuracy jumped to **88% in just the first epoch**. That's the power of transfer learning — the model already "knows" a tremendous amount about images.

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|-------------|
| 1     | 56.87%        | 88.82%      |
| 8     | 87.01%        | 93.55%      |

---

### Step 5 — Phase 2: Fine-Tuning

After Phase 1 settled, I unfroze the **last 30 layers** of ResNet50 and continued training with a much lower learning rate (**1e-5**). The idea is to make tiny, careful adjustments to the pre-trained weights — not relearn from scratch, just gently nudge the features toward plant disease patterns.

I noticed a dramatic improvement here. The validation accuracy climbed from ~93% all the way to **98.78%** over 8 epochs. The model clearly benefited from being allowed to adapt its deeper features to the domain.

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|-------------|
| 1     | 70.48%        | 93.52%      |
| 8     | 96.06%        | **98.78%**  |

---

### Step 6 — Evaluating the Model

Getting high accuracy is one thing, but I wanted to really understand where the model excels and where it might struggle. So I ran a thorough evaluation:

- **Per-class Precision, Recall, F1-Score** — I was genuinely pleased to see most classes scoring above 0.97
- **Confusion Matrix** — I looked at this carefully to spot any systematic confusions between visually similar diseases
- **Top-3 Accuracy** — checks whether the true label appears in the model's top 3 guesses
- **ROC Curves** — plotted for a few sample classes to understand the sensitivity/specificity trade-off

```
Overall Accuracy:     99%
Macro Avg Precision:  98%
Macro Avg Recall:     99%
Macro Avg F1-Score:   99%
```

<!-- PLACEHOLDER: Add confusion matrix screenshot here -->
> <img width="638" height="528" alt="image" src="https://github.com/user-attachments/assets/3f3845cb-2895-4a17-9732-084de1817c1c" />

---

### Step 7 — Visualizations & Explainability

This was honestly one of my favourite parts of the entire project. I feel that a model you can't explain is hard to trust — especially in a domain like agriculture where the stakes are real and the end user is often a farmer, not a data scientist.

#### Grad-CAM Heatmaps
I implemented **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize which regions of a leaf image the model pays attention to when making a prediction. It was genuinely satisfying to see the heatmap light up around the diseased spots on the leaf — confirming that the model isn't just pattern-matching on background colours or irrelevant noise.

<!-- PLACEHOLDER: Add Grad-CAM heatmap here -->
> <img width="443" height="435" alt="Grad-CAM Heatmap" src="https://github.com/user-attachments/assets/62840939-7f07-4e8e-86da-85971af061e1" />


#### t-SNE Feature Visualization
I used t-SNE to project the model's internal feature representations down to 2D. I noticed that different disease classes form reasonably tight, well-separated clusters — which tells me the model has genuinely learned to distinguish between diseases, not just memorize the training set.

<!-- PLACEHOLDER: Add t-SNE plot here -->
> <img width="636" height="528" alt="t-SNE" src="https://github.com/user-attachments/assets/baed252d-979c-4712-b22e-2434b70c1860" />


#### Class Distribution
Plotting this early on was something of an "aha moment" for me — I could immediately see why class weighting was so necessary. Without this step, I think the results would have been much worse on the rare classes.

<!-- PLACEHOLDER: Add class distribution bar chart here -->
> <img width="859" height="393" alt="Class Distribution (Imbalance)" src="https://github.com/user-attachments/assets/f0215532-43db-47da-9098-e228242fdd3e" />


#### Training Accuracy Curves
These clearly show the jump that happens when fine-tuning kicks in — a satisfying visual confirmation that the two-phase approach really does work.

<!-- PLACEHOLDER: Add training accuracy curve here -->
> <img width="700" height="470" alt="Accuracy Improvement After Fine-Tuning" src="https://github.com/user-attachments/assets/62bfb930-0ef9-4a98-a8f1-a12936d274b7" />


#### CNN vs ResNet Comparison
I also trained a simple custom CNN as a baseline, just to put the results in perspective. Even on a small data subset, ResNet50 comfortably outperformed it — reinforcing why I reached for a pre-trained model in the first place.

---

### Step 8 — Deploying with Gradio

Once I was happy with the model, I wanted to make it accessible to someone who has never touched Python. Gradio was perfect for this — a few lines of code and you have a shareable web interface that works on any browser.

```python
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🌿 Plant Disease Detection",
    description="Upload a leaf image to detect plant disease"
)
interface.launch(share=True)
```

It generates a link where one can upload a leaf photo and get a prediction instantly — no installation, no coding.

---

## Results

| Metric              | Value        |
|--------------------|--------------|
| Validation Accuracy | **98.78%**  |
| Macro F1 Score      | **0.99**    |
| Top-3 Accuracy      | ~99%         |
| Classes             | 38           |
| Validation Samples  | 10,861       |

Honestly, I was not expecting to get this close to 99% on a 38-class problem. I think the combination of transfer learning, proper class weighting, and careful fine-tuning really paid off here.

---

## Sample Visualizations

<!-- PLACEHOLDER: Replace with actual screenshots from your notebook -->

| Visualization | What It Shows |
|--------------|---------------|
| 📊 Confusion Matrix | How often each class is predicted correctly, and where it gets confused |
| 📈 Accuracy Curves | How accuracy evolved across both training phases |
| 🔍 t-SNE Plot | How the model's internal features cluster by disease class |
| 🌡️ Grad-CAM Heatmap | Which parts of the leaf the model focuses on when predicting |
| 📉 ROC Curve | Per-class classifier performance trade-offs |


---

## How to Run This Yourself

### Prerequisites

```bash
pip install tensorflow tensorflow-datasets scikit-learn matplotlib seaborn gradio
```

### On Kaggle (Recommended — free GPU)

1. Open the notebook on Kaggle
2. Enable GPU: Settings → Accelerator → GPU T4 x2
3. Run all cells from top to bottom

### Locally

```bash
# Clone the repo
gh repo clone badrinathjt7/Plant-Disease-Detection-using-Deep-Learning-ResNet50-Transfer-Learning-
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook plant_disease_detection.ipynb
```

> ⚠️ Training locally without a GPU will be very slow. I'd strongly recommend Kaggle or Google Colab for the training steps.

---

## File Structure

```
plant-disease-detection/
│
├── plant_disease_detection.ipynb   # Main notebook (end-to-end)
├── plant_disease_model.h5          # Saved trained model weights
├── README.md                       # This file
└── requirements.txt                # All Python dependencies
```

---

## What I Learned

Looking back on this project, a few things really stood out to me:

- **Transfer learning is genuinely transformative.** Seeing 88% accuracy in just the first epoch — before the model has even been fine-tuned — really drove home how much pre-trained features can do.
- **Two-phase training is worth the extra complexity.** I initially considered just training end-to-end from the start, but the staged approach was far more stable and gave meaningfully better results.
- **Class imbalance is easy to overlook and costly to ignore.** Once I saw the distribution plot, it was obvious why simply ignoring it would have tanked performance on rare diseases — which are arguably the most critical ones to catch.
- **Explainability matters as much as accuracy.** I feel strongly that a model achieving 99% accuracy but offering no insight into its reasoning is less useful than one at 95% that you can actually understand, verify, and trust.
- **Gradio is a game-changer for demos.** It took me about 10 minutes to go from "trained model saved to disk" to "shareable web app with a public URL."

---

## What I'd Do Differently Next Time

- [ ] Add **data augmentation** (random flips, rotations, brightness/contrast changes) — I think this would help the model generalize better to real-world, non-lab photos
- [ ] Try **EfficientNet** or a **Vision Transformer (ViT)** backbone — I'm genuinely curious whether the accuracy ceiling can be pushed even higher
- [ ] Deploy to **Hugging Face Spaces** for permanent, free hosting instead of a temporary Gradio share link
- [ ] Build a **TensorFlow Lite** version for mobile devices — that's where farmers actually are, out in the field
- [ ] Collect some **real-world field images** to test how well the model holds up outside controlled lab conditions
- [ ] Explore **multi-label classification** — in practice, a single leaf can show symptoms of more than one disease at a time

---

## Connect With Me

I'd love to hear your thoughts on this project — whether it's feedback, questions, or ideas for collaboration. Feel free to reach out!

<!-- PLACEHOLDER: Add your social media handles below -->

| Platform     | Handle / Link |
|-------------|--------------|
| 🐙 GitHub    | [Your GitHub Profile URL] |
| 💼 LinkedIn  | [Your LinkedIn URL] |
| 🐦 Twitter/X | [@YourHandle] |
| 📧 Email     | your.email@example.com |
| 🤗 Kaggle    | [Your Kaggle Profile URL] |

---

<p align="center">
  Built with curiosity, a lot of GPU hours, and genuine care for the problem 🌿
</p>
