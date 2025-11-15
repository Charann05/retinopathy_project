
#  Diabetic Retinopathy Detection (AI Project) 🩺

A simple AI tool that uses **EfficientNet-B0** and **Streamlit** to predict the severity of **Diabetic Retinopathy (DR)** from retinal fundus images.

---

##  Features

* Upload fundus images (JPG/PNG)
* Predicts DR severity:
  **No DR · Mild · Moderate · Severe · Proliferative**
* Color-coded results
* Clean UI, works offline

---

##  Model Details

* **Architecture:** EfficientNet-B0
* **Model:** `tf_efficientnet_b0.ns_jft_in1k`
* **Framework:** PyTorch + timm


---

##  Project Structure

```
ftend.py
best_model.pth
src/
 ├── __init__.py
 ├── data.py
 ├── evaluate.py
 ├── infer.py
 ├── model.py
 └── train.py
data/
 ├── images/
 ├── labels_train.csv
 └── labels_val.csv

```

---

##  How to Run

### 1️⃣ Install Requirements

```
pip install -r requirements.txt
```

### 2️⃣ Install PyTorch (choose one)

**CPU:**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.1):**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3️⃣ Place Your Model

You will obtain **best_model.pth** after training your model (run `train.py`).
Place this file in the **same folder as `ftend.py`**.

### 4️⃣ Start the App

```
streamlit run ftend.py
```

---

##  How It Works

* Upload or search for an image
* AI analyzes the retina
* DR severity is shown with a color label
* Option to test more images



---
##  Purpose

This project is for **learning and research** only<br>
It is **not for medical diagnosis**.

---




