# ✍ Signature Verification App

This is a **Streamlit-based** web application for **offline signature verification** using a **Siamese Network with ResNet-18** as the feature extractor.

---

## 📌 Features

- Upload **two signature images** — the original and the one to verify.
- Model calculates the **feature embeddings** for both signatures and computes the **Euclidean distance** between them.
- A decision is made:
  - ✅ **Genuine** – if the distance is below the threshold.
  - ❌ **Forged** – if the distance exceeds the threshold.
- Works entirely in the browser interface, no coding needed.

---

## 🧠 Model Details

- **Architecture:** Siamese Network with ResNet-18 backbone.
- **Training:** Model was trained on a signature dataset and fine-tuned for verification.
- **Threshold:** `0.4275` (chosen based on validation set performance).
- **Frameworks Used:**  
  - PyTorch for model definition & inference  
  - Torchvision for transformations  
  - Streamlit for web interface  

---

## 🚀 How to Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/bitsbuild/signature-verification-interface.git
cd signature-verification-interface
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add the trained model

Download the trained model from the **model repository**:
🔗 [https://github.com/OdhakByJash/secure-signature-verification](https://github.com/OdhakByJash/secure-signature-verification)

Place the model weights file here:

```
trained_model_parameters/siamese_resnet_18.pth
```

### 4️⃣ Start the app

```bash
streamlit run interface.py
```

---

## 📂 Project Structure

```
.
├── interface.py               # Main UI for signature upload & prediction
├── pages/
│   └── result_page.py         # Displays result (Genuine / Forged)
├── trained_model_parameters/
│   └── siamese_resnet_18.pth  # Trained model weights (not included in repo)
├── requirements.txt           # Python dependencies
```

---

## 🌐 Live Demo

You can try the deployed app here:
[**🔗 Signature Verification App**](https://sigver.streamlit.app/)

---

## 🔗 Related Repositories

* **Frontend (Streamlit interface):**
  [https://github.com/bitsbuild/signature-verification-interface](https://github.com/bitsbuild/signature-verification-interface)
* **Model training code & weights:**
  [https://github.com/OdhakByJash/secure-signature-verification](https://github.com/OdhakByJash/secure-signature-verification)
* **Backend (API-based signature verification/authentication):**
  [https://github.com/bitsbuild/signature-based-authentication](https://github.com/bitsbuild/signature-based-authentication)
