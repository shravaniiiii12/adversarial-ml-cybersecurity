# Adversarial Machine Learning for Cybersecurity

## 📌 Overview
This project analyzes the vulnerability of deep learning models to adversarial attacks and implements defense mechanisms to improve robustness.

---

## 🚀 Features
- CNN-based image classification using CIFAR-10
- FGSM (Fast Gradient Sign Method) attack implementation
- PGD (Projected Gradient Descent) attack implementation
- Adversarial training defense
- Feature squeezing defense
- Multi-epsilon robustness evaluation
- Accuracy vs epsilon visualization

---

## 🛠️ Tech Stack
- Python
- PyTorch
- Torchvision
- Matplotlib

---

## 📊 Results

| Epsilon | Accuracy (%) |
|--------|-------------|
| 0      | ~42%        |
| 0.05   | ~17%        |
| 0.1    | ~6%         |
| 0.2    | ~6%         |
| 0.3    | ~6%         |

---

## 🔍 Key Insights
- Deep learning models are highly vulnerable to adversarial attacks.
- Strong attacks like PGD can reduce accuracy close to zero.
- Defense techniques such as adversarial training and feature squeezing improve robustness.
- There is a tradeoff between model accuracy and robustness.

---

## 📁 Project Structure
adversarial-ml-cybersecurity/
│── train.py
│── model.py
│── fgsm.py
│── pgd.py
│── feature_squeeze.py
│── requirements.txt
│── README.md


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
```
🎯 Future Improvements
1.Add confusion matrix visualization
2.Implement advanced defenses (defensive distillation)
3.Deploy using Streamlit


👤 Author

Shravani Karambelkar
