

# 🧠 **Loan Bias Detection Using Machine Learning & Fairness Metrics**

A complete machine learning project designed to **detect, measure, and reduce bias** in loan approval decisions.
This project identifies whether **protected groups (e.g., gender)** are treated unfairly and applies **fairness mitigation techniques** to correct discrimination.

---

## 📌 **Project Overview**

Loan datasets often contain **hidden bias**, leading to unequal treatment of applicants.
For example:

* One gender may receive loans more often
* One group may face higher rejection rates
* Models trained on biased data may continue the discrimination

This project provides an end-to-end solution to **detect**, **measure**, and **mitigate** such bias using:

* **Machine Learning (Logistic Regression)**
* **AIF360 Fairness Toolkit**
* **Fairness Metrics** (Disparate Impact, Equal Opportunity, Demographic Parity)
* **Reweighing Technique** (to reduce bias)

All code is **Google Colab–friendly** and easily reproducible.

---

## 🎯 **Objectives**

✔ Detect whether loan approval decisions are biased
✔ Measure fairness using statistical fairness metrics
✔ Train a baseline ML model and analyze its fairness
✔ Apply bias mitigation techniques
✔ Compare model performance before and after fairness correction
✔ Provide a clean, interpretable conclusion

---

## ⚙️ **Technologies Used**

| Category             | Tools / Libraries   |
| -------------------- | ------------------- |
| **Language**         | Python              |
| **ML**               | Scikit-Learn        |
| **Fairness Toolkit** | IBM AIF360          |
| **Visualization**    | Matplotlib, Seaborn |
| **Data Processing**  | Pandas, NumPy       |
| **Environment**      | Google Colab        |

---

## 📂 **Project Structure**

```
📦 Loan-Bias-Detection
 ┣ 📄 loan.csv
 ┣ 📄 loan_bias_analysis.ipynb
 ┣ 📄 README.md
 ┗ 📄 requirements.txt
```

---

# 🚀 **How to Run the Project (Google Colab)**

### **1️⃣ Upload the dataset**

Upload `loan.csv` to Colab or mount Google Drive.

```python
import pandas as pd
df = pd.read_csv("loan.csv")
```

---

### **2️⃣ Preprocess the data**

* Handle missing values
* Encode categorical columns
* Scale numeric features

---

### **3️⃣ Define sensitive attribute**

Example:

```python
sensitive = "Gender"
label = "Loan_Status"
```

---

### **4️⃣ Train logistic regression model**

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### **5️⃣ Convert dataset to AIF360 format**

Handles fairness metrics and protected attributes.

---

### **6️⃣ Calculate fairness metrics**

* **Disparate Impact**
* **Demographic Parity**
* **Equal Opportunity Difference**
* **Average Odds Difference**

These metrics show **how differently each gender is treated**.

---

### **7️⃣ Apply Bias Mitigation (Reweighing)**

IBM’s AIF360 Reweighing balances sample weights to remove discrimination.

```python
RW = Reweighing(privileged_groups=[{sensitive:1}],
                unprivileged_groups=[{sensitive:0}])

train_transformed = RW.fit_transform(train_bds)
```

---

### **8️⃣ Re-train fair model & compare results**

You will see:

* Improved fairness
* Reduced bias
* Minimal accuracy loss (ideally)

---

### **9️⃣ Loan Approval % by Gender**

The project calculates:

* Actual approval rates (Male vs Female)
* Bias gap in approval
* Whether bias is statistically significant

---

# 📝 **Why Was This Project Needed?**

Loan decision-making systems directly impact people’s lives.
If the data used to train models contains bias, AI can become discriminatory.

This project helps solve these practical issues:

### ✔ Identify unfair treatment in loan approvals

### ✔ Ensure ML models do not continue discrimination

### ✔ Provide transparency in AI decision-making

### ✔ Help financial institutions maintain ethical & legal compliance

---

# 🔧 **How This Project Solves the Problem**

### ✅ Detects Bias

Uses statistical and ML-based fairness metrics
(e.g., Disparate Impact, Equal Opportunity).

### ✅ Explains Bias

Shows which gender gets higher approval and why.

### ✅ Fixes Bias

Applies **Reweighing** to correct dataset imbalance.

### ✅ Validates Improvement

Compares fairness metrics **before and after** mitigation.

### Result:

A **fairer**, **more ethical**, and **more reliable** loan approval system.

---

# 📊 **Final Output & Interpretation**

The notebook automatically prints:

* Loan approval percentage for each gender
* Bias gap (in %)
* Whether the dataset is biased
* Fairness metrics before and after correction
* Final, readable conclusion

Example:

```
⚠️ Approval gap = 18.5% → High bias
⚠️ Disparate Impact = 0.67 → Discrimination detected
✓ After Reweighing, Disparate Impact = 0.95 → Bias reduced
```

---

# 🤝 **Contributions**

Feel free to open issues, suggest improvements, or contribute new fairness techniques.

---

# 📜 **License**

This project is released under the **MIT License**.

---

# 🙌 **Acknowledgments**

Thanks to **IBM AIF360** and the open-source community for fairness tools.

---


