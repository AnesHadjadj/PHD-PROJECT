# PHD-PROJECT
# Anes Hadjadj

Short introduction

## 🔬 Research

My doctoral research focuses on **interpretable deep learning for medical classification and diagnosis**, with a particular emphasis on **brain MRI analysis and glioblastoma**.

Deep learning has demonstrated strong potential for medical image analysis, but the complexity of modern models remains a major challenge for clinical adoption. A model may achieve high predictive performance while providing little insight into **why** a particular prediction was made. My research therefore investigates the balance between **predictive performance and interpretability**.

A central application of this research is the **non-invasive prediction of MGMT promoter methylation status in glioblastoma (GBM)** from brain MRI. MGMT methylation is an important molecular biomarker associated with treatment response, but its assessment traditionally relies on invasive tissue sampling followed by molecular analysis.

The objective is to investigate deep learning approaches that can learn predictive patterns directly from MRI while providing **transparent and objectively evaluable explanations** of their decisions.

### 🎯 Research Objectives

* Develop deep learning approaches for **MGMT promoter methylation prediction** from brain MRI.
* Investigate hybrid approaches that combine predictive modeling with **interpretability mechanisms**.
* Compare different **Explainable AI (XAI)** methods for medical imaging.
* Develop objective criteria for evaluating the quality and reliability of generated explanations.
* Investigate whether interpretable models can maintain competitive predictive performance while providing **clinically meaningful insights**.

Ultimately, the research is built around two complementary objectives:

> **Performance + Transparency**

The goal is not only to predict *what* the model identifies, but also to better understand **why it makes a particular prediction**.

## 🧠 Research Interests
* Deep Learning / Machine Learning
* Brain MRI / Neuroimaging
* Explainable / Interpretable AI (XAI)
* Computer Vision
* Radiogenomics
* Medical Diagnosis / Classification
* Hybrid Deep Learning Models

## 📚 Publication

**Deep Learning-Based MGMT Promoter Methylation Prediction: A Comparative Interpretability-Oriented Review**

*2026 International Conference on Cyber Management and Engineering (CyMaEn'26)* — **Accepted**

A comparative review of three deep learning frameworks for MRI-based MGMT promoter methylation prediction, focusing on predictive performance, interpretability, generalization, and clinical relevance.

[📄 Paper](./My%20Documents/MGMT_Review_Conference_Paper.pdf)

## 🔎 Current Research
### XAI-Guided Improvement of KBF for MGMT Prediction

My current doctoral research investigates the use of **Explainable AI (XAI)** to improve Knowledge-Based Filtering (KBF) for **MGMT promoter methylation prediction from brain MRI**.

The work builds upon the approach proposed by Capuozzo et al., using FLAIR and T1-weighted MRI to perform knowledge-based filtering before CNN-based prediction.

The main research direction is to investigate whether the regions selected by KBF correspond to the regions considered important by the model, and to use this information to develop an **XAI-guided improvement of the filtering stage**.

**Current focus:**
- Reproducing and establishing the Capuozzo baseline
- Defining objective criteria for XAI evaluation
- Benchmarking candidate XAI methods
- Comparing KBF-selected regions with model explanations and available ground truth
- Designing an XAI-guided KBF improvement
- Evaluating its impact on prediction performance and explanation quality

> **Research goal:** Improve the relationship between *where the model looks* and *where the relevant pathology is located*, while maintaining competitive predictive performance.

[📖 Detailed research notes](./Current_Research/)
## 📄 Curriculum Vitae
📄 All my documents, including my Resume, publications, and certificates, are available in the [`My Documents`](./My%20Documents/) folder.
