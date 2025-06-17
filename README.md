# Ensemble-Based Face Authentication for Secure POS Transactions: ArcFace, One-Class SVM, and Neural Networks

**Author:** Vimal Ramnarain
**Institution:** Syracuse University, College of Engineering and Computer Science
**Location:** Syracuse, US
**Email:** [vimalramnarain@gmail.com](mailto:vimalramnarain@gmail.com)

---

## Abstract

This research paper explores the potential improvements that are made to the prior research *Exploring the Effectiveness of Face Authentication in Payment Systems Across POS Devices and Mobile Platforms* by leveraging advanced machine learning techniques. Building on earlier work using DeepFace with FaceNet, this study integrates **ArcFace** for highly discriminative face embeddings, **One-Class Classification (OCC)** for anomaly detection, and a complementary **Neural Network classifier** trained with class-weighted supervision. A key contribution of this work is the development of an **ensemble framework** that combines One-Class SVM and Neural Network predictions to improve robustness, reduce false rejections, and strengthen spoof resistance. Furthermore, the system incorporates a comprehensive **data augmentation pipeline** to simulate real-world challenges such as blur, occlusion, noise, lighting variation, and spoof attacks (e.g., print/photo attacks).

This research aims to enhance the accuracy, security, and robustness of the prior face recognition system, particularly in identifying authorized individuals and detecting spoof attempts. This study delves into using **metric learning, data augmentation, and anomaly detection techniques** to address challenges in the prior facial recognition systems, with particular emphasis on overcoming issues such as class imbalance, spoof attacks, and varying image quality.

---

## I. Introduction

Face authentication is a crucial biometric security measure in digital transactions, with great potential to be a less intrusive and a quick way to securely verify payment in POS or mobile payment systems (when combined with PINs, passwords, or other biometric systems). Previous research developed a facial recognition system using the DeepFace library with FaceNet, evaluated initially on the LFW dataset and later the CelebA dataset. While the system demonstrates high accuracy and AUC scores in distinguishing between authorized users and impostors, it suffers from high **False Rejection Rates (FRR)**, vulnerability to spoofing, and class imbalance issues.

To address these issues, the current research introduces a redesigned face authentication pipeline focused on enhancing robustness, security, and real-world applicability. At the core of the system is **ArcFace**, a face recognition model offering better intra-class compactness and inter-class separation than FaceNet. Building on this embedding foundation, the proposed method leverages a dual-path classification strategy:

* **One-Class SVM**, trained only on authorized users, serves as an anomaly detector for identifying impostors or spoof attempts in the absence of negative training samples.
* **Supervised Neural Network**, trained on labeled authorized vs. impostor embeddings with class-weighting, captures subtle classification boundaries that OCC might miss.

These two paths are then combined in an **ensemble framework**, where predictions are fused to balance security (minimizing FAR) and usability (reducing FRR). To further improve generalization and robustness under real-world conditions, a data augmentation pipeline introduces transformations such as blur, occlusion, noise, low lighting, and spoof simulations.

This ensemble-based approach significantly outperforms the prior system, particularly in reducing the previously high FRR and improving resistance to spoof attacks. The resulting model demonstrates near-perfect classification performance under ideal conditions and maintains strong security and usability across challenging scenarios—making it a compelling solution for real-world biometric authentication systems.

---

## II. Previous Related Work

Recent advancements in face recognition have significantly improved the reliability of biometric authentication, largely due to the adoption of deep learning models like FaceNet and ArcFace. In earlier iterations, FaceNet demonstrated strong classification performance, particularly in terms of AUC. However, the system encountered limitations in realistic payment scenarios: high FRR, poor generalization under challenging visual conditions, and vulnerability to spoofing attacks.

**ArcFace** ([1](#references)) utilizes an additive angular margin loss to produce highly discriminative embeddings, offering superior face verification performance over FaceNet. Additional studies explored enhancements to DeepFace using Canonical Correlation Analysis ([2](#references)), but ArcFace demonstrated greater generalizability across challenging datasets.

Anomaly detection techniques such as **One-Class SVMs** ([3](#references)) are well-suited for highly imbalanced POS datasets. Furthermore, Presentation Attack Detection (PAD) remains a critical area, as highlighted by Sharma and Sharma ([4](#references)), with complementary liveness detection approaches using Siamese Networks also showing promise ([5](#references)).

Finally, **data augmentation** remains a proven strategy to improve generalization. Techniques such as blur, occlusion, noise, and lighting variation help simulate real-world deployment conditions ([6](#references)).

---

## III. Experiment Design

The proposed system integrates the following key components:

1. **ArcFace Model**: Pre-trained on large-scale datasets for highly discriminative face embeddings.
2. **One-Class Classification (OCC)**: Using One-Class SVM trained only on authorized embeddings to detect anomalies or spoof attempts.
3. **Data Augmentation**: Including occlusion, noise, blur, low light, and synthetic spoof attacks to simulate challenging conditions.

The **OCC model** addresses class imbalance by learning only the authorized class. Robust augmentation techniques informed by literature ([4](#references), [5](#references)) simulate real-world noise and adversarial conditions.

The system is evaluated on the **CelebA dataset** using metrics such as FAR, FRR, and AUC. Challenges include tuning augmentation and managing high-dimensional SVMs, addressed through empirical refinement.

Future privacy-preserving approaches are also acknowledged ([7](#references), [8](#references)).

---

## IV. Performance Evaluation

The system demonstrates strong impostor rejection, with:

* **False Acceptance Rate (FAR):** 0.00%
* **Accuracy:** 99.41%
* **False Rejection Rate (FRR):** 40.00% (indicating need for improvement)

The **ArcFace + One-Class SVM pipeline** successfully models authorized identities but struggles with intra-class variation and aggressive augmentation. Enhancements include:

* Refining augmentation for better balance.
* Improving face detection reliability.
* Adding ensemble evaluation (One-Class SVM + Neural Network).

**Technical improvements:**

* **TensorFlow GPU optimization** with memory growth control ([9](#references)).
* **imgaug library** for diverse data augmentation ([10](#references)).
* **Class-balanced sampling** and **stratified splits** ([11](#references)).
* **ArcFace (buffalo\_l variant)** via InsightFace ([12](#references), [13](#references)).
* **Neural Network architecture** with batch normalization ([14](#references)), dropout ([15](#references)), and class weighting ([16](#references)).
* **Complementary One-Class SVM** anomaly detection ([17](#references)).

**Evaluation metrics:**

* Accuracy
* ROC-AUC ([18](#references))
* Confusion matrices
* FAR & FRR

**Results Summary (5 Runs):**

* Best: 100% Ensemble Accuracy, AUC 1.000, FAR 0%, FRR 0%.
* Worst: 93.13% Accuracy, AUC 0.988, FAR 6.92%, FRR 0%.
* Average: \~97.7% Ensemble Accuracy, AUC \~0.998, FAR \~1.14%, FRR 0%.

---

## V. Future Work

Key avenues for improvement include:

* **Scalability:** Replace One-Class SVM with **Autoencoders** or **Deep SVDD** ([19](#references), [20](#references)) for large datasets.
* **Multi-user support:** Transition from one-class to multi-class or hybrid architectures.
* **Advanced augmentation:** Use **GANs** for realistic variations.
* **Liveness detection:** Integrate motion or depth-based presentation attack detection.
* **Domain-specific fine-tuning:** Apply contrastive learning or fine-tune ArcFace on POS datasets.

---

## References

1. Deng et al., ["ArcFace: Additive Angular Margin Loss for Deep Face Recognition"](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html), CVPR 2019.
2. Ahmad & Ali, ["Improving DeepFace CNN Performance with Canonical Correlation Analysis"](https://dl.acm.org/doi/10.1145/3633598.3633602), ICSCC 2023.
3. Ruff et al., ["Deep One-Class Classification"](https://proceedings.mlr.press/v80/ruff18a.html), ICML 2018.
4. Sharma & Sharma, ["Presentation Attack Detection: A Systematic Literature Review"](https://dl.acm.org/doi/10.1145/3687264), ACM Computing Surveys 2024.
5. Deng et al., ["Face Liveness Detection Based on Client Identity Using Siamese Network"](https://arxiv.org/abs/1903.05369), arXiv 2019.
6. Zhang et al., ["Full-Convolution Siamese Network Algorithm Under Deep Learning"](https://link.springer.com/article/10.1007/s11227-022-04439-x), J. Supercomput. 2023.
7. Sharma et al., ["Toward a Privacy-Preserving Face Recognition System"](https://dl.acm.org/doi/10.1145/3673224), ACM Computing Surveys 2024.
8. Yang & Zhang, ["PRO-Face: Privacy-Preserving Obfuscation"](https://dl.acm.org/doi/10.1145/3503161.354820), CCS 2022.
9. Abadi et al., ["TensorFlow: A System for Large-Scale Machine Learning"](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf), OSDI 2016.
10. Jung, ["imgaug Library"](https://github.com/aleju/imgaug), 2020.
11. Pedregosa et al., ["Scikit-learn"](https://jmlr.org/papers/v12/pedregosa11a.html), JMLR 2011.
12. InsightFace Contributors, ["InsightFace"](https://github.com/deepinsight/insightface), 2021.
13. Deng et al., ["ArcFace (Original Paper)"](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html), CVPR 2019.
14. Ioffe & Szegedy, ["Batch Normalization"](https://proceedings.mlr.press/v37/ioffe15.html), ICML 2015.
15. Srivastava et al., ["Dropout"](http://jmlr.org/papers/v15/srivastava14a.html), JMLR 2014.
16. Goodfellow et al., *Deep Learning*, MIT Press, 2016.
17. Schölkopf et al., ["One-Class SVM"](https://www.jmlr.org/papers/volume13/schölkopf02a/schölkopf02a.pdf), Neural Computation 2001.
18. Fawcett, ["ROC Analysis"](https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X), Pattern Recognit. Lett. 2006.
19. Schölkopf et al., *Deep SVDD*, Neural Computation 2001.
20. Ruff et al., *Deep One-Class Classification*, ICML 2018.

---

