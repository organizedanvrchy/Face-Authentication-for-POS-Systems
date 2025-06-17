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

**ArcFace** [[1](#references)] utilizes an additive angular margin loss to produce highly discriminative embeddings, offering superior face verification performance over FaceNet. Additional studies explored enhancements to DeepFace using Canonical Correlation Analysis [[2](#references)], but ArcFace demonstrated greater generalizability across challenging datasets.

Anomaly detection techniques such as **One-Class SVMs** [[3](#references)] are well-suited for highly imbalanced POS datasets. Furthermore, Presentation Attack Detection (PAD) remains a critical area, as highlighted by Sharma and Sharma [[4](#references)], with complementary liveness detection approaches using Siamese Networks also showing promise [[5](#references)].

Finally, **data augmentation** remains a proven strategy to improve generalization. Techniques such as blur, occlusion, noise, and lighting variation help simulate real-world deployment conditions [[6](#references)].

---

## III. Experiment Design

This research proposes the development of a more robust and secure face authentication system for POS and Mobile Payments by building upon the limitations observed in prior research. The proposed system integrates/replaces the following components into the prior system:
  1. ArcFace Model: For face embedding extraction, which provides highly discriminative features for face recognition. The model is pre-trained on large-scale datasets, making it well-suited for accurate identity matching.
  2. One-Class Classification (OCC): To detect anomalies or spoof attempts, One-Class SVM will be employed to model the distribution of authorized users' embeddings. The OCC model will classify unknown faces as either normal (authorized) or abnormal (impostor/spoofed).
  3. Data Augmentation: To improve the system's resilience to varied input conditions. Techniques such as occlusion, random noise, blur, low light, and synthetic spoof attacks (e.g., printed photos) will be included in the training dataset.

OCC is implemented for spoof and impostor detection, which addresses the class imbalance inherent in POS systems—where only one authorized user is enrolled per device. Here, a One-Class SVM is implemented to model only the authorized class. By training on ArcFace embeddings from the legitimate user, the classifier treats unknown faces or attacks as anomalies, significantly improving resistance to spoofing [[3](#references)], [[1](#references)].

Robust data augmentation is used for realistic training, along with application of a wide range of augmentation techniques including blur, occlusion, noise, lighting changes, and synthetic spoof attacks (e.g., print/photo spoofing). These transformations are informed by best practices from literature [[4](#references)], [[5](#references)] and aim to simulate real-world noise and adversarial conditions, improving the system’s generalization and resilience.

The enhanced system is evaluated on the same CelebA dataset, with prior performance metrics such as False Acceptance Rate (FAR), False Rejection Rate (FRR), and AUC to assess security and usability improvements–if any. A key objective is to reduce the FRR of authorized users, which was a significant weakness in the previous system—without compromising spoof resistance.

In terms of potential roadblocks, challenges might arise in finetuning the data augmentation pipeline for optimal performance or dealing with the complexity of the One-Class SVM model in highdimensional spaces. These challenges will most likely be addressed through empirical testing and iterative refinement of preprocessing, model parameters, and evaluation metrics.

Finally, while not the core focus of this paper, privacypreserving techniques in facial recognition have received increasing attention [[7](#references)], [[8](#references)]. These works underline future research avenues for creating user-controllable or obfuscated facial biometric systems that maintain utility while minimizing privacy risks.

---

## IV. Performance Evaluation

The current implementation of the face authentication system demonstrates a strong ability to reject impostor attempts, as evidenced by the 0.00% False Acceptance Rate (FAR) and near-perfect accuracy of 99.41% (shown in Figure 1 below). This level of security against unauthorized access is critical in high-risk applications such as point-of-sale (POS) systems or mobile payment platforms.

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Old_Result/Iter1_Run1_CF.png?raw=true" width=575/>
</p>

**Figure 1.** Resultant Confusion Matrix of 1st Iteration of Face Authentication System with proposed enhancements

The current system achieves this robustness using ArcFace embeddings combined with a One-Class SVM trained solely on a single authorized identity (ID 2880). The preprocessing pipeline applies several augmentation techniques to improve robustness under varying conditions, including blur, occlusion, noise, and lighting changes (shown in Figure 2 below).

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Old_Result/Iter1_Run1_AUG.png?raw=true" width=800/>
</p>

**Figure 2.** Example of Image Augmentation during Preprocessing

However, a major shortcoming is reflected in the False Rejection Rate (FRR), at 40.00%. This indicates that nearly half of the supposed “authorized” access attempts are mistakenly rejected. A high FRR can degrade the user experience and lead to frustration or system abandonment, and thus, must be handled. The root causes are multifold: the use of only one authorized user for training restricts the SVM's ability to generalize to intra-class variations; the data augmentation, while beneficial for robustness, is applied too aggressively, potentially distorting the feature space; and fallback embeddings of all-zero vectors during face detection failures can compromise both the training and evaluation phases.

To address these limitations, the system is enhanced through a series of targeted improvements aimed at increasing generalization and reducing false rejections. These enhancements include refining the augmentation pipeline to balance robustness with feature integrity, improving face detection reliability to eliminate fallback embeddings, and incorporating ensemble evaluation (One-Class Support Vector Machine and Neural Network) across multiple runs to better capture performance consistency. The results from these improvements demonstrate a marked shift in the model’s ability to maintain high security while substantially improving user accessibility and classification performance, as detailed in the following evaluation.

The updated face authentication pipeline incorporates several enhancements that contribute to improved robustness, accuracy, and computational efficiency compared to prior implementations. First, the integration of TensorFlow GPU configuration (moving away from the previous slow CPU configuration) with memory growth control ensures optimized hardware utilization by preventing memory pre-allocation bottlenecks. This dynamic GPU memory management enables the training process to scale efficiently with available resources, thus improving throughput during neural network optimization \[[9](#references)].

The data augmentation pipeline, using the imgaug library \[[10](#references)], introduces a diverse set of realistic perturbations including additive Gaussian noise, Gaussian and motion blur, intensity scaling, and coarse dropout. These augmentations simulate real-world distortions such as sensor noise, motion artifacts, lighting variations, and occlusions. By exposing the model to such synthetic variations during training, the system enhances its generalization capability and robustness against adversarial conditions or spoofing attacks, which is critical in biometric authentication domains.

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Run1/Image_1.png?raw=true"/>
</p>

**Figure 3.** Improved Image Augmentation using imgaug during Preprocessing of Run #1 with the updated model.

From a dataset perspective, the pipeline employs class-balanced sampling strategies by enforcing minimum samples per authorized identity and random subsampling of impostor classes. This controls class imbalance inherent in face authentication datasets, somewhat balancing bias towards the dominant impostor class and improving the classifier's sensitivity and specificity. The stratified train-test split further preserves label distributions, thereby ensuring a reliable estimation of the model's true performance on unseen data \[[11](#references)].

The feature extraction stage leverages the ArcFace model (buffalo\_l variant) via the InsightFace framework \[[12](#references)], known for its discriminative 512-dimensional embeddings that map facial images to a hypersphere manifold with angular margin constraints. This approach significantly improves inter-class separability and intra-class compactness compared to traditional embeddings \[[13](#references)], allowing for more accurate downstream classification. Notably, the implementation includes rigorous checks for embedding validity, substituting zero vectors for failures to maintain consistent input dimensionality without introducing training bias.

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Run1/Figure_1.png?raw=true"/>
</p>

**Figure 4.** Training and Validation Accuracy achieved during Run #1 with the updated model.

The classifier architecture is now a deep neural network with multiple dense layers, batch normalization \[[14](#references)], and dropout regularization \[[15](#references)]. Batch normalization accelerates training convergence and mitigates internal covariate shift, while dropout reduces overfitting by randomly deactivating neurons during training. The use of a sigmoid output layer with binary cross-entropy loss optimizes the model for the two-class (authorized vs impostor) problem, and class-weighting compensates for label imbalance by emphasizing the minority authorized class \[[16](#references)].

Furthermore, the pipeline integrates a complementary One-Class Support Vector Machine (OC-SVM) \[[17](#references)] trained exclusively on authorized user embeddings to detect anomalous samples. This anomaly detection technique enhances spoof resistance by modeling the genuine user distribution without requiring negative examples, thus providing a principled way to reject impostors outside the learned feature manifold.

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Run1/Figure_2.png?raw=true"/>
</p>

**Figure 5.** Receiver Operator Characteristic graph achieved during Run #1 with the updated model.

For evaluation, the system computes standard metrics including accuracy, Receiver Operator Characteristic (ROC-AUC) \[[18](#references)], confusion matrices, and class-specific FAR and FRR. These metrics offer comprehensive insight into the trade-offs between security (minimizing FAR) and usability (minimizing FRR). Visualizations of training progress (shown in Figure 4), ROC curves (shown in Figure 5 above), and confusion matrices (shown in Figure 6 below) support qualitative assessment and facilitate debugging and fine-tuning.

<p align="center">
<img src="https://github.com/organizedanvrchy/Face-Authentication-for-POS-Systems/blob/main/Results/Run1/Figure_3.png?raw=true"/>
</p>

**Figure 6.** Resultant Confusion Matrix achieved during Run #1 with the updated model.

The updated face authentication model demonstrates significant improvements in both training stability and classification accuracy across multiple runs. Notably, across 5 runs, the best-performing epoch history shows the validation accuracy reaching as high as 99.74%, with ensemble accuracy peaking at 100%, an AUC score of 1.000, and both FAR and FRR effectively at 0%. This indicates nearly perfect recognition and spoof resistance under ideal conditions. Conversely, the worst observed run still achieves a respectable ensemble accuracy of 93.13% with an AUC of 0.988, although it exhibits a higher FAR of 6.92%, which signals some vulnerability to false acceptances in more challenging scenarios. On average, across the five runs, the model attains approximately 97.7% ensemble accuracy with AUC scores consistently near 0.998, FAR averaging around 1.14%, and FRR remaining at 0%. These metrics collectively highlight the robustness and generalization capability of the improved system, balancing low false acceptance with zero false rejections, and reflect the effectiveness of the enhancements such as data augmentation, ArcFace embeddings, and one-class anomaly detection incorporated in the pipeline.

This newer pipeline achieves improved performance through hardware-aware training optimization, realistic data augmentation, robust feature extraction with state-of-the-art embeddings, enhanced classifier architecture with regularization and class balancing, and complementary use of anomaly detection. These improvements (from the previous model) collectively strengthen this model's resilience and accuracy in face authentication tasks, positioning it as a viable solution for real-world biometric security applications.

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

