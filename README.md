# Completed Feature Disentanglement Learning for Multimodal MRIs Analysis
Code for our paper "Completed Feature Disentanglement Learning for Multimodal MRIs Analysis" published in JBHI 2025. [https://arxiv.org/pdf/2407.04916]

We propose an effective MML framework called CFDL, incorporating a novel Complete Feature Disentanglement (CFD) strategy that separates multimodal information into modality-shared, modality-specific, and modality-partial-shared features, the last of which has been overlooked in previous FD-based methods. Our analysis and experiments demonstrate the critical role of modality-partial-shared features in prediction. Additionally, we present the Dynamic Mixture-of-Experts Fusion (DMF) module, which explicitly and dynamically fuses the decoupled features. The LinG_GN within the DMF module can generate the decoupled feature weights by capturing their local-global relationships.

![image](https://github.com/user-attachments/assets/476949e5-6ba2-4561-a298-686270664ad3)

**Create environment:**

conda env create -f CFDL.yml

**Run code:**

python main.py --dataset_name='brats' 
(change "dataset_name" to train different dataset, 'men' for MEN dataset,  'mrnet-meniscus' for MRNet dataset and 'brats' for BraTS 2021 dataset)

**If you have any questions, please contact liu_dling@tju.edu.cn!**

