# CFDL
Code for "Completed Feature Disentanglement Learning for Multimodal MRIs Analysis" publised in JBHI. [https://arxiv.org/pdf/2407.04916]

We propose an effective MML framework called CFDL, incorporating a novel Complete Feature Disentangle-
ment (CFD) strategy that separates multimodal information into modality-shared, modality-specific, and modality-partial-shared features, the last of which
has been overlooked in previous FD-based methods. Our analysis and experiments demonstrate the critical role of modality-partial-shared features in prediction. Additionally, we present the (ynamic Mixture-of-Experts Fusion) DMF module, which explicitly and dynamically fuses the decoupled features. The LinG_GN within the DMF module can generate the decoupled feature weights by capturing their local-global relationships.

![image](https://github.com/user-attachments/assets/476949e5-6ba2-4561-a298-686270664ad3)

**Create environment:**

conda env create -f CFDL.yml

**Run code:**

python main.py --dataset_name='men' (change "dataset_name" to train different dataset)

**If you have any questions, please contact liu_dling@tju.edu.cn!**

