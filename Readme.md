# Malware Classification method with Self-supervised learning

This is the repository of our paper **"MalSSL-Self-Supervised Learning for Accurate and Label-Efficient Malware Classification"**
We developed a malware detection system with contrastive learning and Convolutional Neural Network Resnet18 architecture. 

We trained the model with unlabeled Imagenette dataset as a pretext task. 
Then, we retrained the model to recognize malware image representation in Downstream Task.  
We tested the system on two Downstream tasks: 
(1) Malware family classification with Malimg dataset, and 
(2) Malware benign classification with Debi dataset. 

We tested also the system with GAN-generated samples

Malimg Dataset is malware image dataset from Nataraj Paper:
https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

Maldeb Dataset is collected by Debi Amalia Septiyani and Halimul Hakim Khairul
D. A. Septiyani, “Generating Grayscale and RGB Images dataset for windows PE malware using Gist Features extaction method,” Institut Teknologi Bandung, 2022.

GAN-generated samples are created with DCGAN algorithm from Maldeb dataset
DCGAN - Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
