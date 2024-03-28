# Malware Classification method with Self-supervised learning

This is the repository of our paper **"MalSSL-Self-Supervised Learning for Accurate and Label-Efficient Malware Classification"**
We developed a malware detection system with contrastive learning and Convolutional Neural Network Resnet18 architecture. 

We trained the model with unlabeled Imagenette dataset as a pretext task. 
Then, we retrained the model to recognize malware image representation in Downstream Task.  
We tested the system on two Downstream tasks: 
(1) Malware family classification with Malimg dataset, and 
(2) Malware benign classification with Debi dataset. 

The Dataset were collected from several malware repositories, including TekDefense, TheZoo, The Malware-Repo, Malware Database amd Malware Bazar. The benign samples were collected from Microsoft 10 and 11 system apps and several open source software repository including CNET, Sourceforge, FileForum, PortableFreeware.
Malimg Dataset is malware image dataset from Nataraj Paper:
https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

Maldeb Dataset is collected by Debi Amalia Septiyani and Halimul Hakim Khairul
D. A. Septiyani, “Generating Grayscale and RGB Images dataset for windows PE malware using Gist Features extaction method,” Institut Teknologi Bandung, 2022.
Additional benign samples wer collected by Dani Agung Prastiyo, "Design and implementation of a machine learning-based malware classification system with an audio signal feature Analysis Approach," Institut Teknologi Bandung, 2023
