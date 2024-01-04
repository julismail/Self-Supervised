# Malware Classification method with Self-supervised learning

This is the repository of our paper **"MalSSL-Malware classification system with self-supervised learning and Image Representation"**
We developed a malware detection system with contrastive learning and Convolutional Neural Network Resnet18 architecture. 

We trained the model with unlabeled CIFAR10 dataset as a pretext task. 
Then, we retrained the model to recognize malware image representation in Downstream Task.  
We tested the system on two Downstream tasks: 
(1) Malware family classification with Malimg dataset, and 
(2) Malware benign classification with Debi dataset. 

Malimg Dataset is malware image dataset from Nataraj Paper:
https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

Maldeb Dataset is collected by Debi Amalia Septiyani and Halimul Hakim Khairul
D. A. Septiyani, “Generating Grayscale and RGB Images dataset for windows PE malware using Gist Features extaction method,” Institut Teknologi Bandung, 2022.
