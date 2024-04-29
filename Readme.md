# Malware Classification method with Self-supervised learning

This is the repository of our paper **"MalSSL-Self-Supervised Learning for Accurate and Label-Efficient Malware Classification"**
We developed a malware detection system with contrastive learning and Convolutional Neural Network Resnet18 architecture. 

We trained the model with unlabeled Imagenette dataset as a pretext task. 
Then, we retrained the model to recognize malware image representation in Downstream Task.  

We tested the system on two Downstream tasks: 
1. Malware family classification with Malimg dataset, and  
2. Malware benign classification with Debi dataset. 

**The Maldeb Dataset** is an Image representation of Malware-benign dataset. The Dataset were compiled from various sources malware repositories:  *The Malware-Repo, TheZoo,Malware Bazar, Malware Database, TekDefense*. Meanwhile benign samples were sourced from system application of Microsoft 10 and 11, as well as open source software repository such as *Sourceforge, PortableFreeware, CNET, FileForum*. The samples were validated by scanning them using Virustotal Malware scanning services. The Samples were pre-processed by transforming the malware binary into grayscale images following rules from Nataraj (2011). 

**Malimg Dataset** is malware image dataset from Nataraj Paper:
https://vision.ece.ucsb.edu/research/signal-processing-malware-analysis

build with pytorch
https://pytorch.org/
and lightly
https://docs.lightly.ai/self-supervised-learning/index.html

link to our paper: http://doi.org/10.1109/ACCESS.2024.3392251

