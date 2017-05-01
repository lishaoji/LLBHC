# STA663_FinalProject
STA 663 SP17 Final Project by Shaoji Li &amp; Xichu Liu


Bayesian Hierarchical Clustering Implementation in Python


=========================================================

In this project, we tend to implement the Bayesian Hierarchical Clustering algorithm, presented by Dr. Katherine Heller and Dr. Zoubin Ghahramani back in 2005. The algorithm, similar to any agglomerative clustering with distance metrics, is a one-pass, bottom-up method initializing each point in its own cluster and then iteratively merges pairs of cluster by measuring the marginal likelihood and determining whether to proceed. In Python, we first break down the algorithm into multiple parts, then implement each part individually, and eventually resemble the whole process. Using both synthetic and real-world data, we test our implementation's correctness and efficiency, where optimization methods are applied. The goal is to efficiently implement the algorithm in its original setting, where Dirichet Process is its generative model and the dataset follows a multinomial distribution. At the end, we discuss the applications of our implementation, as well as the Bayesian Hierarchical Clustering algorithm in general.

==========================================================


Instruction to load packge:

1. Download the package in zip
2. Unzip the file
3. Open terminal and change to the directory of the file
4. Input "pip install ."

Whenever the package is needed, input "import LLBHC"
