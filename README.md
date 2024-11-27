# Graph-Enhanced Multi-Scale Contrastive Learning for Graph Anomaly Detection with Adaptive Diffusion Models


This is the PyTorch source code for the GCLAD. 
The code runs on Python 3. 
Install the dependencies and prepare the datasets with the following commands:



## Dataset

The five public datasets used in the paper are shown below.

The datasets can be downloaded from the following:

### Cora Dataset
Cora[29]: https://linqs.org/datasets/

### Citeseer Dataset
Citeseer[29]: https://linqs.org/datasets/

### Pubmed Dataset
Pubmed[29]: https://linqs.org/datasets/

### Flickr Dataset
Flickr[30]: https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection/tree/main/Datasets

### Blogcatalog Dataset
Blogcatalog[30]: https://github.com/XiaoxiaoMa-MQ/Awesome-Deep-Graph-Anomaly-Detection/tree/main/Datasets

We added the links in Section IV. A. “Experimental Settings” (Page 7).

[29] Collective classification in network data, AI magazine 2008.
[30] Relational learning via latent social dimensions, KDD 2009.




### Requirements

The proposed GCLAD is implemented with python 3.7 on a NVIDIA 3070 GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.10.2
- dgl==0.4.1
- numpy==1.19.2

### Quick Start

python execute.py
