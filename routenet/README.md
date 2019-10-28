# RouteNet

RouteNet is a neural architecture for network performance evaluation first proposed in the paper 

*Unveiling the potential of GNN for network modeling and optimization in SDN* by K. Rusek, J. Suárez-Varela, A. Mestres, P. Barlet-Ros, A. Cabellos-Aparicio accepted for ACM Symposium on SDN Research, April 2019, San Jose, CA, USA. [arXiv:1901.08113](https://arxiv.org/abs/1901.08113). 

An extended version of the model is presented in the paper *RouteNet: Leveraging Graph Neural Networks for network modeling and optimization in SDN*
Krzysztof Rusek, José Suárez-Varela, Paul Almasan, Pere Barlet-Ros, Albert Cabellos-Aparicio [arXiv:1910.01508](https://arxiv.org/abs/1910.01508). 

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
@inproceedings{Rusek:2019:UPG:3314148.3314357,
 author = {Rusek, Krzysztof and Su\'{a}rez-Varela, Jos{\'e} and Mestres, Albert and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
 title = {Unveiling the Potential of Graph Neural Networks for Network Modeling and Optimization in SDN},
 booktitle = {Proceedings of the 2019 ACM Symposium on SDN Research},
 series = {SOSR '19},
 year = {2019},
 isbn = {978-1-4503-6710-3},
 location = {San Jose, CA, USA},
 pages = {140--151},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/3314148.3314357},
 doi = {10.1145/3314148.3314357},
 acmid = {3314357},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Graph Neural Networks, SDN, network modeling, network optimization},
} 

```

## Dataset
Datasets used for training are available at [KDN website](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v1)
For training simulation, data must be converted to TFrecords using `upcdataset.py` script. 