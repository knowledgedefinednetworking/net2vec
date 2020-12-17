# RouteNet

RouteNet is a neural architecture for network performance evaluation first proposed in the paper 

*Unveiling the potential of GNN for network modeling and optimization in SDN* by K. Rusek, J. Suárez-Varela, A. Mestres, P. Barlet-Ros, A. Cabellos-Aparicio accepted for ACM Symposium on SDN Research, April 2019, San Jose, CA, USA. [arXiv:1901.08113](https://arxiv.org/abs/1901.08113). 

An extended version of the model is presented in the paper *RouteNet: Leveraging Graph Neural Networks for network modeling and optimization in SDN*
Krzysztof Rusek, José Suárez-Varela, Paul Almasan, Pere Barlet-Ros, Albert Cabellos-Aparicio [arXiv:1910.01508](https://arxiv.org/abs/1910.01508). 

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
@ARTICLE{9109574,
  author={K. {Rusek} and J. {Suárez-Varela} and P. {Almasan} and P. {Barlet-Ros} and A. {Cabellos-Aparicio}},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={RouteNet: Leveraging Graph Neural Networks for Network Modeling and Optimization in SDN}, 
  year={2020},
  volume={38},
  number={10},
  pages={2260-2270},
  doi={10.1109/JSAC.2020.3000405}}

```

## Dataset
Datasets used for training are available at [KDN website](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v1)
For training simulation, data must be converted to TFrecords using `upcdataset.py` script. 
