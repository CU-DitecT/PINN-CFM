# PINN_CFM

## Description
Implementation for "A physics-informed deep learning paradigm for car-following models" [[TRC link]](https://www.sciencedirect.com/science/article/pii/S0968090X21002539).

While our original code was based on Tensorflow V1, we rewrote the core code using PyTorch for easier implementation (much easier). Please contact us if you should encounter any questions using this version of code. We also include the original Tensorflow V1 code in the 'tf1' folder.

If you use our work then please cite
```
@article{mo2021physics,
  title={A physics-informed deep learning paradigm for car-following models},
  author={Mo, Zhaobin and Shi, Rongye and Di, Xuan},
  journal={Transportation research part C: emerging technologies},
  volume={130},
  pages={103240},
  year={2021},
  publisher={Elsevier}
}
```






# Training
Run the following command in the terminal
```python
python main.py
```

# Contact

For any question, please contact zm2302@columbia.edu

# Other Papers
Our other PINN papers:
1. Trafficflowgan: Physics-informed flow based generative adversarial network for uncertainty quantification, *Zhaobin Mo, Yongjie Fu, Daran Xu, Xuan Di*, ECML, 2023. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-26409-2_20)][[code](https://github.com/ZhaobinMo/TrafficFlowGAN)]
2. Quantifying Uncertainty In Traffic State Estimation Using Generative Adversarial Networks, *Zhaobin Mo, Yongjie Fu, Xuan Di*, ISTC, 2022. [[paper](https://ieeexplore.ieee.org/document/9921791)]
3. Uncertainty quantification of car-following behaviors: Physics-informed generative adversarial networks, *Zhaobin Mo, Xuan Di*, KDD UrbComp, 2022. [[paper](http://urban-computing.com/urbcomp2022/file/UrbComp2022_paper_3574.pdf)]
4. Physics-informed deep learning for traffic state estimation: A hybrid paradigm informed by second-order traffic models, *Rongye Shi, Zhaobin Mo, Xuan Di*, AAAI, 2021. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16132)]
5. A physics-informed deep learning paradigm for traffic state and fundamental diagram estimation, *Rongye Shi, Zhaobin Mo, Kuang Huang, Xuan Di, Qiang Du*, IEEE T-ITS, 2021. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9531557)]

