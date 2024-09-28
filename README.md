__Learning-Short-Codes-for-Fading-Channels-with-No-or-Receiver-Only-Channel-State-Information__

- Authors:
_Rishabh Sharad Pomaje_ and _Rajshekhar V. Bhat_

- This repository contains the code base for the simulations and neural networks used to produce the results in the above research.
- Here's the link to arXiv pre-print: [arXiv/2409.08581](https://arxiv.org/abs/2409.08581) 

- ${\color{cyan}{Motivation}}$: Neural Networks have been shown to be _Universal Function Approximators_. The question we try and approach is can explore new solution spaces in the Wireless Communication Systems which can be represented as _autoencoders_, a type of Neural Network, very well. Particularly, we focus on systems involving no channel estimation and thus those that do not use Channel State Information (CSI) at any stage. This can be particularly helpful if the hardware and other constraints are quite stringent as CSI estimation and equalization can involve considerable overhead in the system. 

- If you directly want to head into the code, there are two main experiments 
    1. SISO with No CSI: In this, no assumption of CSI has been made either at the transmitter or at the receiver. We learn an orthogonal signalling scheme. This is confirmed by the detailed breakdown analysis of the learned/ trained neural network.
    We expand this neural network to other code rates and also explore other channel distributions to demonstrate the generality of the framework/ paradigm.

    2.  SISO with CSIR: This experiment mainly demonstrates the inefficiency of:
        a. Using AWGN codes for Fading channels.
        b. Partial Augmentation of ML in conventional communication systems. (or rather intervened autoencoders.)
        c. Gains in performance are possible and thus, techinques such as non-linear combining/ equalization of channel should be considered. 
