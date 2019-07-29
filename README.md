*This is my submission to the [MICCAI Educational Challenge 2019](https://miccai-sb.github.io/challenge.html).*

*You can run the notebook on [Google Colab](https://colab.research.google.com/github/fepegar/miccai-educational-challenge-2019/blob/master/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb) or render an already executed version on [nbviewer](https://nbviewer.jupyter.org/github/fepegar/miccai-educational-challenge-2019/blob/master/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb).*

<a href="https://colab.research.google.com/drive/1vqDojKuC4Svb97LdoEyZQygm3jccX4hr" 
   target="_parent">
   <img align="left" 
      src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

<a href="https://nbviewer.jupyter.org/github/fepegar/miccai-educational-challenge-2019/blob/master/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb" 
   target="_parent">
   <img align="right" 
      src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.png" 
      width="109" height="20">
</a>

<br />

---

# Combining the power of PyTorch and NiftyNet

[NiftyNet](https://niftynet.io/) is "an open source convolutional neural networks platform for medical image analysis and image-guided therapy" built on top of [TensorFlow](https://www.tensorflow.org/). It is probably the easiest way to get started with deep learning for medical image.

[PyTorch](https://pytorch.org/) is "an open source deep learning platform that provides a seamless path from research prototyping to production deployment". It's low-level enough to offer a lot of control over what's going on under the hood during training, and its [dynamic computational graph](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1) allows for very easy debugging. Being a generic deep learning framework, it's not tailored to the needs of the medical image field, although its popularity in this field is increasing rapidly.

One can [extend a NiftyNet application](https://niftynet.readthedocs.io/en/dev/extending_app.html), but it's not straightforward without being quite familiar with the framework and fluent in TensorFlow 1.X. Therefore it is convenient to implement applications in PyTorch using NiftyNet models and functionalities. This can be specially used for further experimenting with the model or for transfer learning applications.

So why not use [both](https://www.youtube.com/watch?v=vqgSO8_cRio&feature=youtu.be&t=5)? This tutorial shows how to port the parameters of a model trained on NiftyNet to a PyTorch model and test the PyTorch model while using NiftyNet's I/O modules, which specialize in medical image processing.
