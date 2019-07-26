[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fepegar/miccai-educational-challenge-2019/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb)

*This is my submission to the [MICCAI Educational Challenge 2019](https://miccai-sb.github.io/challenge.html). You can run the notebook on [Google Colab](https://colab.research.google.com/github/fepegar/miccai-educational-challenge-2019/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb)*.

---

# Combining the power of PyTorch and NiftyNet

[NiftyNet](https://niftynet.io/) is "an open source convolutional neural networks platform for medical image analysis and image-guided therapy" built on top of [TensorFlow](https://www.tensorflow.org/). It is probably the easiest way to get started with deep learning for medical image.

[PyTorch](https://pytorch.org/) is "an open source deep learning platform that provides a seamless path from research prototyping to production deployment". It is low-level enough to offer a lot of control over what's going on under the hood during training, and its [dynamic computational graph](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1) allows for very easy debugging. However, being a generic deep learning framework, it is not adapted to the needs of the medical image field.

One can [extend a NiftyNet application](https://niftynet.readthedocs.io/en/dev/extending_app.html), but it's not straightforward without being familiar with the framework and being fluent in TensorFlow 1.X.

So why not use [both](https://www.youtube.com/watch?v=vqgSO8_cRio&feature=youtu.be&t=5)? This tutorial shows how to port the parameters of model trained on NiftyNet to a PyTorch model and test the model while using NiftyNet's I/O modules, which specialize in medical image processing.
