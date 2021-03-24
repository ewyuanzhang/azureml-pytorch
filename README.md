# Azure Machine Learning PyTorch example notebooks

In this tutorial, you will train, hyperparameter tune, and deploy a PyTorch model using the Azure Machine Learning (Azure ML) Python SDK.

This tutorial will train an image classification model using transfer learning, based on PyTorch's [Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The model is trained to classify chickens and turkeys by first using a pretrained ResNet18 model that has been trained on the [ImageNet](http://image-net.org/index) dataset.

The notebook is modified from [PyTorch hyperparameter tuning notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch/train-hyperparameter-tune-deploy-with-pytorch.ipynb) in Azure ML tutorials and the [model packaging documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-package-models). The main difference is that this notebook focuses more on non-Notebook VM development environment behind GFW, more specifically:
- Upload image dataset from local machine and mount it to GPU cluster during training.
- Use Conda channel and PyPI repository mirrors behind GFW to avoid network connection issue when building images in Azure ML in Azure China.
- Package the output model to docker image explicitly.

## Prerequisites
If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, make sure you have the Azure Machine Learning Python SDK install through the command below and create an [Azure ML Workspace](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb).
```
pip install notebook azureml-core azure-cli-core azureml-sdk azureml-widgets
```
Then run launch Jupyter notebook.
```
jupyter notebook
```