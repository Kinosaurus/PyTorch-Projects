# CT Scan Image Classification

Have recently learned PyTorch using @mrdbourke's amazing resources:
  * FreeCodeCamp x @mrdbourke Youtube Tutorial: https://www.youtube.com/watch?v=V_xro1bcAuA
  * @mrdbourke's amazing website textbook: https://www.learnpytorch.io/

Data from @mohamedhanyyy:
  * https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code

Here's the main gist of this notebook:
1. Transfer learning from EfficientNet_V2_S model, available on torchvision.
2. Customisation of the model to the 4 CT Scan Image categories.
3. Training the model on the CT Scan Image datasets.
4. Running experiments through Weights and Biases (wandb) to obtain the optimum learning rate.
5. Fine-tuning the model.
6. Visualising model's actual predictions on random images.
7. Model evaluation with a Confusion Matrix.
