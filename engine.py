"""
Contains functionality for training and testing the PyTorch model, creates train_step, test_step and train functions.
"""
import torch

from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
  model: nn.Module,
  train_dataloader: torch.utils.data.DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  loss_fn: nn.Module=nn.CrossEntropyLoss()) -> Tuple[float, float]:
  """Trains the model for 1 epoch.

  Takes in a train dataloader, optimizer, device and loss function.

  Returns the average loss and accuracy per training batch.

  Args:
    model (nn.Module): A PyTorch model.
    train_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the training data.
    optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
    device (torch.device): The target device to compute on. (e.g. "cuda" or "cpu").
    loss_fn (nn.Module): A PyTorch loss function to minimize. Default: nn.CrossEntropyLoss().

  Returns:
    A tuple of training loss and training accuracy metrics, in the form (train_loss, train_accuracy).

  Example usage:
    train_step(
      model=model_0,
      train_dataloader=train_dataloader,
      optimizer=optimiser,
      device=device,
      loss_fn=loss_fn,
    )
  """
  # Activate train mode
  model.train()
  # Setup variables to track and accumulate loss and accuracy
  total_loss, total_acc = 0, 0
  # Place model on target device
  model.to(device)
  # Iterate through DataLoader
  for X, y in train_dataloader:
    # Place data on target device
    X, y = X.to(device), y.to(device)
    # Forward pass
    train_logits = model(X)
    # Calculate Loss
    loss = loss_fn(train_logits, y)
    # Accumulate Loss
    total_loss += loss.item()
    # Optimiser Zero-grad
    optimizer.zero_grad()
    # Loss Backward
    loss.backward()
    # Optimiser Step
    optimizer.step()

    # Calculate Acc
    train_labels = train_logits.argmax(dim=1)
    acc = torch.eq(train_labels, y).sum().item()/len(y)
    # Accumulate Acc
    total_acc += acc
  # Calculate Average Loss and Accuracy per batch
  avg_loss = total_loss / len(train_dataloader)
  avg_acc = total_acc /len(train_dataloader)
  return avg_loss, avg_acc

def test_step(
  model: nn.Module,
  test_dataloader: torch.utils.data.DataLoader,
  device: torch.device,
  loss_fn: nn.Module=nn.CrossEntropyLoss()) -> Tuple[float, float]:
  """Evaluates the model for 1 epoch.

  Takes in a test dataloader, device and loss function.

  Returns the average loss and accuracy per testing batch.

  Args:
    model (nn.Module): A PyTorch model.
    test_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the testing data.
    device (torch.device): The target device to compute on. (e.g. "cuda" or "cpu")
    loss_fn (nn.Module): A PyTorch loss function to minimize.

  Returns:
    A tuple of testing loss and testing accuracy metrics, in the form (test_loss, test_accuracy)

  Example usage:
    test_step(
      model=model_0,
      test_dataloader=test_dataloader,
      device=device,
      loss_fn=loss_fn
    )
  """
  # Activate eval mode
  model.eval()
  # Setup variables to track and accumulate loss and accuracy
  total_loss, total_acc = 0, 0
  # Place model on target device
  model.to(device)
  # Activate inference mode
  with torch.inference_mode():
    # Iterate through DataLoader
    for X, y in test_dataloader:
      # Place data on target device
      X, y = X.to(device), y.to(device)
      # Forward pass
      test_logits = model(X)
      # Calculate Loss
      loss = loss_fn(test_logits, y)
      # Accumulate Loss
      total_loss += loss.item()
      # Calculate Acc
      test_labels = test_logits.argmax(dim=1)
      acc = torch.eq(test_labels, y).sum().item()/len(y)
      # Accumulate Acc
      total_acc += acc
  # Calculate Average Loss and Accuracy per batch
  avg_loss = total_loss / len(test_dataloader)
  avg_acc = total_acc /len(test_dataloader)
  return avg_loss, avg_acc

def train(
  model: nn.Module,
  train_dataloader: torch.utils.data.DataLoader,
  test_dataloader: torch.utils.data.DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  epochs: int,
  step_size: int=10,
  gamma: float=0.1,
  loss_fn: nn.Module=nn.CrossEntropyLoss()) -> Dict[str,List]:
  """Performs training and testing on PyTorch model for given number of epochs.

  Takes in a PyTorch model, train dataloader, test dataloader, optimizer, device, epochs, loss_fn.

  Args:
    model (nn.Module): A PyTorch model.
    train_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the training data.
    test_dataloader (torch.utils.data.DataLoader): A PyTorch DataLoader for the testing data.
    optimizer (torch.optim.Optimizer): A PyTorch optimizer to help minimize the loss function.
    device (torch.device): The target device to compute on. (e.g. "cuda" or "cpu")
    epochs (int): The number of iterations to train the model for.
    step_size (int): After every step_size epochs, learning rate to decay. Default: 10.
    gamma (float): Factor for learning rate to drop by. Default: 0.1
    loss_fn (nn.Module): A PyTorch loss function to minimize.

  Returns:
    A dictionary of training and testing loss, training and testing accuracy metrics.
    Each metric has a value in a list for each epoch.
  """
  # Setup empty results dict
  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
  # Setup learning rate scheduler
  scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  # Iterate through epochs
  for epoch in tqdm(range(epochs)):
    # Training
    train_loss, train_acc = train_step(
      model=model,
      train_dataloader=train_dataloader,
      optimizer=optimizer,
      device=device,
      loss_fn=loss_fn
    )
    # Evaluation
    test_loss, test_acc = test_step(
      model=model,
      test_dataloader=test_dataloader,
      device=device,
      loss_fn=loss_fn
    )
    # Update Learning Rate
    scheduler.step()
    # Latest Learning Rate
    learning_rate = scheduler.get_last_lr()[0]
    # Visualise results for this epoch
    print(f"Epoch: {epoch+1} | Learning Rate: {learning_rate} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy {test_acc:.4f}")
    # Update results dict
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  # Return filled results at end of epochs
  return results
