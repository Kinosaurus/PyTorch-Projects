"""
Contains functionality for creating PyTorch dataloaders for image classification data
"""
import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    train_transform:transforms.Compose,
    test_transform:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS):
  """ Creates training and testing dataloaders.

  Takes in a training and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir (str): Path to training directory.
    test_dir (str): Path to testing directory.
    train_transform (torchvision.transforms.Compose): Apply a series of torchvision transforms to training data.
    test_transform (torchvision.transforms.Compose): Apply a series of torchvision transforms to testing data.
    batch_size (int): Number of samples per batch in each DataLoader.
    num_workers (int): Number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names), where class_names is a list of the target classes.
  
  Example usage:
    create_dataloaders(
      train_dir=path/to/train/dir, 
      test_dir=path/to/test/dir, 
      train_transform=train_transformation, 
      test_transform=test_transformation, 
      batch_size=32, 
      num_workers=2
    )
  """
  # Create train data
  train_data = datasets.ImageFolder(
      root=train_dir,
      transform=train_transform
  )
  # Create test data
  test_data = datasets.ImageFolder(
      root=test_dir,
      transform=test_transform,
  )
  # Get list of class names
  class_names = train_data.classes
  # Create train dataloader
  train_dataloader = DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers
  )
  # Create test dataloader
  test_dataloader = DataLoader(
      dataset=test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers
  )
  return train_dataloader, test_dataloader, class_names
