# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:42:47 2023.

@author: Panagiotis Doupidis
"""
import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, Type, Union
import numpy as np
import os.path
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from utils import smape as sMAPE
from functools import singledispatchmethod
from model import FusedRTRNN


class RTRNNTrainer:
    """
    A class for training and evaluating a RTRNN model.

    Parameters
    ----------
    model : nn.Module
        The fused RTRNN model to train and evaluate.
    criterion : callable
        The loss function to use during training.
    optimizer : torch.optim.Optimizer
        The optimizer to use during training.
    patience : int
        The patience for early stopping.
    threshold : float
        The threshold for early stopping.
    num_epochs : int
        The number of epochs to train the model for.
    """

    def __init__(self, model: FusedRTRNN, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 patience: int, threshold: float, num_epochs: int):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.patience_counter = 0
        self.prev_loss = float('inf')
        self.fitted = False
        self.es = EarlyStopping(patience, threshold)

    def early_stopping(self, loss: float) -> bool:
        """
        Check if training should be stopped early based on the current loss.

        Parameters
        ----------
        loss : float
            The current loss.

        Returns
        -------
        bool
            A boolean indicating whether to stop training early.
        """
        if abs(self.prev_loss - loss) < self.threshold or self.prev_loss < loss:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                return True
        else:
            self.patience_counter = 0
        return False

    @singledispatchmethod
    def train(self, *args, **kwargs):
        """Generic train method."""
        raise NotImplementedError("Invalid number of arguments provided")

    def _(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Train the TRNN model on the provided training data.

        Parameters
        ----------
        x_train : torch.Tensor
            The training data inputs.
        y_train : torch.Tensor
            The training data targets.
        """
        # Set the model to training mode
        self.model.train()

        for epoch in range(self.num_epochs):
            outputs = self.model(x_train)
            loss = self.criterion(outputs.squeeze(), y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(
                    f'Epoch: {epoch+1:4d}/{self.num_epochs}, Loss: {loss.item():.4f}')

            stop = self.early_stopping(loss.item())
            if stop:
                self.fitted = True
                print(
                    f'Early stopping at epoch {epoch+1}. Loss has not decreased by more than {self.threshold} for {self.patience} consecutive epochs.')
                break

            self.prev_loss = loss.item()

        self.fitted = True

    @train.register
    def _(self, x_train: torch.Tensor, y_train: torch.Tensor,
          x_val: torch.Tensor, y_val: torch.Tensor, pretrain_epochs: Union[int, None]) -> None:
        """
        Train the TRNN model on the provided training data.

        Parameters
        ----------
        x_train : torch.Tensor
            The training data inputs.
        y_train : torch.Tensor
            The training data targets.
        x_val : torch.Tensor
            The validation data inputs.
        y_val : torch.Tensor
            The validation data targets.
        pretrain_epochs : int
            The number of pre-training epochs.
        """
        # Set the model to training mode
        self.model.train()

        if pretrain_epochs is not None:
            # Pre-training stage
            with tqdm(total=pretrain_epochs, desc='Pre-training', unit='epoch',
                      ascii=True, bar_format='{desc}: {n_fmt}/{total_fmt} {postfix}',
                      initial=0) as pbar:
                for epoch in range(pretrain_epochs):
                    outputs = self.model(x_train)
                    loss = self.criterion(outputs.squeeze(), y_train)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.update(1)
                pbar.set_postfix_str('done')

        # Main training loop
        for epoch in range(self.num_epochs):
            outputs = self.model(x_train)
            loss = self.criterion(outputs.squeeze(), y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Evaluate the model on the validation set
            with torch.no_grad():
                val_outputs = self.model(x_val)
                val_loss = self.criterion(val_outputs.squeeze(), y_val)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch: {epoch+1:4d}/{self.num_epochs}, Loss: {loss.item():.4f}, Val. Loss: {val_loss.item():.4f}')

            if self.es(val_loss):
                self.fitted = True
                print(
                    f'Early stopping at epoch {epoch+1}. Validation loss has not decreased by more than {self.threshold} for {self.patience} consecutive epochs.')
                break

        self.fitted = True

    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Evaluate the fused RTRNN model on the provided testing data and 
        calculate evaluation metrics.

        Parameters
        ----------
        x_test : torch.Tensor
            The testing data inputs.
        y_test : torch.Tensor
            The testing data targets.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the model's predictions on the testing data and
            the testing data targets as numpy arrays.
        """

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(x_test)
            predictions = predictions.detach().cpu().numpy()
            y_test_np = y_test.detach().cpu().numpy()

        # Calculate evaluation metrics
        mse = MSE(y_test_np, predictions)
        rmse = MSE(y_test_np, predictions, squared=False)
        mae = MAE(y_test_np, predictions)
        smape = sMAPE(y_test_np, predictions)

        print('=' * 35)
        print(
            f'Performance metrics for an evaluation periods of {len(y_test)} timesteps')
        print('=' * 35)
        print(f'MSE:{mse:10.5f}')
        print(f'RMSE:{rmse:9.5f}')
        print(f'MAE:{mae:10.5f}')
        print(f'sMAPE:{smape:6.3f} %')

        return predictions, y_test_np

    @torch.no_grad()
    def predict_single(self, x_test: torch.Tensor) -> float:
        """Get single one-step-ahead prediction."""
        return self.model(x_test).detach().item()

    def save_model(self, dirname, overwrite: bool = False) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        dirname : str
            The directory to save the model and its parameters.
        overwrite : bool
            Whether to overwrite the file if it already exists.
        """
        # Check if the model is fitted before saving it
        if not self.fitted:
            raise ValueError("The model must be fitted before it can be saved")

        # Append the current timestamp to the directory name
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dirname = f'{dirname}_{timestamp}'

        # Create the directory
        os.makedirs(dirname, exist_ok=overwrite)

        # Check if the file already exists
        filename = os.path.join(dirname, 'model.pt')
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"The file {filename} already exists. Set overwrite=True to overwrite it.")

        # Save the model state dictionary to the specified file
        torch.save(self.model.state_dict(), filename)

        # Save a JSON file containing the path to the saved model file
        with open(os.path.join(dirname, 'model_params.json'), 'w') as f:
            json.dump({'model_state_dict': os.path.abspath(filename),
                       'model_params': self.model.params,
                       'lr': self.optimizer.param_groups[0]['lr'],
                       'patience': self.patience,
                       'threshold': self.threshold,
                       'num_epochs': self.num_epochs}, f,
                      indent=2, sort_keys=True)

        print('Model and params saved to', dirname)

    @classmethod
    def from_file(cls: Type['RTRNNTrainer'], json_file: str) -> 'RTRNNTrainer':
        """
        Load a RTRNNTrainer instance from a JSON file.

        The JSON file should have the following structure:

            {
                "model_params": {...},
                "model_state_dict": "path/to/state_dict/file",
                "lr": ...,
                "patience": ...,
                "threshold": ...,
                "num_epochs": ...
            }

        where `model_params` is a dictionary containing the parameters for 
        initializing the RTRNN model, `model_state_dict` is the path to a file
        containing the saved state dictionary of the model, `lr` is the learning
        rate for the optimizer, `patience` is the patience for early stopping,
        `threshold` is the threshold for early stopping, and `num_epochs`
        is the number of epochs to train the model for.

        Parameters
        ----------
        json_file : str
            Path to the JSON file containing the model parameters.

        Returns
        -------
        RTRNNTrainer
            Instance of the RTRNNTrainer class.
        """

        with open(json_file, 'r') as f:
            params = json.load(f)

        # Load the model from the saved state dict
        model = FusedRTRNN(**params['model_params'])
        model.load_state_dict(torch.load(params['model_state_dict'],
                                         map_location=torch.device(
                                             params['model_params']['device']))
                              )

        # Create the criterion, optimizer, and other trainer parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                      foreach=True)
        patience = params['patience']
        threshold = params['threshold']
        num_epochs = params['num_epochs']

        # Create and return the trainer instance using the class constructor
        return cls(model, criterion, optimizer, patience, threshold, num_epochs)


class EarlyStopping:
    """Early stopping w/ validation loss."""

    def __init__(self, patience: int, delta: float):
        """
        Initialize the early stopping object.

        Parameters
        ----------
        patience : int
            The number of epochs to wait before stopping training if the
            validation loss does not improve.
        delta : float
            The minimum change in validation loss required to consider it as
            an improvement.
        """

        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped early based on current val. loss.

        Parameters
        ----------
        val_loss : float
            The current validation loss.

        Returns
        -------
        bool
            A boolean indicating whether to stop training early.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False
