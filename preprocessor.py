# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:44:22 2023.

@author: Panagiotis Doupidis
"""
import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import os
import warnings
import joblib
import os.path
from utils import create_sequences


class FinancialPreprocessor:
    """
    A class for preprocessing financial data for use with a machine learning model.

    Parameters
    ----------
    data : pd.DataFrame
        The financial data to preprocess, as a pandas DataFrame.
    test_size : int
        The size of the test set.
    scaler : BaseEstimator, TransformerMixin, optional
        The scikit-learn scaler to use for scaling the data. Defaults to None.
    """

    def __init__(self, data: pd.DataFrame, test_size: int, window: int,
                 scaler: BaseEstimator = None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("The data must be a pandas DataFrame")

        self.data = data
        self.window = window
        self.test_size = test_size
        self.scaler = scaler if scaler else StandardScaler().set_output(
            transform='pandas')

        # Check that the provided scaler is a valid scikit-learn scaler
        if not isinstance(self.scaler, (BaseEstimator, TransformerMixin)):
            raise ValueError(
                "The provided scaler must be a valid scikit-learn scaler")

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get preprocessed training data as PyTorch tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the preprocessed training data as tensors.
        """
        # Split data into training set
        train_data = self.data[:-self.test_size]

        # Scale the data
        train_data_scaled = self.scaler.fit_transform(train_data)

        x_train, y_train = create_sequences(train_data_scaled, self.window)

        # Convert data to PyTorch tensors
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()

        return x_train, y_train

    def get_train_val_data(self,
                           val_fraction: float = 0.1) -> Tuple[torch.Tensor,
                                                               torch.Tensor,
                                                               torch.Tensor,
                                                               torch.Tensor]:
        """
        Get preprocessed training and validation data as PyTorch tensors.

        Parameters
        ----------
        val_fraction : float
            The size of the validation set as a fraction of the training set.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the preprocessed training and validation data as
            tensors.
        """
        # Split data into training and test sets
        train_data = self.data[:-self.test_size]

        # Calculate the size of the validation set
        self.val_size = int(len(train_data) * val_fraction)
        if self.val_size < self.window:
            raise ValueError(f'Validation size ({self.val_size}) cannot be < window ({self.window})\
. Consider increasing the fraction of data used for validation.')
        # Split the training data into training and validation sets
        train_data, val_data = train_data[:-
                                          self.val_size], train_data[-self.val_size:]

        # Scale the data
        train_data_scaled = self.scaler.fit_transform(train_data)
        val_data_scaled = self.scaler.transform(val_data)

        x_train, y_train = create_sequences(train_data_scaled, self.window)
        x_val, y_val = create_sequences(val_data_scaled, self.window)

        # Convert data to PyTorch tensors
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).float()

        return x_train, y_train, x_val, y_val

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get preprocessed testing data as PyTorch tensors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the preprocessed testing data as tensors.
        """
        # Split data into testing set
        test_data = self.data[-self.test_size:]

        # Scale the data
        self.test_data_scaled = self.scaler.transform(test_data)

        x_test, y_test = create_sequences(self.test_data_scaled, self.window)

        # Convert data to PyTorch tensors
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()

        return x_test, y_test

    @staticmethod
    def get_single(data: Union[pd.DataFrame, np.ndarray],
                   window_size: int,
                   scaler_path: Union[str, os.PathLike]) -> torch.Tensor:
        """
        Convert input data to a float torch tensor.

        Args
        ----
            data(Union[pd.DataFrame, np.ndarray]): Input data, either a pandas
                                                 DataFrame or a 2D numpy array.
            window_size(int): The number of rows that the input data must have.
            scaler_path(Union[str, os.PathLike]): The path to a saved joblib
                                             object containing a fitted scaler.

        Returns
        -------
            torch.Tensor: The resulting float torch tensor.

        Raises
        ------
            AssertionError: If the input data is not a 2D numpy array or if it
                            does not have the specified number of rows.
            ValueError: If the loaded scaler is not a valid estimator.
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            assert data.ndim == 2, "Input numpy array must have 2 dimensions"
        assert data.shape[0] == window_size, f"Input data must have {window_size} rows"

        scaler = FinancialPreprocessor.load_scaler(scaler_path)

        # Suppress any UserWarning raised when transforming the data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            data = scaler.transform(data).to_numpy()
            data = np.expand_dims(data[..., :-1], 0)

        return torch.from_numpy(data).float()

    @staticmethod
    def load_scaler(scaler_path: str) -> Union[BaseEstimator,
                                               TransformerMixin]:
        """
        Load a fitted scaler from the specified file path.

        Parameters
        ----------
        scaler_path : str
            The file path to load the fitted scaler from.

        Returns
        -------
        scaler : BaseEstimator or TransformerMixin
            The loaded fitted scaler.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        ValueError
            If the loaded scaler is not a valid scikit-learn estimator/scaler.
        """
        try:
            with open(scaler_path, 'rb') as f:
                scaler = joblib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError('Path to scaler does not exist.')
        # Check if the loaded scaler is a valid scikit-learn estimator/scaler
        if not isinstance(scaler, (BaseEstimator, TransformerMixin)):
            raise ValueError(
                "The loaded scaler must be a valid scikit-learn estimator")
        return scaler

    def save_scaler(self, filename: str):
        """
        Save a fitted instance of a scikit-learn StandardScaler to a file.

        Args
        ----
            filename(str): The name of the file to save the scaler to.

        Raises
        ------
            NotFittedError: If the scaler is not fitted.

        """
        check_is_fitted(self.scaler)
        # Check if the provided filename contains an extension
        root, ext = os.path.splitext(filename)
        # If an extension is present, replace it with the typical joblib ext.
        filename = root + '.joblib' if not ext or ext != '.joblib' else filename

        with open(filename, 'wb') as f:
            joblib.dump(self.scaler, f)

        print(f"The fitted scaler was saved to: {filename}")
