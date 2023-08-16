# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:45:15 2023.

@author: Panagiotis Doupidis
"""

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
import random
import warnings
import inspect
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator


def create_sequences(data: pd.DataFrame, seq_length: int) -> Tuple[np.ndarray,
                                                                   np.ndarray]:
    """
    Create sequences of data for use with a time series model.

    Parameters
    ----------
    data : pd.DataFrame
        The data to create sequences from, as a pandas DataFrame.
    seq_length : int
        The length of the sequences to create.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the input and target sequences as numpy arrays.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data.iloc[i:(i + seq_length), :-1]
        y = data.iloc[i + seq_length][-1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def get_params(frame):
    args, _, _, values = inspect.getargvalues(frame)
    return {arg: values[arg] for arg in args if arg not in ('self',
                                                            '__class__')}


def print_args(args):
    print("┌" + "─" * 40 + "┐")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}  ")
    print("└" + "─" * 40 + "┘")


class SingleLineFormatter(argparse.HelpFormatter):
    """Argument parser formatter."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(f'{option_string}')
            return ', '.join(parts)


class Seeded:
    """
    A context manager for setting the seed for random number generators.

    This context manager sets the seed for the random number generators in the
    `random`, `numpy`, and `torch` modules, as well as for all CUDA devices if
    CUDA is available. The seed value is specified when creating an instance of
    the `Seeded` class and is used for all random number generators when entering
    the `with` block.

    Parameters
    ----------
    seed : int, optional
        The seed value to use for the random number generators. Defaults to 42.
    """

    def __init__(self, seed=42):
        self.seed = seed

    def __enter__(self):
        """
        Set the random seed for reproducibility.

        This method sets the random seed for the `random`, `numpy`, and
        `torch` modules, as well as for CUDA (if available) to ensure
        reproducibility of results.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            # Ensure that all operations are deterministic on GPU (if used)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager.

        This method is called when exiting the `with` block and currently
        does nothing.
        """
        pass


def inverse_transform_df(test: pd.DataFrame, scaler,
                         new_close_values: np.ndarray) -> pd.DataFrame:
    """Transform the contents of a scaled dataframe back to its original."""
    # Create a copy of the input dataframe to avoid modifying the original
    test_copy = test.copy()

    # Inverse transform the original dataframe
    unscaled_df = pd.DataFrame(scaler.inverse_transform(
        test_copy), columns=test_copy.columns, index=test_copy.index)

    # Store the values in the 'Close' column to a variable
    close_values = unscaled_df['Close'].values

    # Fill the new_close_values array with NaN values from the start to match the length of the dataframe
    new_close_values = np.concatenate(
        (np.full(test_copy.shape[0] - new_close_values.shape[0], np.nan),
         new_close_values))

    # Replace the values in the 'Close' column of the original dataframe with new values using .loc
    test_copy.loc[:, 'Close'] = new_close_values

    # Inverse transform the modified dataframe
    unscaled_df_modified = pd.DataFrame(scaler.inverse_transform(
        test_copy), columns=test_copy.columns, index=test_copy.index)

    # Store the new unscaled values in the 'Close' column to a variable
    new_close_values_unscaled = unscaled_df_modified['Close'].values

    # Create a new dataframe with columns 'actual' and 'pred'
    result = pd.DataFrame(
        {'actual': close_values, 'pred': new_close_values_unscaled},
        index=test_copy.index)

    return result


def check_val_size(val_size: float) -> float:
    """
    Check if the val_size argument is within the desired range.

    Parameters
    ----------
    val_size : float
        The value of the val_size argument.

    Returns
    -------
    float
        The value of the val_size argument if it is within the desired range.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not within the desired range.
    """
    val_size = float(val_size)
    if 0 <= val_size < 1:
        if val_size > 0.5:
            warnings.warn(
                f"Validation fraction is greater than 0.5 ({val_size}), which may result in a small training set")
        return val_size
    else:
        raise argparse.ArgumentTypeError(
            f"val_size must be between 0 and 1, but got {val_size}")


def check_pretrain_epochs(pretrain_epochs: Union[int, None]) -> Union[int,
                                                                      None]:
    """
    Check if the pretrain_epochs argument is within the desired range.

    Parameters
    ----------
    pretrain_epochs : Union[int, None]
        The number of the pretrain_epochs argument or None.

    Returns
    -------
    int
        The value of the pretrain_epochs argument if it is within the
        desired range or None.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not within the desired range.
    """
    if pretrain_epochs is not None:
        value = int(pretrain_epochs)
        _max_pe = 50

        # Check if the pretrain_epochs argument is less than a set number
        if value >= _max_pe:
            raise argparse.ArgumentTypeError(
                f'pretrain_epochs must be less than {_max_pe}')
        return value
    return None


def arr2d_from_val(value: int, n: int) -> np.ndarray:
    """
    Create a 2D numpy array with the given value at the end and 0s in the
    first n-1 spots.

    Args
    ----
        value: The value to place at the end of the array.
        n: The number of spots in the array.

    Returns
    -------
        A 2D numpy array with shape (1, n).
    """
    # Create an array of zeros with shape (1, n)
    result = np.zeros((1, n))

    # Set the last element of the array to the given value
    result[0, -1] = value

    return result

# Define a custom action for the --single argument


class SingleAction(argparse.Action):
    """Ensures required arguments are present in single prediction mode."""

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Check if required arguments are present in single prediction mode.

        This method is called when the `--single` argument is passed to the
        command line. It checks if the `--load_scaler` and `--single_prediction_data`
        arguments are also provided and not None. If either of these arguments
        is missing or None, an error message is displayed and the program exits.
        If both arguments are present and not None, the `single` attribute of
        the namespace is set to True.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser object.
        namespace : argparse.Namespace
            The namespace object containing the parsed arguments.
        values : Any
            The associated command-line arguments.
        option_string : str, optional
            The option that was used to invoke this action. Defaults to None.
        """
        # Check if load_scaler is not None
        if not namespace.load_scaler or not namespace.single_prediction_data:
            parser.error(
                "`--single requires` `--load_scaler` and `--single_prediction_data` \
to be provided and not None. If you have specified both arguments and still \
encounter this error make sure that `--load_scaler` and `--single_prediction_data` \
preceed `--single`.")
        # Set the single attribute to True
        setattr(namespace, self.dest, True)


def plot_result(result: pd.DataFrame, title: str,
                save_file: Optional[str] = None):
    """
    Plot the result of a DataFrame with 'actual' and 'pred' columns.

    This function creates a line plot of the 'actual' and 'pred' columns of
    input DataFrame. The x-axis is    formatted as '%d-%m-%y' with weekly
    intervals. The y-axis is formatted with a '€' sign and 3 decimal places of
    precision. The plot can be saved to a file or displayed on screen.

    Parameters
    ----------
    result : pd.DataFrame
        The input DataFrame with 'actual' and 'pred' columns.
    title : str
        The title of the plot.
    save_file : str, optional
        The file path where to save the final plot. If provided, the figure
        will be saved as a png file with the highest possible resolution
        and clarity. If not provided, the plot will be displayed on screen.

    Returns
    -------
    None
    """
    # Create a line plot of the 'actual' and 'pred' columns
    fig, ax = plt.subplots(figsize=(10, 6))
    result.plot(y=['actual', 'pred'], style=['-', '-'], ax=ax)

    # Add circle markers at weekly intervals, falling on the actual lines
    first_date = result.index[0]
    for label, color in zip(['actual', 'pred'], ['tab:blue', 'tab:orange']):
        x = result.index[result.index >= first_date]
        y = result.loc[x, label]
        ax.plot(x[::5], y[::5], 'o', color=color)

    # Format the x-axis as '%d-%m-%y' with weekly intervals
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())

    # Remove the vertical grid lines
    ax.yaxis.grid(True)

    # Append sign to the y-axis ticker labels while keeping 3 decimals
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}$'))

    # Increase the number of y-axis tickers for higher precision
    ax.yaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))

    # Add a title, legend, and labels
    ax.set_title(title)
    ax.legend(['Actual', 'Prediction'], title='Values', title_fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    plt.tight_layout()
    # Save or show the plot
    if save_file:
        plt.savefig(save_file, format='png', dpi=300)
        print(f"Plot saved to {save_file}")
    else:
        plt.show()
