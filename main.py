# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:46:37 2023

@author: Panagiotis Doupidis
"""
import argparse
import sys
import pandas as pd
import torch
from torch import nn
from utils import *
from pathlib import Path
from preprocessor import FinancialPreprocessor
from model import FusedRTRNN
from train import RTRNNTrainer


def create_parser():
    """
    Create an ArgumentParser object.

    Returns
    -------
    argparse.ArgumentParser
        The ArgumentParser object.
    """
    parser = argparse.ArgumentParser(
        description='Train a fused RTRNN on financial data',
        formatter_class=SingleLineFormatter)
    parser.add_argument('--data_path', type=str, required=True,
                        help='The path to the CSV file with the data.')
    parser.add_argument('--test_size', type=int, required=True,
                        help='The number of days to be used for testing the model.')
    parser.add_argument('--window_size', type=int, required=True,
                        help='The size of the window to use for creating sequences from the data.')
    parser.add_argument('--input_size', type=int, default=4,
                        help='The size of the input layer of the model.')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256],
                        help='A list of hidden sizes for each RTRNN block in the model.')
    parser.add_argument('--output_size', type=int, default=1,
                        help='The size of the output layer of the model.')
    parser.add_argument('--num_layers', type=int, nargs='+', default=[1],
                        help='A list of the number of layers for each RTRNN block in the model.')
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='The number of RTRNN blocks in the model.')
    parser.add_argument('--pretrain_epochs', type=check_pretrain_epochs, default=None,
                        help='The number of pre-training epochs.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='The learning rate for training the model. Defaults to 0.001.')
    parser.add_argument('--patience', type=int, default=5,
                        help='The patience parameter for early stopping.')
    parser.add_argument('--seed', type=int, default=42, help='Sets the seed.')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='The threshold parameter for early stopping.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='The dropout probability for the linear layers.')
    parser.add_argument('--val_size', type=check_val_size, default=0.1,
                        help='The fraction of training data used for validation.')
    parser.add_argument('--device', type=str, choices=[
                        'cpu', 'cuda'], default='cpu',
                        help='The device to use for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='The number of training epochs.')
    # Add a mutually exclusive group for the load_model and model_save_path arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--load_model', type=str, default=None,
                       help='The path to load a trained model from.')
    group.add_argument('--model_save_path', type=str, default=None,
                       help='The path to save the trained model.')
    # Add a mutually exclusive group for specifying a file path to save or load a sklearn scaler
    scaler_group = parser.add_mutually_exclusive_group()
    scaler_group.add_argument('--scaler_save_path', type=str, default=None,
                              help='The file path to save a sklearn scaler (optional).')
    scaler_group.add_argument('--load_scaler', type=str, default=None,
                              help='The file path to load a sklearn scaler from (optional).')
    # Add an argument for specifying a file-like object or a string with the name of the file to save the comparative plot to
    parser.add_argument('--plot_save_file', type=str, default=None,
                        help='A file-like object or a string with the name of the file to save the comparative plot to.')
    # Add an argument that will store True if present
    parser.add_argument('--single', action=SingleAction, nargs=0,
                        help='Performs day-ahead prediction.')
    parser.add_argument('--single_prediction_data', type=str, required=False,
                        help='The path to the CSV file with the data = window size.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite the existing model if it exists.')

    return parser


def main():
    """
    Main function.

    Raises
    ------
    ValueError

    Returns
    -------
    None
    """
    parser = create_parser()
    args = parser.parse_args()

    # Only day-ahead prediction
    if args.single:
        if args.load_model is not None:
            load_model_path = Path(args.load_model)
            if not load_model_path.is_file() or load_model_path.suffix != '.json':
                parser.error(
                    f"The load_model argument must be a valid JSON file path, but got: {args.load_model}")
            trainer = RTRNNTrainer.from_file(load_model_path)
            if trainer.model.window_size != args.window_size:
                raise ValueError('window size from user (%d) != trained model (%d)' % (
                    args.window_size, trainer.model.window_size))
            # read new data
            s_data = pd.read_csv(args.single_prediction_data,
                                 parse_dates=True, infer_datetime_format=True,
                                 index_col=0, on_bad_lines='warn')
            # model inference
            single_tensor = FinancialPreprocessor.get_single(
                s_data, args.window_size, args.load_scaler)
            with Seeded(args.seed):
                single_pred = trainer.predict_single(
                    single_tensor.to(args.device))
            # inverse transform
            scaler = FinancialPreprocessor.load_scaler(args.load_scaler)
            final_single_pred = scaler.inverse_transform(
                arr2d_from_val(single_pred, s_data.shape[-1])).squeeze()[-1]
            # print prediction
            print(
                f'The predicted value for the next day is {final_single_pred:.5f}')
            sys.exit()
        else:
            raise ValueError(
                'Cannot perform day-ahead prediction without a trained model.')

    print_args(args)
    data_path = Path(args.data_path)

    if not data_path.is_file():
        raise ValueError(f"{data_path} is not a valid file path")

    data = pd.read_csv(data_path, parse_dates=True, infer_datetime_format=True,
                       index_col=0, on_bad_lines='warn')
    print(f'{data_path.stem} contains {data.shape[0]} valid rows')

    test_size = args.test_size
    preprocessor = FinancialPreprocessor(data, test_size, args.window_size)
    x_train, y_train, x_val, y_val = preprocessor.get_train_val_data(
        args.val_size)
    x_test, y_test = preprocessor.get_test_data()

    if args.load_model is not None:
        load_model_path = Path(args.load_model)
        if not load_model_path.is_file() or load_model_path.suffix != '.json':
            parser.error(
                f"The load_model argument must be a valid JSON file path, but got: {args.load_model}")
        print('Loading from JSON file', load_model_path.name)
        trainer = RTRNNTrainer.from_file(load_model_path)
        if trainer.model.window_size != args.window_size:
            raise ValueError('window size from user (%d) != trained model (%d)' % (
                args.window_size, trainer.model.window_size))
    else:
        # create model
        model = FusedRTRNN(args.input_size, args.hidden_sizes,
                           args.output_size, args.num_layers, args.num_blocks,
                           window_size=args.window_size, dropout_prob=args.dropout,
                           device=args.device).to(args.device)

        # define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, foreach=True)

        # Create an instance of the TRNNTrainer class
        trainer = RTRNNTrainer(model=model, criterion=criterion, optimizer=optimizer,
                               patience=args.patience, threshold=args.threshold,
                               num_epochs=args.epochs)

        # Train the model
        print(f'{"=" * 15} Train {"=" * 15}')
        print(
            f'Train size: {x_train.shape[0]} - Validation size: {x_val.shape[0]}')
        print('=' * 37)
        trainer.train(x_train.to(args.device), y_train.to(args.device),
                      x_val.to(args.device), y_val.to(args.device), args.pretrain_epochs)

        if args.model_save_path:
            print('=' * 40)
            trainer.save_model(args.model_save_path, args.overwrite)
        if args.scaler_save_path:
            print('=' * 40)
            preprocessor.save_scaler(args.scaler_save_path)

    # Evaluate the model on test data and calculate evaluation metrics
    predictions, y_test_np = trainer.evaluate(
        x_test.to(args.device), y_test.to(args.device))

    df_pred_act = inverse_transform_df(preprocessor.test_data_scaled,
                                       preprocessor.scaler,
                                       predictions.squeeze())

    # Check if the plot_save_file argument is selected
    if args.plot_save_file:
        plot_save_file = Path(args.plot_save_file)
        if plot_save_file.exists():
            raise ValueError(f"{plot_save_file} already exists")
        if not plot_save_file.suffixes or plot_save_file.suffixes[-1] != '.png':
            plot_save_file = plot_save_file.with_suffix('.png')
        # Call the plot_result function to save the plot to the specified file
        plot_result(df_pred_act, f'Actual vs Prediction {data_path.stem}',
                    save_file=plot_save_file)
    else:
        plot_result(df_pred_act, f'Actual vs Prediction {data_path.stem}')


if __name__ == '__main__':
    main()
