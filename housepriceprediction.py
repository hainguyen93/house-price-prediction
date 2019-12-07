import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential
from keras.metrics import mean_absolute_error, mae, mse
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l2, l1_l2

from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')


def load_data(url, columns=[1, 2, 4, 6, 11]):
    """Short summary.

    Parameters
    ----------
    url : type
    Description of parameter `url`.
    columns : type
    Description of parameter `columns`.

    Returns
    -------
    type
    Description of returned object.

    """
    df = pd.read_csv(url, header=None, usecols=columns)
    print('Data shape: ', np.shape(df))

    # re-name all columns
    column_names = ['Price', 'PurchaseDate',
                    'PropertyType', 'LeaseDuration', 'City']
    df.columns = column_names

    # resplace column values
    df['PropertyType'] = df['PropertyType'].replace(
        {'F': 0, 'D': 1, 'S': 2, 'T': 3, 'O': 4})
    df['LeaseDuration'] = df['LeaseDuration'].replace({'L': 0, 'F': 1, 'U': 2})
    df.loc[df['City'] == 'LONDON', 'City'] = 0
    df.loc[df['City'] != 0, 'City'] = 1

    # convert column values to appropriate dtype to save memory
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    df['Price'] = pd.to_numeric(df["Price"], downcast="integer")
    df['PropertyType'] = pd.to_numeric(df['PropertyType'], downcast='integer')
    df['LeaseDuration'] = pd.to_numeric(
        df["LeaseDuration"], downcast="integer")
    df['City'] = pd.to_numeric(df["City"], downcast="integer")

    return df


def split_train_test(df):
    """Short summary.

    Parameters
    ----------
    df : type
    Description of parameter `df`.

    Returns
    -------
    type
    Description of returned object.

    """
    cutoff = datetime(2016, 1, 1)
    column_sels = ['Price', 'PropertyType', 'LeaseDuration', 'City']
    train_df = df.loc[df['PurchaseDate'] <= cutoff][column_sels]
    test_df = df.loc[df['PurchaseDate'] > cutoff][column_sels]

    # remove duplicates
    train_df.drop_duplicates(keep='first', inplace=True)
    test_df.drop_duplicates(keep='first', inplace=True)

    return train_df, test_df


def split_train_val(train_df):
    """Short summary.

    Parameters
    ----------
    train_df : type
    Description of parameter `train_df`.

    Returns
    -------
    type
    Description of returned object.

    """
    # split to train and val (~20%)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=2019)
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    return train_df, val_df


def prep(train_df, val_df, test_df):
    """Short summary.

    Parameters
    ----------
    train_df : type
    Description of parameter `train_df`.
    val_df : type
    Description of parameter `val_df`.
    test_df : type
    Description of parameter `test_df`.

    Returns
    -------
    type
    Description of returned object.

    """
    # training data
    train_X = train_df[['PropertyType', 'LeaseDuration', 'City']]
    train_y = train_df['Price']

    val_X = val_df[['PropertyType', 'LeaseDuration', 'City']]
    val_y = val_df['Price']

    # testing data
    test_X = test_df[['PropertyType', 'LeaseDuration', 'City']]
    test_y = test_df['Price']

    # one-hot encoding the inputs
    ohc = OneHotEncoder(handle_unknown='ignore')
    ohc.fit(train_X)
    train_X = ohc.transform(train_X)
    val_X = ohc.transform(val_X)
    test_X = ohc.transform(test_X)

    # convert the targets to smaller range
    train_y = np.log1p(train_y * 1e-3)
    val_y = np.log1p(val_y * 1e-3)
    test_y = np.log1p(test_y * 1e-3)

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def build_model(input_size):
    """ Build the model with many fully connected layers
    Arguments:
    input_size : the size of the input layer
    Return: a compiled model
    """
    inp = Input(shape=(input_size,))
    fc1 = Dense(100, activation='relu',
                kernel_regularizer=l2(0.001))(inp)
    do1 = Dropout(0.5)(fc1)
    fc2 = Dense(200, activation='relu',
                kernel_regularizer=l2(0.001))(do1)
    do2 = Dropout(0.5)(fc2)
    fc3 = Dense(100, activation='relu', kernel_regularizer=l1_l2(
        l1=0.001, l2=0.001))(do2)
    do3 = Dropout(0.5)(fc3)
    fc4 = Dense(100, activation='relu', kernel_regularizer=l1_l2(
        l1=0.001, l2=0.001))(do3)
    out = Dense(1)(fc4)

    model = Model(inputs=inp, outputs=out)
    print(model.summary())

    model.compile(optimizer=RMSprop(lr=1e-3),
                  loss=mse, metrics=[mae])
    return model


def train_model(model, epochs=10, b_size=10000):
    """ Perform model traning
    Arguments:
    model : a compiled model
    epochs : number of epochs (i.e., iterations over dataset)
    Return:
    the distionary containing training error/loss and val error/loss
    """
    history = model.fit(train_X, train_y, batch_size=b_size, verbose=1,
                        epochs=epochs, validation_data=(val_X, val_y))

    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    return train_mae, val_mae


def training_val_error_plotter(train_mae, val_mae):
    """Short summary.

    Parameters
    ----------
    train_mae : type
    Description of parameter `train_mae`.
    val_mae : type
    Description of parameter `val_mae`.

    Returns
    -------
    type
    Description of returned object.

    """
    """ Plot the train/val loss over epochs
    """
    epochs = len(train_mae)
    plt.plot(range(1, epochs + 1), train_mae,
             label='Training Loss')
    plt.plot(range(1, epochs + 1), val_mae,
             label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def smooth_curve(points, factor=0.9):
    """Short summary.

    Parameters
    ----------
    points : type
    Description of parameter `points`.
    factor : type
    Description of parameter `factor`.

    Returns
    -------
    type
    Description of returned object.

    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(
                previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
            return smoothed_points


if __name__ == '__main__':

    url = 'pp-complete.csv'
    df = load_data(url)
    train_df, test_df = split_train_test(
        df)
    train_df, val_df = split_train_val(
        train_df)
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prep(
        train_df, val_df, test_df)

    # fitting the model using cross validation with k folds
    k = 10
    num_samples = np.shape(
        train_df)[0] // k
    num_epochs = 500
    b_size = 512
    train_errors = []
    val_errors = []

    for i in range(k):
        print(
            'processing fold {0}'.format(i))
        val_data = train_df.iloc[i * num_samples:(
            i + 1) * num_samples]
        train_data = pd.concat(
            [train_df.iloc[:i * num_samples], train_df.iloc[(i + 1) * num_samples:]])
        (train_X, train_y), (val_X, val_y), (test_X, test_y) = prep(
            train_data, val_data, test_df)
        model = build_model(
            np.shape(train_X)[1])
        train_mae, val_mae = train_model(
            model, num_epochs, b_size)
        train_errors.append(train_mae)
        val_errors.append(val_mae)

        avg_train = [
            np.mean(i) for i in zip(*train_errors)]
        avg_val = [
            np.mean(i) for i in zip(*val_errors)]
        # training_val_error_plotter(avg_train, avg_val)

        smooth_avg_val = smooth_curve(
            avg_val, factor=0.8)
        training_val_error_plotter(
            avg_train, smooth_avg_val)
        test_mse, test_mae = model.evaluate(
            test_X, test_y)
        print(
            'Test MAE: {0}'.format(test_mae))
