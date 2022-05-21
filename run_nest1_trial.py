from tkinter import N
import click
import numpy as np
from models.NEST1 import NEST1
from data.data_loading import load_data
    


@click.command()
@click.argument("train_path")
@click.argument("test_path")
@click.argument("rank")
@click.option('--lr', default=0.001, help='learning rate')
@click.option('--batchsize', default=256, help = "batch size for Atom")
@click.option('--nepoch', default=300, help='nubmer of epochs')
@click.option('--m', default=100, help = 'number of fourier features') 
def run_trial(train_path, test_path, rank, lr, batchsize, nepoch, m):
    
    ind, y = load_data(train_path)
    ind_test, y_test = load_data(test_path)

    model = NEST1(ind, rank, y, m, batchsize, lr)
    model.train(nepoch=nepoch)
    print("Model finished training\n")
    y_pred = model.test(ind_test)

    mse = np.mean(np.power(y_pred - y_test, 2))
    mae = np.mean(np.abs(y_pred - y_test))
    print('Test Data Results: mse = %g, mae = %g'%(mse, mae))