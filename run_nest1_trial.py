import click
import numpy as np
from models.NEST1 import NEST1
from data.data_loading import load_data
    


@click.command()
@click.argument("train_path")
@click.argument("test_path")
@click.argument("rank", type=int)
@click.option('--lr', default=0.001, help='learning rate', type=float)
@click.option('--batchsize', default=256, help = "batch size for Atom", type=int)
@click.option('--nepoch', default=300, help='nubmer of epochs', type=int)
@click.option('--m', default=100, help = 'number of fourier features', type=int) 
def run_trial(train_path, test_path, rank, lr, batchsize, nepoch, m):
    
    ind, y = load_data(train_path)
    ind_test, y_test = load_data(test_path)
    print("Data loaded")

    nvecs = ind_test.max(axis=0)+1
    model = NEST1(ind, nvecs, rank, y, m, batchsize, lr)
    model.train(nepoch=nepoch)
    print("Model finished training\n")
    y_pred = model.test(ind_test)

    mse = np.mean(np.power(y_pred - y_test, 2))
    mae = np.mean(np.abs(y_pred - y_test))
    print('Test Data Results: mse = %g, mae = %g'%(mse, mae))


if __name__ == '__main__':
    run_trial()