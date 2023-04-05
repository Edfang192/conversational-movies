import argparse
import logging
import os
import sys

from dataset import MyDataset
from model import MyModel


def get_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--regenerate',
        help='force regenerate all data, embeddings and model',
        action='store_true'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help='directory where everything is saved',
        default='new_tmp'
    )
    parser.add_argument(
        '-f', '--feature_importances',
        help='whether to print feature importances of the model',
        action='store_true'
    )
    parser.add_argument(
        '-cf', '--cf_type',
        type=str,
        help='which cf model to use',
        choices=['svd', 'svdpp', 'knn'],
        default='knn'
    )
    parser.add_argument(
        '-b', '--bert-model',
        type=str,
        default='all-mpnet-base-v2',
        help='''
            which pretrained SentenceTransformer model to use
            (e.g. "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1", etc)
        '''
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    args = parser.parse_args()
    args.outpath = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])),
        args.directory
    )
    return args


if __name__ == '__main__':
    '''
    args = parse_args()
    args.logger = get_logger()
    dataset = MyDataset(args)
    dataset.load_all_features()
    model = MyModel(args)
    model.fit(dataset)
    '''
    args = parse_args()
    args.logger = get_logger(args)
    dataset = MyDataset(args)
    dataset.load_all_features()

    # Train the NeuralMatrixFactorization baseline model
    num_users = len(dataset.data['user_id'].unique())
    num_items = len(dataset.data['movie_id'].unique())
    num_factors = 20
    hidden_units = [64, 32, 16, 8]
    dropout_rate = 0.2
    l2_reg = 1e-5
    learning_rate = 0.001

    nmf_model = NeuralMatrixFactorization(num_users, num_items, num_factors, hidden_units, dropout_rate, l2_reg, learning_rate)

    # Prepare data for the NMF model (adjust the column names according to your dataset)
    X_nmf = dataset.data[['user_id', 'movie_id']]
    y_nmf = dataset.data['score']

    X_train_nmf, X_test_nmf, y_train_nmf, y_test_nmf = tts(X_nmf, y_nmf, test_size=0.2, random_state=args.seed)

    # Fit the NMF model
    nmf_model.fit(X_train_nmf, y_train_nmf, batch_size=256, epochs=20, validation_split=0.1)

    # Evaluate the NMF model
    y_pred_nmf = nmf_model.predict(X_test_nmf)
    args.logger.info(f'NMF RMSE: {math.sqrt(mse(y_test_nmf, y_pred_nmf)):.4f}')
    args.logger.info(f'NMF MAE: {mae(y_test_nmf, y_pred_nmf):.4f}')

    # Train and evaluate the existing model
    model = MyModel(args)
    model.fit(dataset)
