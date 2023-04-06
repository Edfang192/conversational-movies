from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel


class NeuralMatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_movies: int, emb_size: int = 50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.movie_emb = nn.Embedding(n_movies, emb_size)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(user_ids)
        movie_emb = self.movie_emb(movie_ids)
        user_bias = self.user_bias(user_ids)
        movie_bias = self.movie_bias(movie_ids)
        dot = torch.sum(user_emb * movie_emb, dim=1)
        preds = dot + user_bias.squeeze() + movie_bias.squeeze()
        return preds


class WideAndDeep(nn.Module):
    def __init__(self, n_users: int, n_movies: int, n_genres: int, emb_size: int = 50, hidden_size: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.movie_emb = nn.Embedding(n_movies, emb_size)
        self.genre_emb = nn.Embedding(n_genres, emb_size)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.fc1 = nn.Linear(emb_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor, genre_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(user_ids)
        movie_emb = self.movie_emb(movie_ids)
        genre_emb = self.genre_emb(genre_ids)
        movie_bias = self.movie_bias(movie_ids)
        wide = torch.cat([user_emb, movie_emb, genre_emb], dim=1)
        wide = torch.sum(wide, dim=1)
        deep = torch.cat([user_emb, movie_emb, genre_emb], dim=1)
        deep = self.fc1(deep)
        deep = nn.ReLU()(deep)
        preds = self.fc2(deep)
        preds = preds + movie_bias.squeeze()
        preds = wide + preds
        return preds


class MovieLensDataset(Dataset):
    def __init__(self, ratings_file: str, sep: str = '\t'):
        self.ratings = pd.read_csv(ratings_file, sep=sep, names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.user_enc = {u: i for i, u in enumerate(self.ratings['user_id'].unique())}
        self.movie_enc = {m: i for i, m in enumerate(self.ratings['movie_id'].unique())}
        self.ratings['user_id'] = self.ratings['user_id'].map(self.user_enc)
        self.ratings['movie_id'] = self.ratings['movie_id'].map(self.movie_enc)

    def __len__(self):
        return len(self.ratings)


