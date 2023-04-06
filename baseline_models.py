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

class MovieLensDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        assert split in ['train', 'val', 'test'], "Invalid split. Must be one of ['train', 'val', 'test']"
        self.split = split
        self.transform = transform
        
        # Load data
        if split == 'train':
            ratings_file = os.path.join(data_dir, 'train_ratings.csv')
        elif split == 'val':
            ratings_file = os.path.join(data_dir, 'val_ratings.csv')
        else:
            ratings_file = os.path.join(data_dir, 'test_ratings.csv')
            
        self.ratings = pd.read_csv(ratings_file, usecols=['userId', 'movieId', 'rating'])
        self.num_users = self.ratings['userId'].nunique()
        self.num_items = self.ratings['movieId'].nunique()
        
        # Map user and item IDs to contiguous indices
        self.user_mapping = dict(zip(self.ratings['userId'].unique(), range(self.num_users)))
        self.item_mapping = dict(zip(self.ratings['movieId'].unique(), range(self.num_items)))
        self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.user_mapping[x])
        self.ratings['movieId'] = self.ratings['movieId'].apply(lambda x: self.item_mapping[x])
        
        # Convert dataframe to tensor
        self.ratings = torch.tensor(self.ratings.to_numpy(), dtype=torch.long)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        user_id = self.ratings[index, 0]
        item_id = self.ratings[index, 1]
        rating = self.ratings[index, 2]
        
        if self.transform:
            user_id, item_id, rating = self.transform(user_id, item_id, rating)
        
        return user_id, item_id, rating
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





