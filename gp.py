import numpy as np

#%% PREPROCESSING

def preprocess(data):
    preprocessed_data = []
    for hand in data:
        ranks = np.zeros((5, 13))
        suits = np.zeros((5, 4))
        for i, card in enumerate(hand):
            rank_index = card % 13
            suit_index = card // 13
            ranks[i, rank_index] = 1
            suits[i, suit_index] = 1
        
    
        features = np.concatenate((ranks, suits), axis=-1)
        

        sorted_features = np.sort(features, axis=0)
        rank_diffs = np.diff(sorted_features[:, :13], axis=0)
        sorted_rank = np.sort(rank_diffs)
   
        flattened_features = np.concatenate((sorted_features.flatten(), sorted_rank.flatten()))
        
        preprocessed_data.append(flattened_features)
    
    return np.array(preprocessed_data)


