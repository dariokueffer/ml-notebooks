class NodeTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.special_tokens = {
            '<BOS>': self.num_nodes + 0,  # Beginning of sequence token
            '<PAD>': self.num_nodes + 1,  # Padding token
            '<EOS>': self.num_nodes + 2,  # End of sequence token
            '<DIST>': self.num_nodes + 3,  # Padding token for Distance
            '<DUR>': self.num_nodes + 4,  # Padding token for Duration
            '<CAR_DIST>': self.num_nodes + 5, # Padding token for Car Distance
            '<CAR_DUR>': self.num_nodes + 6, # Padding token for Car Duration
            '<BIKE_DIST>': self.num_nodes + 7, # Padding token for Bike Distance
            '<BIKE_DUR>': self.num_nodes + 8, # Padding token for Bike Duration
        }
        self.offset = len(self.special_tokens)

    def encode(self, path):
        if path is None:
            return []
        return path
    
    def add_special_tokens(self, path, max_len=16, criterion='distance'):
         # Map criterion to the correct token
        if criterion == 'distance':
            feature_token = self.special_tokens['<DIST>']
        elif criterion == 'duration':
            feature_token = self.special_tokens['<DUR>']
        elif criterion == 'car_distance':
            feature_token = self.special_tokens['<CAR_DIST>']
        elif criterion == 'car_duration':
            feature_token = self.special_tokens['<CAR_DUR>']
        elif criterion == 'bike_distance':
            feature_token = self.special_tokens['<BIKE_DIST>']
        elif criterion == 'bike_duration':
            feature_token = self.special_tokens['<BIKE_DUR>']
        else:
            raise NotImplementedError("Supported: 'distance', 'duration', 'car_distance', 'car_duration', 'bike_distance', 'bike_duration'")
        encoded_path = [feature_token] + [self.special_tokens['<BOS>']] + self.encode(path) + [self.special_tokens['<EOS>']]
        if len(encoded_path) < max_len:
            encoded_path += [self.special_tokens['<PAD>']] * (max_len - len(encoded_path))
        else:
            encoded_path = encoded_path[:max_len]
        return encoded_path
    
    def vocab_size(self):
        return self.offset + self.num_nodes
    
    def decode(self, encoded_path):
        id_to_special = {v: k for k, v in self.special_tokens.items()}
        
        decoded_path = []
        for token_id in encoded_path:
            if token_id in id_to_special:
                decoded_path.append(id_to_special[token_id])
            elif 0 <= token_id < self.num_nodes:
                decoded_path.append(token_id)
            else:
                decoded_path.append('<UNK>')

        return decoded_path
