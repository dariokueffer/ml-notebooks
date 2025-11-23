from torch.utils.data import Dataset
from lib.node_tokenizer import NodeTokenizer
import torch


class ShortestPathDataset(Dataset):
    def __init__(self, instances, max_len=16, include_target_and_source=False):
        self.instances = instances

        self.num_nodes = self.get_max_node_count()
        self.max_len = max_len + 3

        self.tokenizer = NodeTokenizer(self.num_nodes)
        self.vocab_size = self.tokenizer.vocab_size()
        self.super_node = self.num_nodes + self.tokenizer.offset

        self.include_target_and_source = include_target_and_source

    def get_max_node_count(self):
        return max(instance["graph"].num_nodes for instance in self.instances)

    def special_tokens(self):
        self.bos_token = 0  # Beginning of sequence token
        self.pad_token = 1  # Padding token
        self.eos_token = 2  # End of sequence token
        self.dist_token = 3  # Feature token for distance
        self.dur_token = 4  # Feature token for duration
        self.car_duration_token = 5  # Feature token for Car Duration
        self.car_distance_token = 6  # Feature token for Car Distance
        self.bike_duration_token = 7  # Feature token for Bike Duration
        self.bike_distance_token = 8  # Feature token for Bike Distance
        self.special_tokens_offset = 9  # Offset for special tokens in the vocabulary

    # add eos and pad tokens to the path
    def create_target_path(self, path, criterion="distance"):
        if path is None:
            token_ids = self.tokenizer.add_special_tokens(
                [], max_len=self.max_len, criterion=criterion
            )
            return torch.tensor(
                token_ids, dtype=torch.long
            )  # Maybe remove whole instance instead of dummy path?
        if self.include_target_and_source:
            path = list(path)
        else:
            path = path[1:-1]
        token_ids = self.tokenizer.add_special_tokens(
            path, max_len=self.max_len, criterion=criterion
        )
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]

        return {
            "graph": instance["graph"],  # PyG Data
            "source": torch.tensor(instance["source"], dtype=torch.long),
            "target": torch.tensor(instance["target"], dtype=torch.long),
            "distance_path": self.create_target_path(
                instance["distance_path"], "distance"
            ),
            "adjacency_matrix": instance["adjacency_matrix"],
            "distance_matrix": instance["distance_matrix"],
        }

    def get_decoded_path(self, path):
        return self.tokenizer.decode(path.tolist())
