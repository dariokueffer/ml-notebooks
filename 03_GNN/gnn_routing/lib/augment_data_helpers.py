import torch


def reverse_path_instance(
    instance,
    pad_idx,
    eos_idx,
    bos_idx,
    dist_idx,
    dur_idx,
    car_dist_idx,
    bike_dist_idx,
    car_dur_idx,
    bike_dur_idx,
    debug=False,
):
    new_distance_path = torch.tensor([], dtype=torch.long)

    for feature in [
        "distance",
    ]:
        path = instance[f"{feature}_path"]
        if not (path is None or len(path) == 0):
            path_without_special_tokens = path[
                (path != pad_idx)
                & (path != bos_idx)
                & (path != eos_idx)
                & (path != dist_idx)
                & (path != dur_idx)
                & (path != car_dist_idx)
                & (path != bike_dist_idx)
                & (path != car_dur_idx)
                & (path != bike_dur_idx)
            ]
            reversed_nodes = path_without_special_tokens.flip(0)
            if not torch.equal(reversed_nodes.flip(0), path_without_special_tokens):
                raise ValueError(
                    "Reversed nodes do not match original path without special tokens."
                )
            new_path = torch.full_like(path, pad_idx)  # start with all PAD
            if debug:
                print(f"{feature}: Path: {path}")
            new_path[1] = bos_idx
            new_path[2 : 2 + len(reversed_nodes)] = reversed_nodes
            new_path[len(reversed_nodes) + 2] = eos_idx
            if feature == "distance":
                new_distance_path = new_path
                new_distance_path[0] = dist_idx
                if debug:
                    print(f"{feature}: Reversed Path: {new_distance_path}\n")

        reversed_instance = {
            "graph": instance["graph"],
            "source": instance["target"],
            "target": instance["source"],
            "distance_path": new_distance_path,
            "adjacency_matrix": instance["adjacency_matrix"],
            "distance_matrix": instance["distance_matrix"],
        }
    return reversed_instance


def augment_with_reversed(dataset, include_target_and_source=False, debug=False):
    augmented = []
    pad_idx = dataset.tokenizer.special_tokens["<PAD>"]
    eos_idx = dataset.tokenizer.special_tokens["<EOS>"]
    bos_idx = dataset.tokenizer.special_tokens["<BOS>"]
    dist_idx = dataset.tokenizer.special_tokens["<DIST>"]
    dur_idx = dataset.tokenizer.special_tokens["<DUR>"]
    car_dist_idx = dataset.tokenizer.special_tokens["<CAR_DIST>"]
    car_dur_idx = dataset.tokenizer.special_tokens["<CAR_DUR>"]
    bike_dist_idx = dataset.tokenizer.special_tokens["<BIKE_DIST>"]
    bike_dur_idx = dataset.tokenizer.special_tokens["<BIKE_DUR>"]
    for instance in dataset:
        augmented.append(instance)
        augmented.append(
            reverse_path_instance(
                instance,
                pad_idx,
                eos_idx,
                bos_idx,
                dist_idx,
                dur_idx,
                car_dist_idx,
                bike_dist_idx,
                car_dur_idx,
                bike_dur_idx,
                debug=debug,
            )
        )
    return augmented
