import casanova
import random
import sys
from itertools import product

random.seed(42)

HPARAM_LEARNING_RATE = [2e-5, 3e-5]
HPARAM_MARGIN_LOSS_SIAMESE = [0.1, 0.3, 0.5]
HPARAM_MARGIN_LOSS_TRIPLET = [0.2, 0.3, 0.4]
HPAREM_DATA_FILTER_PAIR = [0.4]
HPAREM_DATA_FILTER_TRIPLET = [0.4]
HPARAM_LORA_R = [8]

HPARAM_HEADER = [
    "trial",
    "lr",
    "margin_loss_siamese",
    "margin_loss_triplet",
    "data_filter_pair",
    "data_filter_triplet",
    "lora_r"
]

if __name__ == "__main__":
    with casanova.writer(sys.stdout, HPARAM_HEADER) as writer:
        items = list(product(
            HPARAM_LEARNING_RATE,
            HPARAM_MARGIN_LOSS_SIAMESE,
            HPARAM_MARGIN_LOSS_TRIPLET,
            HPAREM_DATA_FILTER_PAIR,
            HPAREM_DATA_FILTER_TRIPLET,
            HPARAM_LORA_R
        ))

        random.shuffle(items)

        for i, params in enumerate(items):
            writer.writerow([i] + list(params))
        
