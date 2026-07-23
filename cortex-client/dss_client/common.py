from enum import Enum


class JobType(str, Enum):
    TRAINING = "training"
    SAMPLING = "sampling"
    LOG_PROB = "log_prob"