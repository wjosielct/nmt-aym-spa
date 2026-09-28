# ==============================
# ./src/utils/reproducibility.py
# ==============================

import os
import random

import numpy as np
import torch


def set_seed(
        seed: int,
        deterministic: bool = True,
) -> None:
    """
    Sets random seeds for Python, NumPy, PyTorch, and CUDA.

    When deterministic=True, PyTorch is instructed to use
    deterministic algorithms whenever supported.

    Notes
    -----
    PYTHONHASHSEED is set here for subsequent/spawned processes.
    For strict control of Python hashing in the current interpreter,
    PYTHONHASHSEED must be defined before Python itself starts.
    """

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    # --------------------------------------------------------------------------
    # Environment variables
    # --------------------------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Required by deterministic CUDA matrix operations when deterministic
    # algorithms are enabled.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # --------------------------------------------------------------------------
    # Random number generators
    # --------------------------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------------------------
    # Deterministic execution
    # --------------------------------------------------------------------------

    if deterministic:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(
            True
        )

    else:

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        torch.use_deterministic_algorithms(
            False
        )


def seed_worker(
        worker_id: int,
) -> None:
    """
    Initializes a PyTorch DataLoader worker with deterministic
    NumPy and Python random seeds.

    PyTorch itself assigns each worker its own deterministic
    torch seed before this function is called.
    """

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )