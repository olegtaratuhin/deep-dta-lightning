import numpy as np
import pandas as pd
import torch


def prepare_result(results: list[torch.Tensor]) -> pd.DataFrame:
    results_np = np.vstack([batch_prediction.numpy() for batch_prediction in results])
    return pd.DataFrame(results_np, columns=["Y_pred"])
