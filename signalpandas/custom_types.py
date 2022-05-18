from typing import Union, TypeAlias

import numpy as np
import pandas as pd

Signal: TypeAlias = Union[pd.DataFrame, pd.Series, np.ndarray]
Pandas: TypeAlias = Union[pd.DataFrame, pd.Series]
