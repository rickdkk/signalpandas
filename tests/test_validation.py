import numpy as np
import pandas as pd

from signalpandas.validation import has_nulls


def test_check_nulls():
    # Numpy arrays
    test_array_clean = np.array([0, 1, 2, 3, 4, 5])
    test_array_nan = np.array([0, 1, np.NaN, 3, 4, 5])
    test_array_inf = np.array([0, 1, np.inf, 3, 4, 5])

    assert not has_nulls(test_array_clean)
    assert has_nulls(test_array_nan)
    assert has_nulls(test_array_inf)

    # Dataframes
    test_df_clean = pd.DataFrame(test_array_clean)
    test_df_nan = pd.DataFrame(test_array_nan)
    test_df_inf = pd.DataFrame(test_array_inf)

    assert not has_nulls(test_df_clean)
    assert has_nulls(test_df_nan)
    assert has_nulls(test_df_inf)

    # Multidimensional
    assert not has_nulls(np.vstack([test_array_clean, test_array_clean]))
    assert has_nulls(np.vstack([test_array_clean, test_array_nan]))
