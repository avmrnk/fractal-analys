import pandas as pd


def load_time_series(file_path, time_col='relative_time', value_col='value'):
    """
    Зчитує CSV або Excel з двома колонками: time_col і value_col.
    Повертає pd.Series, де індекс – це time_col, а значення – value_col.
    """
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Файл повинен містити колонки '{time_col}' та '{value_col}'")

    ts = pd.Series(df[value_col].values, index=df[time_col].values)
    ts.index.name = time_col
    ts.name = value_col
    return ts
