import pandas as pd

def preprocessing_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    df.dropna(inplace=True)

    # Monetary = sum of spending columns
    df['monetary'] = (
        df.get('mntwines', 0).astype(float) +
        df.get('mntfruits', 0).astype(float) +
        df.get('mntmeatproducts', 0).astype(float) +
        df.get('mntfishproducts', 0).astype(float) +
        df.get('mntsweetproducts', 0).astype(float) +
        df.get('mntgoldprods', 0).astype(float)
    )

    # Normalize education names
    df['education'] = df['education'].replace({'2n Cycle': 'Master'})
    edu_map = {'Basic': 0, 'Graduation': 1, 'Master': 2, 'PhD': 3}
    df['education_ord'] = df['education'].map(edu_map)

    # Normalize marital
    df['marital_status'] = df['marital_status'].replace({
        'Widow': 'Divorced',
        'Alone': 'Single',
        'YOLO': 'Single',
        'Absurd': 'Single'
    })
    marital_map = {'Married': 0, 'Together': 1, 'Divorced': 2, 'Single': 3}
    df['marital_status_ord'] = df['marital_status'].map(marital_map)

    # Drop original columns
    drop_cols = [
        'mntwines','mntfruits','mntmeatproducts','mntfishproducts',
        'mntsweetproducts','mntgoldprods','education','marital_status'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df
