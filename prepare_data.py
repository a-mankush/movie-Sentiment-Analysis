from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(config):
    print("Preparing data...")
    df = pd.read_csv(config.data.dataset_file_path)
    df['label'] = pd.factorize(df['sentiment'])[0]
    # print(df.head())

    test_size = config.data.test_split
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['sentiment'], random_state=123)

    train_df.to_csv(config.data.train_csv_safe_path)
    test_df.to_csv(config.data.test_csv_safe_path)


if __name__ == '__main__':
    config = OmegaConf.load("params.yaml")
    prepare_data(config)
