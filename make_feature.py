from omegaconf import OmegaConf
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib


def make_features(config):
    print('Make_Features...')
    train_df = pd.read_csv(config.data.train_csv_safe_path)
    test_df = pd.read_csv(config.data.test_csv_safe_path)

    vectorizer_name = config.features.vectorizer
    vectorizer = {
        'count-vectorizer': CountVectorizer,
        "tfidf-vectorizer": TfidfVectorizer,
    }[vectorizer_name](stop_words='english')

    train_input = vectorizer.fit_transform(train_df['review'])
    test_input = vectorizer.transform(test_df['review'])

    joblib.dump(train_input, config.features.train_input_safe_path)
    joblib.dump(test_input, config.features.test_input_safe_path)


if __name__ == '__main__':
    config = OmegaConf.load("params.yaml")
    make_features(config)
