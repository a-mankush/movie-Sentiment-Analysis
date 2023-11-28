from omegaconf import OmegaConf
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression


def train(config):
    print("Traning...")

    train_input = joblib.load(config.features.train_input_safe_path)
    train_output = pd.read_csv(config.data.train_csv_safe_path)['label'].values

    penalty = config.train.penalty
    C = config.train.C
    solver = config.train.solver
    max_iter = config.train.max_iter

    model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    model.fit(train_input, train_output)

    joblib.dump(model, config.train.model_path)


if __name__ == '__main__':
    config = OmegaConf.load("params.yaml")
    train(config)
