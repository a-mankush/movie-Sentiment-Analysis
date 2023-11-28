from omegaconf import OmegaConf
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


def evaluate(config):
    print("Evaluating...")
    test_input = joblib.load(config.features.test_input_safe_path)
    test_output = pd.read_csv(config.data.test_csv_safe_path)['label'].values

    model = joblib.load(config.train.model_path)

    predictions = model.predict(test_input)

    matrix_name = config.evaluate.matrix
    matrix = {
        "accuracy": accuracy_score,
        "f1_score": f1_score,
    }[matrix_name]

    result = matrix(test_output, predictions)
    result_dict = {matrix_name: float(result)}
    OmegaConf.save(result_dict, config.evaluate.result_save_path)


if __name__ == '__main__':
    config = OmegaConf.load("params.yaml")
    evaluate(config)
