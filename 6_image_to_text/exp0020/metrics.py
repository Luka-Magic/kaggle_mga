import re
from typing import List, Dict, Union, Tuple, Any
import numpy as np
import pandas as pd
from polyleven import levenshtein

BOS_TOKEN = "<|BOS|>"
START = "<|start|>"
END = "<|end|>"

# LINE_TOKEN = "<line>"
# VERTICAL_BAR_TOKEN = "<vertical_bar>"
# HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
# SCATTER_TOKEN = "<scatter>"
# DOT_TOKEN = "<dot>"

# CHART_TYPE_TOKENS = [
#     LINE_TOKEN,
#     VERTICAL_BAR_TOKEN,
#     HORIZONTAL_BAR_TOKEN,
#     SCATTER_TOKEN,
#     DOT_TOKEN,
# ]


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function for the given value.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
        x -> 0: sig(x) -> 1
        x -> +∞: sig(x) -> 0
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the normalized Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The normalized Root Mean Square Error.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0  # 正解が1つ & 正解したら
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate the normalized Levenshtein distance between two lists of strings.

    Args:
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.

    Returns:
        float: The normalized Levenshtein distance.
    """
    total_distance = np.sum([levenshtein(yt, yp)
                            for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(
    y_true: List[Union[float, str]], y_pred: List[Union[float, str]]
) -> float:
    """
    Calculate the score for a series of true and predicted values.

    Args:
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.

    Returns:
        float: The score for the series.
    """
    notna_y_true = []
    notna_y_pred = []
    for i, y in enumerate(y_true):
        if isinstance(y, float) and np.isnan(y):
            continue
        notna_y_true.append(y_true[i])
        notna_y_pred.append(y_pred[i])

    y_true = notna_y_true.copy()
    y_pred = notna_y_pred.copy()
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(
            f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(
            index=False), predictions.itertuples(index=False)
    )
    scores = []
    n_chart_type = 0
    n_point_correct = 0
    chart_type_correct = 0
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        n_chart_type += 1
        if gt_type != pred_type:  # Check chart_type condition
            score = 0.0
        elif len(gt_series) != len(pred_series):
            chart_type_correct += 1
            score = 0.0  # Score with RMSE or Levenshtein as appropriate
        else:
            chart_type_correct += 1
            n_point_correct += 1
            score = score_series(gt_series, pred_series)
        scores.append(score)

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)[
        "score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    chart_type_acc = chart_type_correct / n_chart_type
    n_point_acc = n_point_correct / n_chart_type

    return np.mean(scores), chart_type2score, scores, chart_type_acc, n_point_acc


def string2triplet(pred_string: str) -> Tuple[str, List[str], List[str]]:
    """
    Convert a prediction string to a triplet of chart type, x values, and y values.

    Args:
        pred_string (str): The prediction string.

    Returns:
        Tuple[str, List[str], List[str]]: A triplet of chart type, x values, and y values.
    """

    chart_type = "scatter"
    # for tok in CHART_TYPE_TOKENS:
    #     if tok in pred_string:
    #         chart_type = tok.strip("<>")

    pred_string = re.sub(r"<one>", "1", pred_string)

    # x = pred_string.split(X_START)[1].split(X_END)[0].split(";")
    # y = pred_string.split(Y_START)[1].split(Y_END)[0].split(";")

    x = []
    y = []
    data_series = pred_string.split(START)[1].split(END)[0].split(";")
    for data in data_series:
        if '|' in data:
            data_split = data.split('|')
            if len(data_split) >= 2:
                x.append(data_split[0])
                y.append(data_split[1])
            else:
                print(f'\n    check data: {data}')

    if len(x) == 0 or len(y) == 0:
        return chart_type, [], []

    # min_length = min(len(x), len(y))

    # x = x[:min_length]
    # y = y[:min_length]

    return chart_type, x, y


def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate validation metrics for a set of outputs, ids, and ground truth dataframe.

    Args:
        val_outputs (List[str]): A list of validation outputs.
            str: <|BOS|><line><start> ... <end></s>
        val_ids (List[str]): A list of validation ids.
            str: index(0 ~ length_of_dataset)
        gt_df (pd.DataFrame): The ground truth dataframe.
            index: id_x or id_y
            data_series: v1;v2;...vn
            chart_type: select from 5 chart types.

    Returns:
        Dict[str, float]: A dictionary containing the validation scores.
    """
    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [START, END]]):
            pred_triplets.append(("scatter", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))
            # <?_start> ~ <?_end>までをx, yそれぞれ切り取り;でsplitしてlist化, 短い方の長さに揃える。
            # Tuple(chart_type, x, y)

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] +
        [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        }
    )

    overall_score, chart_type2score, scores, chart_type_acc, n_point_acc = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )
    # scores: pred_dfの順に並ぶ
    print(len(val_ids))

    pred_list_for_table = []
    for (id_, (chart_type, x, y), score) in zip(val_ids, pred_triplets, scores):
        pred_list_for_table.append(
            {'id': id_, 'x': x, 'y': y, 'chart_type': chart_type, 'score': score})

    return {
        "valid_score": overall_score,
        "chart_acc": chart_type_acc,
        'n_point_acc': n_point_acc,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }, pred_list_for_table
