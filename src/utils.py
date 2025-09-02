import pandas as pd


def transform_iou_data_to_dataframe(data):
    words, modes, iou_scores = [], [], []

    for key, values in data.items():
        word, mode = key.split('_')

        scores = values['iou_scores']

        for score in scores:
            words.append(word)
            modes.append(mode)
            iou_scores.append(score)

    df = pd.DataFrame({'word': words, 'mode': modes, 'iou_score': iou_scores})

    return df