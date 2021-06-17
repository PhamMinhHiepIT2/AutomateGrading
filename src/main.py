import cv2
import numpy as np
import argparse
from collections import defaultdict


from src.model import CNN_Model
from src.preprocessor import (crop_image,
                              process_ans_blocks,
                              process_list_ans,
                              )


def map_answer(index):
    num = index % 4
    if num == 0:
        return "A"
    elif num == 1:
        return "B"
    elif num == 2:
        return "C"
    else:
        return "D"


def predict(input_path: str):
    img = cv2.imread(input_path)
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)

    results = defaultdict(list)
    model = CNN_Model('model/weight.h5').setup_CNN(rt=True)
    answers = np.array(list_ans)
    scores = model.predict_on_batch(answers / 255.0)
    for index, score in enumerate(scores):
        question = index // 4

        # score [unchoice_confidence, choice_confidence]
        if score[1] > 0.9:
            chosed_answer = map_answer(index)
            results[question + 1].append(chosed_answer)
    return results


def add_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to dataset", required=True)
    parser.add_argument("--mode", default=["train", "infer"], required=True)
    args = parser.parse_args()
    return args


def main():
    input_path = "1.jpg"
    args = add_argument()
    data_path = args.data_path
    mode = args.mode
    if mode == "train":
        CNN_Model().train(data_path)
    elif mode == "infer":
        result = predict(input_path)
        print(result)


if __name__ == "__main__":
    main()
