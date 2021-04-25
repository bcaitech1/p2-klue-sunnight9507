import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == "blind":
            label.append(100)
        else:
            label.append(label_type[i])
    out_dataset = pd.DataFrame(
        {
            "sentence": dataset[1],
            "entity_01": dataset[2],
            "entity_02": dataset[5],
            "label": label,
        }
    )
    return out_dataset


def load_data(dataset_dir, truncation=False, token_highlighting_entity=False):
    # load label_type, classes
    with open("/opt/ml/input/data/label_type.pkl", "rb") as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter="\t", header=None)

    if token_highlighting_entity:
        dataset[1] = dataset.apply(add_token_highlighting_entity, axis=1)

    if truncation:
        dataset[1] = dataset.apply(data_truncation, axis=1)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset


def tokenized_dataset(dataset, tokenizer):
    concat_entity = []
    for e01, e02 in zip(dataset["entity_01"], dataset["entity_02"]):
        temp = ""
        temp = e01 + ", " + e02 + "의 관계는?"
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200,
        add_special_tokens=True,
    )

    return tokenized_sentences


def train_valid_split(train_dataset):
    """
    training, validation dataset split

    Args:
        train_dataset ([pandas dataframe]): train dataset

    Returns:
        [pandas dataframe, pandas dataframe]: train dataset, validation dataset
    """
    # label이 40인 data는 하나이므로 split 불가
    temp_train_dataset = train_dataset[train_dataset["label"] != 40]
    data = temp_train_dataset[["sentence", "entity_01", "entity_02"]]
    target = temp_train_dataset["label"]

    # target 분포 기준으로 split
    x_train, x_valid, y_train, y_valid = train_test_split(
        data, target, test_size=0.1, stratify=target
    )

    train_dataset = pd.concat(
        [
            pd.concat([x_train, y_train], axis=1),
            train_dataset[train_dataset["label"] == 40],
        ]
    )
    valid_dataset = pd.concat([x_valid, y_valid], axis=1)

    print("train_dataset shape: ", train_dataset.shape)
    print("valid_dataset shape: ", valid_dataset.shape)

    return train_dataset, valid_dataset


def data_truncation(data):
    """
    entity1, 2 기준으로 양 옆 50만 check
    1. ent1, ent2 양 옆이 겹치지 않는 경우
        - [~ ent1 ~] + "</s>" + [~ ent2 ~]

    2. ent1, ent2 양 옆이 겹치는 경우
        - [~ ent1 ~ ent2 ~]

    ent1, ent2 자리 바뀌여도 적용 가능

    Args:
        data (pandas dataframe): [description]

    Returns:
        [pandas dataframe]: ent1, 2 기준으로 양 옆 50이 잘린 sentence
    """
    padding_length = 50

    entity_min_index = min(data[3], data[6])
    entity_max_index = max(data[4], data[7])

    min_entity_start, min_entity_end = (
        entity_min_index - padding_length,
        entity_min_index + padding_length,
    )
    max_entity_start, max_entity_end = (
        entity_max_index - padding_length,
        entity_max_index + padding_length,
    )

    if min_entity_end < max_entity_start:
        min_entity_start = max(min_entity_start, 0)
        max_entity_end = min(max_entity_end, len(data[1]))
        return (
            data[1][min_entity_start:min_entity_end]
            + " </s> "
            + data[1][max_entity_start:max_entity_end]
        )
    else:
        min_entity_start = max(min_entity_start, 0)
        max_entity_end = min(max_entity_end, len(data[1]))
        return data[1][min_entity_start:max_entity_end]


def add_token_highlighting_entity(data):
    """
    sentence에 있는 ent1, ent2를 강조하는 함수

    Args:
        data ([pandas dataframe]): [description]

    Returns:
        [pandas dataframe]: ent1, ent2에 <e1>, <e2>로 강조된 sentence
    """
    # entity1이 entity2보다 앞에 나온 경우
    if data[3] < data[6]:
        return (
            data[1][: data[3]]
            + "<e1>"
            + data[2]
            + "</e1>"
            + data[1][data[4] + 1 : data[6]]
            + "<e2>"
            + data[5]
            + "</e2>"
            + data[1][data[6] + 1 :]
        )
    # entity2가 entity1보다 앞에 나온 경우
    else:
        return (
            data[1][: data[6]]
            + "<e2>"
            + data[5]
            + "</e2>"
            + data[1][data[7] + 1 : data[3]]
            + "<e1>"
            + data[2]
            + "</e1>"
            + data[1][data[4] + 1 :]
        )
