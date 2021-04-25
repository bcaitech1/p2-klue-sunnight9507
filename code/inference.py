from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from transformers import (
    ElectraModel,
    ElectraTokenizer,
    ElectraConfig,
    ElectraForSequenceClassification,
)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device):
  """[summary]

  Args:
      model : transformers model
      tokenized_sent : tokenized sentence
      device : device

  Returns:
      [type]: [description]
  """
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir, truncation=True, token_highlighting_entity=False)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
  test

  Args:
      args ([type]): [description]
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = "xlm-roberta-large" #"monologg/koelectra-base-v3-discriminator"
  tokenizer = XLMRobertaTokenizer.from_pretrained(TOK_NAME)

  # load my model
  MODEL_NAME = args.model_dir # model dir.
  model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device)
  
  # make csv file with predicted answer
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv('/opt/ml/code/prediction/submission_' + args.model_dir.split("-")[-1] + '.csv', index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/checkpoint-10000")
  args = parser.parse_args()

  # print(args)
  main(args)
  
