import random

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "True"
# Define your train examples
#
train_size = 0.8
f = open("train_trans.tsv", "r+", encoding="utf-8")
data = f.readlines()
train_data = []
random.shuffle(data)
sentences1 = []
sentences2 = []
scores = []
for i,line in enumerate(data):
  line = line.strip("\n")
  line = line.split("\t")
  if i<len(data)*train_size:
    train_data.append(InputExample(texts=[line[0], line[1]], label=float(line[2])))
  else:
    sentences1.append(line[0])
    sentences2.append(line[1])
    scores.append(float(line[2]))
model = SentenceTransformer('bert-base-chinese')
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=50,
          evaluation_steps=500,
          output_path="oppo.model")
model.save("opp.pin")