import os
import sys

from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch


print("Importation du modèle NLP RoBERTa Classifieur pour identifier l'action")

classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")

print("Importation du modèle NLP RoBERTa Génératif pour l'identification des cibles")
nlp = pipeline('question-answering',
               model="deepset/roberta-base-squad2",
               tokenizer="deepset/roberta-base-squad2")




def findAction(sentence,candidate_labels = ["Take", "Drop", "Perform an action"]):
    result=classifier(sentence, candidate_labels)
    action=result['labels']
    score=result['scores']
    index=action.index("Perform an action")
    action.remove("Perform an action")
    score.pop(index)
    return action,score


def findTarget(text,action):
    if action=="Take":
        question="What should be taken?"
    else:
        question="Where should we drop the object?"
    QA_input = {
        'question': question,
        'context': text
    }
    res = nlp(QA_input)
    return res['answer']
