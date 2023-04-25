from transformers import pipeline

from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch



classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')




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
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    return tokenizer.decode(predict_answer_tokens)
