
# BERT MODEL
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer

def load_bert_model(num_classes):

    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_classes
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer