from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def load_bert_model(num_classes):

    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_classes
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model.compile(
        optimizer=Adam(learning_rate=3e-5),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model, tokenizer