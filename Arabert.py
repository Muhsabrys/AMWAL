import json
import os
from datasets import Dataset, concatenate_datasets

def load_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for i in range(len(json_data['text'])):
                    text = json_data['text'][i]
                    entities = json_data['entities']
                    data.append({"text": text, "entities": entities})
    return data

data = load_data('/geode2/home/u060/mabdo/Quartz/NERWITHLLMS/Data')



# Define your entity type to ID mapping based on your label map
label_map = {
    'BANK': 0,
    'ORG': 1,
    'PERSON': 2,
    'OFFICIAL': 3,
    'NATIONALITY': 4,
    'COUNTRY': 5,
    'MEDIA': 6,
    'FINANCIAL_INSTRUMENT': 7,
    'TIME': 8,
    'QUNATITY_OR_UNIT': 9,
    'GOVERNMENT_ENTITY': 10,
    'CORP': 11,
    'PRODUCT_OR_SERVICE': 12,
    'STOCK_EXCHANGE': 13,
    'CURRENCY': 14,
    'ROLE': 15,
    'GPE': 16,
    'CITY': 17,
    'FinMarket': 18,
    'Metrics': 19,
    'Events': 20,
}

# Optionally, create a reverse mapping for clarity
reverse_label_map = {v: k for k, v in label_map.items()}


from transformers import AutoTokenizer
from datasets import Dataset

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

def tokenize_and_align_labels(examples):
    # Tokenize the input text, maintaining offsets
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding=True, return_offsets_mapping=True)
    
    labels = []
    
    for i, entities in enumerate(examples['entities']):
        label_ids = [-100] * len(tokenized_inputs['input_ids'][i])  # Initialize with -100 for ignored labels
        
        for entity in entities:
            start, end = entity['start'], entity['end']
            word_ids = tokenized_inputs['offset_mapping'][i]  # Get offsets for this example
            
            for j, (start_offset, end_offset) in enumerate(word_ids):
                if start_offset >= start and end_offset <= end:
                    # Assign the entity ID based on the mapping
                    label_ids[j] = label_map[entity['type']]  # Convert entity type to ID
                elif start_offset < end and end_offset > start:
                    # Handle overlaps
                    label_ids[j] = label_map[entity['type']]  # Set ID if it overlaps

        labels.append(label_ids)

    # Add labels to tokenized inputs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Create dataset and map the function
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification

# Load the model for token classification
model = AutoModelForTokenClassification.from_pretrained("aubmindlab/bert-base-arabertv02", num_labels=len(label_map))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',              # Output directory
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch
    learning_rate=2e-5,                  # Learning rate
    per_device_train_batch_size=16,      # Batch size per device during training
    per_device_eval_batch_size=64,       # Batch size for evaluation
    num_train_epochs=3,                  # Total number of training epochs
    weight_decay=0.01,                   # Strength of weight decay
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Ideally, you should use a separate validation dataset
)


trainer.train()


trainer.evaluate()


model.save_pretrained('/geode2/home/u060/mabdo/Quartz/NERWITHLLMS/Data/model')
tokenizer.save_pretrained('/geode2/home/u060/mabdo/Quartz/NERWITHLLMS/Data/model')