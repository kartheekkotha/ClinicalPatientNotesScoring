import sys
sys.path.append('../../')
from logHelper import logger
#Import the necessary modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval
from itertools import chain
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, RobertaTokenizerFast, RobertaModel
import nltk
from transformers import RobertaTokenizerFast
import time
from sklearn.model_selection import KFold
from transformers import DebertaModel, DebertaTokenizerFast


base_config = {
    "Base_data_path": "../../../data/nbme-score-clinical-patient-notes",
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "dropout": 0.2,
    "lr": 1e-5,
    "test_size": 0.2,
    "seed": 1268,
    "batch_size": 6,
    "model_name": "microsoft/deberta-base",
}

class prepare_data():
    def __init__(self , config):
        self.config = config
    def merge_data(self):
        features = pd.read_csv(f"{self.config['Base_data_path']}/features.csv")
        patient_notes = pd.read_csv(f"{self.config['Base_data_path']}/patient_notes.csv")
        train_df = pd.read_csv(f"{self.config['Base_data_path']}/train.csv")
        train_df['annotation_list'] = [literal_eval(x) for x in train_df['annotation']]
        train_df['location_list'] = [literal_eval(x) for x in train_df['location']]
        
        merged = train_df.merge(patient_notes, how='left')
        merged = merged.merge(features, how='left')

        merged['pn_history'] = merged['pn_history'].apply(lambda x: x.lower())
        merged['feature_text'] = merged['feature_text'].apply(lambda x: x.lower())
        merged['feature_text'] = merged['feature_text'].apply(lambda x: x.replace('-', ' ').replace('-OR-', ";-"))
        return merged
    
class pre_process_data():
    def __init__(self , config):
        self.config = config
    def split_loc(self , loc_list):
        final_loc = []
        for loc in loc_list:
            locations = loc.split(';')
            for location in locations:
                start , end = location.split()
                final_loc.append((int(start) , int(end)))
        return final_loc
    # so basically tokenizer divides the text so to assign whether the label belong to this or not we assign values accordingly
    def tokenize_and_addLabels(self , data , tokenizer , config):
        tokenized = tokenizer(
            data['feature_text'],
            data['pn_history'],
            truncation= config['truncation'],
            max_length= config['max_length'],
            padding= config['padding'],
            return_offsets_mapping=config['return_offsets_mapping']
        )
        labels = [0.0] * len(tokenized['input_ids'])
        tokenized['location_int'] = self.split_loc(data['location_list'])
        tokenized['sequence_ids'] = tokenized.sequence_ids()

        for idx, (seq_id, offsets) in enumerate(zip(tokenized["sequence_ids"], tokenized["offset_mapping"])):
            if not seq_id or seq_id == 0:
                labels[idx] = -1
                continue

            token_start , token_end = offsets
            for feature_start , feature_end in tokenized['location_int']:
                if token_start >= feature_start and token_end <= feature_end:
                    labels[idx] = 1.0
                    break
        
        tokenized['labels'] = labels
        return tokenized

class score_class():
    def __init__(self , config):
        self.config = config
    def get_location_predictions(self , preds , offset_mapping , sequence_ids , test = False):
        all_predictions = []
        for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
            pred = 1 / (1+ np.exp(-pred))
            start_idx = None
            end_idx = None
            current_preds = []
            for pred , offset , seq_id in zip(pred , offsets , seq_ids):
                if seq_id is None or seq_id == 0:
                    continue
                if pred >0.5:
                    if start_idx is None:
                        start_idx = offset[0]
                    end_idx = offset[1]
                elif start_idx is not None:
                    if test:
                        current_preds.append(f"{start_idx} {end_idx}")
                    else:
                        current_preds.append((start_idx, end_idx))
                    start_idx = None
            if test:
                all_predictions.append("; ".join(current_preds))
            else:
                all_predictions.append(current_preds)
            
        return all_predictions
    def calculate_char_cv(self , predictions, offset_mapping, sequence_ids, labels):
        all_labels = []
        all_preds = []
        for preds, offsets, seq_ids, labels in zip(predictions, offset_mapping, sequence_ids, labels):

            num_chars = max(list(chain(*offsets)))
            char_labels = np.zeros(num_chars)

            for o, s_id, label in zip(offsets, seq_ids, labels):
                if s_id is None or s_id == 0:
                    continue
                if int(label) == 1:
                    char_labels[o[0]:o[1]] = 1

            char_preds = np.zeros(num_chars)

            for start_idx, end_idx in preds:
                char_preds[start_idx:end_idx] = 1

            all_labels.extend(char_labels)
            all_preds.extend(char_preds)

        results = precision_recall_fscore_support(all_labels, all_preds, average="binary", labels=np.unique(all_preds))
        accuracy = accuracy_score(all_labels, all_preds)
        

        return {
            "Accuracy": accuracy,
            "precision": results[0],
            "recall": results[1],
            "f1": results[2]
        }
    
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        tokenizerObj = pre_process_data(self.config)
        tokens = tokenizerObj.tokenize_and_addLabels( data, self.tokenizer, self.config)

        input_ids = np.array(tokens["input_ids"])
        attention_mask = np.array(tokens["attention_mask"])
#         token_type_ids = __getitem__np.array(tokens["token_type_ids"])

        labels = np.array(tokens["labels"])
        offset_mapping = np.array(tokens['offset_mapping'])
        sequence_ids = np.array(tokens['sequence_ids']).astype("float16")
        
        return input_ids, attention_mask, labels, offset_mapping, sequence_ids
    
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained(config['model_name'])  # DeBERTa model
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc1(outputs[0])
        logits = self.fc2(self.dropout(logits))
        logits = self.fc3(self.dropout(logits)).squeeze(-1)
        return logits

def train_model(model , dataloader , optimizer , criterion):
    model.train()
    train_loss = []
    count = 0
    for batch in tqdm(dataloader):
        # print(f'Batch {count}')
        optimizer.zero_grad()
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        labels = batch[2].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = torch.masked_select(loss, labels > -1.0).mean()
        train_loss.append(loss.item() * input_ids.size(0))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logger.info(f"Training : batch {count} Loss: {loss.item()}")
        count+=1
    return sum(train_loss) / len(train_loss)

def eval_model(model, dataloader, criterion):
        model.eval()
        valid_loss = []
        preds = []
        offsets = []
        seq_ids = []
        valid_labels = []
        count = 0 
        for batch in tqdm(dataloader):
            # print(f'batch {count}')
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
#             token_type_ids = batch[2].to(DEVICE)
            labels = batch[2].to(DEVICE)
            offset_mapping = batch[3]
            sequence_ids = batch[4]

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = torch.masked_select(loss, labels > -1.0).mean()
            valid_loss.append(loss.item() * input_ids.size(0))

            preds.append(logits.detach().cpu().numpy())
            offsets.append(offset_mapping.numpy())
            seq_ids.append(sequence_ids.numpy())
            valid_labels.append(labels.detach().cpu().numpy())
            logger.info(f"Eval Epoch : batch {count} Loss: {loss.item()}")
            count+=1

        preds = np.concatenate(preds, axis=0)
        offsets = np.concatenate(offsets, axis=0)
        seq_ids = np.concatenate(seq_ids, axis=0)
        valid_labels = np.concatenate(valid_labels, axis=0)
        score_obj = score_class(base_config)
        location_preds = score_obj.get_location_predictions(preds, offsets, seq_ids, test=False)
        score = score_obj.calculate_char_cv(location_preds, offsets, seq_ids, valid_labels)
        logger.info(f"The score of the eval model is {score}")
        return sum(valid_loss)/len(valid_loss), score

if __name__ == '__main__' :
    obj = prepare_data(base_config)
    train_df = obj.merge_data()
    tokenizer = DebertaTokenizerFast.from_pretrained(base_config['model_name'])
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=base_config['seed'])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'The device is {DEVICE}')
    model = CustomModel(base_config).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
    optimizer = optim.AdamW(model.parameters(), lr=base_config['lr'])
    train_loss_data, valid_loss_data = [], []
    score_data_list = []
    valid_loss_min = np.Inf
    since = time.time()
    epochs = 3
    best_loss = np.inf
    fold_train_loss = []
    fold_valid_loss = []
    fold_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(train_df)):
        logger.info(f"Fold {fold + 1}/{k_folds}")
        print(f"Fold {fold + 1}/{k_folds}")
        X_train, X_test = train_df.iloc[train_index], train_df.iloc[test_index]

        training_data = CustomDataset(X_train, tokenizer, base_config)
        train_dataloader = DataLoader(training_data, batch_size=base_config['batch_size'], shuffle=True)

        testing_data = CustomDataset(X_test, tokenizer, base_config)
        test_dataloader = DataLoader(testing_data, batch_size=base_config['batch_size'], shuffle=False)

        best_loss = np.inf

        for epoch in range(epochs):
            logger.info(f"Fold {fold + 1}/{k_folds} , Epoch: {epoch + 1}/{epochs}")
            print("Epoch: {}/{}".format(epoch + 1, epochs))
            # Train model
            train_loss = train_model(model, train_dataloader, optimizer, criterion)
            train_loss_data.append(train_loss)
            print(f"Train loss: {train_loss}")
            fold_train_loss.append(train_loss)
            # Evaluate model
            valid_loss, score = eval_model(model, test_dataloader, criterion)
            valid_loss_data.append(valid_loss)
            fold_valid_loss.append(valid_loss)
            fold_scores.append(score)
            score_data_list.append(score)
            print(f"Valid loss: {valid_loss}")
            print(f"Valid score: {score}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), f"nbme_bert_fold{fold}_v2.pth")
            with open(f"fold_{fold}_data.txt", "w") as file:
                file.write(f"Train Loss: {fold_train_loss}\n")
                file.write(f"Valid Loss: {fold_valid_loss}\n")
                file.write(f"Scores: {fold_scores}\n")

            # Clear fold data lists for next iteration
            fold_train_loss.clear()
            fold_valid_loss.clear()
            fold_scores.clear()

    # After all folds are done, calculate average scores if needed
    avg_train_loss = sum(train_loss_data) / len(train_loss_data)
    avg_valid_loss = sum(valid_loss_data) / len(valid_loss_data)
    print(f"Average Train Loss: {avg_train_loss}")
    print(f"Average Valid Loss: {avg_valid_loss}")

    # Plotting
    pd.to_pickle(train_loss_data, "train_loss_data.pkl")
    pd.to_pickle(valid_loss_data, "valid_loss_data.pkl")
    plt.plot(train_loss_data, label="Training loss")
    plt.plot(valid_loss_data, label="Validation loss")
    plt.legend(frameon=False)
    plt.show()

    # Convert score data to dataframe if needed
    score_df = pd.DataFrame.from_dict(score_data_list)
    score_df.head()