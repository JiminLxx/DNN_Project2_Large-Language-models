DNN_Project2_Large-Language-models
===
The task is to generate a set of movie reviews using the pre trained GPT 2 model and then classify their sentiment (positive or negative) using a fine tuned BERT model.

# 0. Install the necessary libraries
Install HuggingFace Transformers, TensorFlow, and PyTorch, which are essential for this homework.
```
!pip install wget evaluate
```

# 1. Generate movie reviews
### a. Load the pre trained GPT 2 model ("heegyu/gpt2-emotion") from the HuggingFace and a tokenizer from the Hugging Face Transformers library like tokenizer = GPT2Tokenizer.from_pretrained('heegyu/gpt2 emotion')
You should use the Dataloader of Pytorch (i.e., torch.utils.data.DataLoader) when loading your data. Please use a movie review dataset from the link below and randomly divide the data into these proportions: Training, Validation, and Testing.
> Dataset Link: https://ai.stanford.edu/~amaas/data/sentiment/. Please follow the following procedure:
> ![image](https://github.com/JiminLxx/DNN_Project2_Large-Language-models/assets/166349621/62f9a098-53d1-4708-b401-16eca85195a3)

First, we have to import libraries for the code implementation.
The device configuration makes sure to connect code and GPU(NVIDIA L4) of colab.
```
# 1. Import libraries (Set up your environment with the necessary packages)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import tarfile
import random
import os

# Device configuration
if torch.cuda.is_available(): # Check GPU availability
    # Create a GPU Device Object
    device = torch.device("cuda:0")
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu") # Creating a CPU Device Object
    print("Running on CPU")
```


Now, let's download the moview review dataset using the link directly.
It checks whether `dataset_path` file or `extracted_path`file are already on your directory and download or extract it if there are not exist.

Then, load the GPT-2 model and tokenizer as well.
```
# 2. Download the movie review dataset
dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path = "aclImdb_v1.tar.gz"
extracted_path = "aclImdb"

if not os.path.exists(dataset_path):
    import wget
    wget.download(dataset_url, dataset_path)

if not os.path.exists(extracted_path):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall()

# 2-1. Load the GPT-2 model and tokenizer (Initialize for the first time)
model_name = "heegyu/gpt2-emotion"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```


Because we have to make sure that train and validation dataset have equal ratio of positive and negative dataset, I splited it into 8:2 respectively and merge into a one dataset; `train_dataset` and `val_dataset`.

Using class MovieReviewDataset, I make instances of tokenizer, reivews, labels, and filenames. I additionally added filename for convenience to compare ground truth data and generated data.
```
# 3. Load and process the movie review dataset
class MovieReviewDataset(Dataset):
    def __init__(self, directory, tokenizer, label):
        self.tokenizer = tokenizer
        self.reviews = []
        self.labels = []
        self.filenames = []  # list to save file name

        for file in os.listdir(directory):
            if file.endswith('.txt'):
                file_path = os.path.join(directory, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    review = f.read()
                    self.reviews.append(review)
                    self.labels.append(label)
                    self.filenames.append(file)  # save a file name

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        filename = self.filenames[idx]  # get a file name
        inputs = self.tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['filename'] = filename  # add a file name into a dictionary
        return inputs

# 4-1. Initialize training dataset (pos: 1, neg: 0)
train_pos_dataset = MovieReviewDataset(os.path.join(extracted_path, 'train', 'pos'), tokenizer, 1)
train_neg_dataset = MovieReviewDataset(os.path.join(extracted_path, 'train', 'neg'), tokenizer, 0)

# Split the data into training and validation sets (train:val = 8:2)
def split_dataset(dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

train_pos_dataset, val_pos_dataset = split_dataset(train_pos_dataset)
train_neg_dataset, val_neg_dataset = split_dataset(train_neg_dataset)

# Merge the dataset respectively
train_dataset = torch.utils.data.ConcatDataset([train_pos_dataset, train_neg_dataset])
val_dataset = torch.utils.data.ConcatDataset([val_pos_dataset, val_neg_dataset])

# 4-2. Initialize test dataset (pos: 1, neg: 0)
test_dataset = MovieReviewDataset(os.path.join(extracted_path, 'test', 'pos'), tokenizer, 1) + \
               MovieReviewDataset(os.path.join(extracted_path, 'test', 'neg'), tokenizer, 0)
```


function `collate_fn` pads batch data to have a same length.
The padding value is different between the variables.
* input_ids(-> tokenizer.pad_token_id): tokenized input sequences
* attention_maks(-> 0): It used to distinguish the real token and padding token of input_ids. There are 1 in real token site and 0 in padded site.
* labels(-> -100): the ground truth sequence to compare with the model outputs (-100 is used to ignore the padded location when calculating loss)

```
# 4-3. Define Collate Function for padding
def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'].squeeze(0) for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'].squeeze(0) for x in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'].squeeze(0) for x in batch], batch_first=True, padding_value=-100)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
```


I used the small batch_size because of the GPU limitation.
```
# 5. Use PyTorch's DataLoader to manage your dataset
# Hyperparameters
batch_size = 4
epochs = 8
num_workers = 4

# Load the dataset
trainloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                         collate_fn=collate_fn, shuffle=True, num_workers=num_workers)

valloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,
                       collate_fn=collate_fn, shuffle=False, num_workers=num_workers)

testloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                        collate_fn=collate_fn, shuffle=False, num_workers=num_workers)
```

---

### b. Fine-tune GPT-2 using a movie review dataset (IMDB) to generate more realistic movie reviews. Please define your own loss function in your report. You can use any metric or loss functions for leveraging training and validatation datasets.

Before starting the training, I mounted the code to the google drive to save checkpoint data. If I mount it before loading a dataset, it would take much longer time.
```
import os
from google.colab import drive

# Google Drive mount
drive.mount('/content/drive')
```

The code below is my customed loss function. I used log softmax with smoothing strategy.
It is composed with some techniques like,
###### 1. `shift_logits` and `shift_labels`
: It is a common strategy for sequence prediction task. In language modeling, the model uses current token(logits) to predict a next token(labels). `shift_logits` is a output of a model excluding the last dimension of data because there's nothing left to predict using the last token. `shift_labels` is a ground truth sequence excluding the first dimension because there no need and way to predict the first token.
(e.g. seq: [A, B, C, D] -> input: [A, B, C] & output [B, C, D] )

###### 2. Dimension transformation
* `shift_logits = shift_logits.view(-1, shift_logits.size(-1))`
* `shift_labels = shift_labels.view(-1)`
These codes make them into a 2-D tensor enabling to predict and compare.

###### 3. Mask generation and application
* `mask = shift_labels != -100`
  We mask the padding spots not going to use for loss calculation. It is the location of labels' value is -100 as we intended at the fn `collate_fn`.
* `shift_logits = shift_logits[mask]`
* `shift_labels = shift_labels[mask]`
  We filter the logit and label excluding the masked spots.

###### 4. Label smoothing
smoothing coefficient is set as 0.1 and it lower the probability of actual labels distributing it to the other classes. (confidence get lower and low_confidence get higher)

###### 5. Calculate log probability and loss
Generate one-hot label assigning confidence value at the actual label position. 
Calculate the log probability using softmax and loss multiplying it with label smoothed one-hot label, using CrossEntropy Loss.

```
import os
import torch
import torch.nn.functional as F

def custom_loss_fn(outputs, labels, smoothing=0.1):
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    mask = shift_labels != -100

    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]

    if shift_logits.size(0) == 0:
        return torch.tensor(0.0, device=shift_logits.device)

    n_classes = shift_logits.size(-1)
    confidence = 1.0 - smoothing
    low_confidence = smoothing / (n_classes - 1)

    one_hot_labels = torch.full((shift_labels.size(0), n_classes), low_confidence).to(shift_logits.device)
    one_hot_labels.scatter_(1, shift_labels.unsqueeze(1), confidence)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    loss = -(one_hot_labels * log_probs).sum(dim=-1).mean()

    if loss.item() < 0:
        print(f"Negative loss detected: {loss.item()}")
        print(f"Shift logits: {shift_logits}")
        print(f"One hot labels: {one_hot_labels}")

    return loss
```

This function is to save the checkpoint model parameters information. The epoch, batch, model, optimizer, and loss all are the values saved in the checkpoint. 
Using `max_num_checkpoints`, if there are more checkpoint models than the number of `max_num_checkpoints` at the `checkpoint_dir`, the oldest models are deleted keeping the total number of saved models.
```
def save_checkpoint(epoch, batch, model, optimizer, loss, checkpoint_dir, max_num_checkpoints=3):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_batch_{batch}.pth")
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    all_checkpoints.sort(key=os.path.getctime, reverse=True)
    if len(all_checkpoints) > max_num_checkpoints:
        for checkpoint in all_checkpoints[max_num_checkpoints:]:
            try:
                os.remove(checkpoint)
                print(f"Deleted old checkpoint: {checkpoint}")
            except Exception as e:
                print(f"Error deleting old checkpoint: {e}")

    return checkpoint_path
```

In the `train_model` function, we not only train the model but also calculate validation loss for early stopping. The `patience` is two which means that if the validation loss get lower for two time in a row, we determine that the model is getting overfitting and stop training.
In there, I saved the checkpoint model when step is 4999 right before calculating validation loss. It helps to compare validation loss after reloading checkpoint model before starting training again.

The thing we have to careful is adding a reseting code, `start_batch = 0`. Unless we use it, the model will only train the last step.
```
def train_model(model, train_loader, val_loader, optimizer, custom_loss_fn, num_epochs=1, start_epoch=0, start_batch=0, prev_loss=None, accumulation_steps=4, checkpoint_dir="model_checkpoints", patience=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    trials = 0

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        iter_train_loader = enumerate(train_loader)

        # Skip batches until reaching the desired starting position
        for _ in range(start_batch):
            try:
                next(iter_train_loader)
            except StopIteration:
                print(f"Reached end of dataset within starting offset (start_batch={start_batch})")
                break

        for i, batch in iter_train_loader:
            inputs = {key: val.squeeze().to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = custom_loss_fn(outputs, inputs['labels'])
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if (i + 1) % 1000 == 0:  # Print loss every 50 steps
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

            try:
                # Save checkpoint only at batch 4999
                if (i + 1) == 4999:
                    checkpoint_path = save_checkpoint(epoch, i, model, optimizer, loss, checkpoint_dir)
                    print(f"Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
                # Delete incomplete checkpoint if an exception occurs
                if os.path.exists(checkpoint_path):
                    try:
                        os.remove(checkpoint_path)
                        print(f"Deleted incomplete checkpoint: {checkpoint_path}")
                    except Exception as e:
                        print(f"Error deleting incomplete checkpoint: {e}")

        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: val.squeeze().to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = custom_loss_fn(outputs, inputs['labels'])
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, i, model, optimizer, val_loss, checkpoint_dir)
            print(f"New best validation loss: {val_loss}. Model saved.")
            trials = 0
        else:
            trials += 1
            print(f"Validation loss did not improve. Trials: {trials}")
            if trials >= patience:
                print("Early stopping triggered")
                break

        model.train()

        # Reset start_batch for the next epoch
        start_batch = 0
```

The `epochs` is the total number of epoch we want the train the model. If the saved model already did 10 epochs, then when we implement the code with `epochs = 12`, it only iterates two times.

The optimizer `AdamW` is a modification of Adam optimizer, having different application method of weight decay. `AdamW` processes weight decay and gradient update independently, which allows to get a better generalization performance.

When there are more than one checkpoint models in the `checkpoint_dir` path, the model get the information from the latest one. However, if it is unstable, it can use the second latest checkpoint information. After reloading the information, the model resumes training from the step(batch) and epoch it halted.

If there are no checkpoints at all, the model starts the training from the scratch.
```
epochs = 12

optimizer = AdamW(model.parameters(), lr=5e-5)

# Setting the checkpoint save path
checkpoint_dir = '/content/drive/MyDrive/Colab Notebooks/DNN/project2/model_checkpoints'

# Generate directory
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
  print(f"Created checkpoint directory: {checkpoint_dir}")

else:
  print(f"Using existing checkpoint directory: {checkpoint_dir}")


# Check for file existence and find the latest checkpoint file
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
if len(checkpoint_files) > 1:
    # Sort checkpoint files by generated time
    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)), reverse=True)

    try:
        # Choose the most latest file
        latest_checkpoint = checkpoint_files[0]

        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f'Loading checkpoint from {checkpoint_path}')

        # load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # load the model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # load the optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # load the epoch, batch, and loss value
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        prev_loss = checkpoint['loss']
        print(f'Resuming training from epoch {start_epoch} with loss {prev_loss}')
        # Resume the training
        train_model(model, trainloader, valloader, optimizer, custom_loss_fn, num_epochs=epochs, start_epoch=start_epoch, start_batch=start_batch, prev_loss=prev_loss, checkpoint_dir=checkpoint_dir)

    except Exception as e: # If the latest checkpoint save was unstable
        # Choose the second latest file
        latest_checkpoint = checkpoint_files[1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f'Loading checkpoint from {checkpoint_path}')

        print(f'Error loading checkpoint: {e}')

        # load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # load the model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # load the optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # load the epoch, batch, and loss value
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        prev_loss = checkpoint['loss']
        print(f'Resuming training from epoch {start_epoch} with loss {prev_loss}')

        # Resume the training
        train_model(model, trainloader, valloader, optimizer, custom_loss_fn, num_epochs=epochs, start_epoch=start_epoch, start_batch=start_batch, prev_loss=prev_loss, checkpoint_dir=checkpoint_dir)

else:
    print('No checkpoint found. Starting training from scratch.')
    start_epoch = 0
    start_batch = 0
    prev_loss = None

    # Train a new model
    train_model(model, trainloader, valloader, optimizer, custom_loss_fn, num_epochs=epochs, start_epoch=start_epoch, start_batch=start_batch, prev_loss=prev_loss, checkpoint_dir=checkpoint_dir)
```

---
### c. Create a set of prompts (e.g., "This movie was", "The actors in the film") to guide the generation of movie reviews. You can choose any prompts from your own criteria.
I used a prompts from the test dataset to compare the generated output and ground-truth reviews. Some have positive and negative words like or unlike their label is positive or negative. 
```
# 1-c: Create prompts for generating movie reviews
prompts = ["I've seen this movie more than once. It isn't the", # pos
           "Sadly, every single person I ask about this series says",
           "I loved this movie! Yes, it is rather cheap and ",
           "Okay, maybe this movie not a revolution. But it a",
           "Terrible psychological thriller that is almost painful to sit through", # neg
           "How pointless, hideous characters and boring film. Saved by brief",
           "Ray Bradbury, run and hide! This tacky film version of"]


# 1-d: Generate and save movie reviews
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.tokenize import word_tokenize
import nltk
from evaluate import load

# Download a NLTK punkt data
nltk.download('punkt')
```

After freezing model, I made a `generated reviews` containing reviews created along after prompts respectively and saved it as a text file.

* `top_k` sampling is a method to select top k canditate words having the highest probability for next word prediction and choose one of them randomly. It increases diversity preventing generated text to become too simple or repeat specific predictable words. As it increases, the text can be creative but the content could lost the meaning.

* `top_p` sampling selects high probability candidate words until the accumulative probability become higher than p when the model predicts the next word. Then, when it surpasses, it select one of them randomly. It is much flexible sampling method because it can remove words having very low probability, controling the number of words that need according to the situation. As it increases, the word range that can be selected be broaden, allowing generating more diverse and creative text generation.
```
model.eval()
generated_reviews = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # Move input to the GPU
    attention_mask = torch.ones(input_ids.shape, device=device)  # Setting the Attention mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=100, do_sample=True, top_k=40, top_p=0.85, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    review = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_reviews.append(review)

with open("generated_reviews.txt", "w", encoding="utf-8") as f:
    for review in generated_reviews:
        f.write(review + "\n")
```

I
```
# 1-e: Evaluate using BLEU metric

bleu = load("bleu")

# Compute BLEU scores
bleu_scores = []
for review, batch in zip(generated_reviews, testloader):
    true_text_ids = batch['labels'][0].tolist()
    # Keep only valid tokens
    true_text_ids = [token for token in true_text_ids if token is not None and isinstance(token, int) and token >= 0]

    if len(true_text_ids) > 0:  # Decode only if list is non-empty
        true_text = tokenizer.decode(true_text_ids, skip_special_tokens=True)
        print("Generated Review:", review)
        print("True Text:", true_text)
        prediction_tokens = " ".join(word_tokenize(review))
        reference_tokens = " ".join(word_tokenize(true_text))
        bleu_score = bleu.compute(predictions=[prediction_tokens], references=[[reference_tokens]])
        bleu_scores.append(bleu_score["bleu"])

print(f"\nMean BLEU score: {round(sum(bleu_scores) / len(bleu_scores), 4)}")
```


---
### d. Generate and report 30 movie reviews using the GPT 2 model and the prompts. Save the generated reviews as a list or a text file.



---
### e. Evaluate your model using a BLEU metric https://huggingface.co/spaces/evaluate-metric/bleu with your testing dataset. Report the mean value of Bleu Scores from using the testing dataset. 
We are going to use first 10 words of a review as prompts for your model and compare the generated text with the true text to compute the BLEU score.
