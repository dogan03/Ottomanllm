import torch
from tqdm import tqdm
import random 
class PreTrainer:
  def __init__(self,Training_args,dataset):
    self.Training_args = Training_args
    self.dataset = dataset
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def GetReady(self):

    with open(self.Training_args.TEXT_PATH, "r") as f:
      text = f.read().split("\n")

    bag = [sentence for para in text for sentence in para.split(".") if sentence != ""]

    sentence_a = []
    sentence_b = []
    label = []

    for paragraph in text:
      sentences = [
          sentence for sentence in paragraph.split(".") if sentence != ""
      ]
      num_sentences = len(sentences)
      if num_sentences > 1:
        start = random.randint(0, num_sentences - 2)
        sentence_a.append(sentences[start])
        if random.random() > 0.5:
          sentence_b.append(bag[random.randint(0, len(bag) - 1)])
          label.append(1)
        else:
          sentence_b.append(sentences[start + 1])
          label.append(0)

    inputs = self.Training_args.TOKENIZER(sentence_a, sentence_b, return_tensors="pt",
                      max_length = self.Training_args.MAX_LENGTH,padding="max_length",
                      truncation=True)
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < self.Training_args.RANDOM_PERCENTAGE) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    inputs["next_sentence_label"] = torch.LongTensor([label]).T
    inputs["labels"] = inputs.input_ids.detach().clone()
    
    dataset = self.dataset(inputs)
    self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.Training_args.BATCH_SIZE, shuffle=True)
    
  def Train(self):
    model = self.Training_args.MODEL.to(self.device)
    optim = self.Training_args.OPTIMIZER(model.parameters(), lr=self.Training_args.LEARNING_RATE)

    for epoch in range(self.Training_args.EPOCHS):
      loop = tqdm(self.dataloader, leave=True)
      for batch in loop:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        next_sentence_labels = batch["next_sentence_label"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
            next_sentence_label=next_sentence_labels
        )
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())
  def Save(self):
    pass
class PreTrainerDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

class TrainingArgs:
    def __init__(self, model, tokenizer, text_path, max_length, random_percentage, batch_size, learning_rate, num_epochs, optimizer):
        self.MODEL = model
        self.TOKENIZER = tokenizer
        self.TEXT_PATH = text_path
        self.MAX_LENGTH = max_length
        self.RANDOM_PERCENTAGE = random_percentage
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.EPOCHS = num_epochs
        self.OPTIMIZER = optimizer

    def __str__(self):
        pass