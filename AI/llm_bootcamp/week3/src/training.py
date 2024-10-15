# pytorch related imports
from torch.utils.data import DataLoader
from torch import optim
import torch

# others
from utils import HUGGINGFACE_TOKEN
from accelerate import Accelerator
from huggingface_hub import login
from tqdm import tqdm
import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        num_epochs: int,
        batch_size: int,
        push_to_hub: bool,
        repo_name: str = "distributed-model",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accelerator = Accelerator()
        self.optimizer = optim.Adam(params=model.parameters())
        self.push_to_hub = push_to_hub
        self.repo_name = repo_name

    def create_dataloaders(self, tokenized_data):
        train_dataloader = DataLoader(
            dataset=tokenized_data["train"],
            shuffle=True,
            batch_size=self.batch_size
        )
        eval_dataloader = DataLoader(
            dataset=tokenized_data["validation"],
            shuffle=False,
            batch_size=self.batch_size
        )
        return train_dataloader, eval_dataloader

    def train(self, tokenized_data):
        train_dataloader, eval_dataloader = self.create_dataloaders(tokenized_data)
        self.model.to(DEVICE)

        model, optimizer, train_dataloader, eval_dataloader = (
            self.accelerator.prepare(  # this method is added after we wrote the vanilla training loop
                self.model,  # model: see the commented-out changes below
                self.optimizer,
                train_dataloader,
                eval_dataloader,
            )
        )

        # training loop
        self.accelerator.print("--- Training has started. ---")
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            # self.model.train()
            model.train()

            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                # accelerator will take care of the "##" lines

                ## input_ids = batch["input_ids"].to(DEVICE)
                ## attention_mask = batch["attention_mask"].to(DEVICE)
                ## labels = batch["labels"].to(DEVICE)

                # self.optimizer.zero_grad()
                optimizer.zero_grad()

                ## outputs = self.model(
                ##     input_ids=input_ids,
                ##     attention_mask=attention_mask,
                ##     labels=labels
                ## )
                outputs = model(**batch)
                loss = outputs.loss
                ## loss.backward()
                self.accelerator.backward(loss)
                # self.optimizer.step()
                optimizer.step()

            eval_metric = self.eval(model, eval_dataloader)  # self.model is gone too.
            self.accelerator.print(f"\nEpoch {epoch+1}:", eval_metric)
            self.accelerator.print()

        if self.push_to_hub:
            self.save(
                model=model,
                tokenizer=self.tokenizer,
                repo_name=self.repo_name
            )
    
    @torch.no_grad()  # we need this too alongside .eval() due to normlayers
    def eval(self, model, eval_dataloader):
        model.eval()
        all_predictions = []
        all_labels = []
        accuracy_metric = evaluate.load("accuracy")

        for batch in eval_dataloader:
            outputs = model(**batch)
            # logits' shape: [b, seq_length, num_labels]
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.append(
                self.accelerator.gather(predictions)
            )  # need to gather distributed data to compute the metrics
            all_labels.append(self.accelerator.gather(batch["labels"]))

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        eval_metric = accuracy_metric.compute(
            predictions=all_predictions, references=all_labels
        )
        return eval_metric

    def save(self, model, tokenizer, repo_name: str):
        unwrapped_model = self.accelerator.unwrap_model(model=model)
        login(token=HUGGINGFACE_TOKEN)
        unwrapped_model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
