# pytorch related imports
from torch.utils.data import DataLoader
import torch

# other imports
from accelerate.utils import DeepSpeedPlugin
from accelerate import Accelerator
from dotenv import load_dotenv
import huggingface_hub as hf
from tqdm import tqdm
import evaluate
import wandb
import json
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    with open("./hf_ds_config.json", 'r') as f:
        deepspeed_configs = json.load(f)
except FileNotFoundError:
    deepspeed_configs = None
    print("DeepSpeedPlugin configuration file couldn't be found!")

# TODO: Initialize the wandb library by using the login function
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.require("core")
# wandb.init(
#     project="llm-project3",
#     entity="gulmert89-kariyer-net",
#     # name="unique_run_name",
#     # reinit=True
# )
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"


class BasicTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        num_epochs: int,
        batch_size: int,
        hf_token: str,
        push_to_hub: bool = False,
        repo_name: str = "project3-supervised-model"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.push_to_hub = push_to_hub
        self.repo_name = repo_name
        self.optimizer = torch.optim.Adam(params=model.parameters())

    def train(self, tokenized_data):
        # TODO: implement train. There are a few steps to follow in the train function:
        # from the tokenized data, we need to create the data loaders
        train_dataloader, eval_dataloader = self.create_dataloaders(
            tokenized_data=tokenized_data
        )
        # TODO: we need to project the model onto the right device (.to(device))

        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            # TODO: set the model up in train mode: model.train()
            self.model.train()
            print("[TRAIN] Epoch:", epoch+1)
            for batch in tqdm(train_dataloader, total=len(train_dataloader)):
                # TODO: zero out the optimizer: optimizer.zero_grad() 
                self.optimizer.zero_grad()
                # TODO: we project the data batch onto the right device
                batch = {k: v.to(device) for k, v in batch.items()}  # items contain `input_ids`, `attention_masks` & `labels`
                # TODO: we feed the batch to the model and get the model outputs
                outputs = self.model(**batch)
                loss = outputs.loss
                # TODO: we call the backward function on the loss function: loss.backward()
                loss.backward()
                # TODO: we step the optimizer: optimizer.step()
                self.optimizer.step()

            # TODO: compute the evaluation metric on the validation data
            # TODO: as an optional task, we can implement an early stopping to avoid overfitting.
            eval_metrics = self.eval(
                model=self.model, eval_dataloader=eval_dataloader
            )
            print(f"[EVALUATION] Epoch {epoch+1} accuracy:", eval_metrics)
            print()

        if self.push_to_hub:
            self.save(
                model=self.model,
                tokenizer=self.tokenizer,
                repo_name=self.repo_name,
                hf_token=self.hf_token
            )

    @torch.no_grad()
    def eval(self, model, eval_dataloader):
        # TODO: Implement eval. The eval function computes a validation metric on the validation data. 
        # You can use the evaluate package to get access to the evaluation metric you prefer:
        # 
        # accuracy_metric = evaluate.load("accuracy")
        # 
        # To accurately compute the validation metric on the validation dataset, there are a few steps:
        # - You need to set the model in eval mode: model.eval()
        # - When you iterate through each batch in the eval_dataloader, you need to project the data on the right device: 
        # batch = {k: v.to(device) for k, v in batch.items()}
        # - You need to infer the batch with the model without aggregating the gradients: with torch.no_grad()
        # - You need to compare the prediction to the labels to compute the metric. For example:
        # eval_metric = accuracy_metric.compute(
        #        predictions=all_predictions, 
        #        references=all_labels
        # ) 
        # You can then return the evaluation metric
        model.eval()
        metrics_evaluator = evaluate.load(path="accuracy")
        all_predictions, all_labels = [], []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}  # items contain `input_ids`, `attention_masks` & `labels`
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            all_predictions.append(predictions)
            all_labels.append(labels)
        all_predictions = torch.cat(tensors=all_predictions)
        all_labels = torch.cat(tensors=all_labels)
        return metrics_evaluator.compute(
            predictions=all_predictions,
            references=all_labels
        )

    def create_dataloaders(self, tokenized_data):

        # TODO: Use the torch.utils.data.DataLoader class to create iterators around the data. 
        # Make sure to create a data loader for the training data and one for the validation data.
        train_dataloader, eval_dataloader = (
            DataLoader(
                dataset=tokenized_data["train"],
                batch_size=self.batch_size,
                shuffle=True,
            ),
            DataLoader(
                dataset=tokenized_data["validation"],
                batch_size=self.batch_size,
                shuffle=False
            )
        )
        return train_dataloader, eval_dataloader
    
    def save(self, model, tokenizer, repo_name: str, hf_token: str):
        # TODO: Let's save the model to the HuggingFace model hub. Implement the save function:
        # - Give a name to the repo
        # - call the login function with your HuggingFace token
        # - call push_to_hub on the model and tokenizer
        hf.login(token=hf_token, write_permission=True)
        model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)


class AcceleratedTrainer:

    """_summary_
    A Modified version of the BasicTrainer to handle distributed training
    """
    def __init__(
        self,
        model,
        tokenizer,
        num_epochs: int,
        batch_size: int,
        hf_token: str,
        push_to_hub: bool = False,
        repo_name: str = "project3-supervised-distributed-model"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.push_to_hub = push_to_hub
        self.repo_name = repo_name
        self.optimizer = torch.optim.AdamW(params=model.parameters())
        # TODO: instantiate as class attribute an accelerator
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_configs)
        # print(deepspeed_plugin.deepspeed_config)  # Mert: to see the default configs
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin, log_with="wandb"
        )
        # TODO: We then need to set up the accelerator to log with wandb. Pass the argument log_with="wandb"
        # TODO: Enable Zero Redundancy Optimizer Strategy. 
        # You may need to update the deepspeed package in the requirements.txt.

    def train(self, tokenized_data):
        
        train_dataloader, eval_dataloader = self.create_dataloaders(
            tokenized_data=tokenized_data
        )

        # TODO: use the prepare function to prepare the model, optimizer, train_dataloader, 
        # and eval_dataloader for distributed training. 
        # Because of this, we don't need to project the model and the data on the device, 
        # as Accelerate does it automatically.
        model, optimizer, train_dataloader, eval_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader, eval_dataloader
        )

        # TODO: Just before the code training the model, we need to initialize the tracker 
        # with the init_trackers function. You can pass additional hyperparameters to that function
        log_configs = {
            "model_name": self.model.name_or_path,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "total_training_batches": len(train_dataloader),
            "optimizer": str(self.optimizer.__class__),
            "lr": self.optimizer.defaults["lr"],
            "weight_decay": self.optimizer.defaults["weight_decay"],
            "repo_name": self.repo_name,
        }
        self.accelerator.init_trackers(
            project_name="llm-project3", config=log_configs
        )  # -----------------------------------------[TRAINING LOGGING STARTS]

        training_step_counter = 0
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            model.train()
            metrics_evaluator = evaluate.load(path="accuracy")
            all_predictions, all_labels = [], []
            self.accelerator.print("-----> Epoch:", epoch+1, "has started.")
            training_loss = 0.0
            
            for batch in tqdm(
                iterable=train_dataloader,
                total=len(train_dataloader)
            ):
                # TODO: Modify the backward call to be handled by the 
                # accelerator instead of being directly called on the loss function
                optimizer.zero_grad()
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                all_predictions.append(predictions)
                all_labels.append(labels)
                loss = outputs.loss
                loss_value = loss.detach().item()
                training_loss += loss_value
                self.accelerator.backward(loss=loss)
                optimizer.step()
                self.accelerator.log(
                    values={
                        "training_batch_loss": loss_value,
                        "step": training_step_counter
                    }
                )
                training_step_counter += 1
            # training metrics
            all_predictions = torch.cat(tensors=all_predictions)
            all_labels = torch.cat(tensors=all_labels)
            train_metrics = metrics_evaluator.compute(
                predictions=all_predictions,
                references=all_labels
            )
            train_metrics["loss"] = training_loss / len(train_dataloader.dataset)
            # evaluation
            eval_metrics = self.eval(
                model=model, eval_dataloader=eval_dataloader
            )
            self.accelerator.log(
                values={
                    "training_epoch_loss": train_metrics["loss"],
                    "training_epoch_accuracy": train_metrics["accuracy"],
                    "eval_epoch_loss": eval_metrics["loss"],
                    "eval_epoch_accuracy": eval_metrics["accuracy"],
                    "epoch": epoch
                }
            )
        # TODO: At the end of the training, make sure to disconnect the tracker
        self.accelerator.end_training()  # -------------[TRAINING LOGGING ENDS]
        if self.push_to_hub:
            self.save(
                model=model,  # if model is distributed as well, do we need to call `.gather` method for the model?
                tokenizer=self.tokenizer,
                repo_name=self.repo_name,
                hf_token=self.hf_token
            )
        
    @torch.no_grad()
    def eval(self, model, eval_dataloader):
        # TODO: With Accelerate, we cannot directly compute the evaluation metrics 
        # as the data is spread across multiple machines or processes, 
        # so we need to bring it back to the main thread. 
        # To do that, we use the function gather. 
        # Modify the eval function by using the gather function 
        # before computing the evaluation metric.
        model.eval()
        metrics_evaluator = evaluate.load(path="accuracy")
        all_predictions, all_labels = [], []
        total_eval_loss = 0.0
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            loss_value = outputs.loss.detach().item()
            total_eval_loss += loss_value
            labels = batch["labels"]
            all_predictions.append(self.accelerator.gather(tensor=predictions))
            all_labels.append(self.accelerator.gather(tensor=labels))
        all_predictions = torch.cat(tensors=all_predictions)
        all_labels = torch.cat(tensors=all_labels)
        total_eval_loss = total_eval_loss / len(eval_dataloader.dataset)
        eval_metrics = metrics_evaluator.compute(
            predictions=all_predictions,
            references=all_labels
        )
        eval_metrics["loss"] = total_eval_loss
        return eval_metrics
        

    def create_dataloaders(self, tokenized_data):
        # TODO: Use the torch.utils.data.DataLoader class to create iterators around the data. 
        # Make sure to create a data loader for the training data and one for the validation data.
        train_dataloader, eval_dataloader = (
            DataLoader(
                dataset=tokenized_data["train"],
                batch_size=self.batch_size,
                shuffle=True
            ),
            DataLoader(
                dataset=tokenized_data["validation"],
                batch_size=self.batch_size,
                shuffle=False
            )
        )
        return train_dataloader, eval_dataloader
    
    def save(self, model, tokenizer, repo_name: str, hf_token: str):
        # TODO:  Before we can save the model, we need to undo what the prepare function did. 
        # For that, we need to call the unwrap_model function.
        #  Modify the save function by calling the unwrap_model function before saving it to the hub.
        hf.login(token=hf_token, write_permission=True)
        unwrapped_model = self.accelerator.unwrap_model(model=model)
        unwrapped_model.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
