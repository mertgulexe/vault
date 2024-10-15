from transformers import (
    AdamW,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)
import torch



class LoRATrainer:

    def __init__(
        self,
        model,
        tokenizer,
        data_collator,
        group_size_ratio,
        push_to_hub: bool = False,
        repo_name: str = "gulmert89/lora_adapter_examples"
    ) -> None:
        # TODO: Complete the LoRATrainer to train with AdamW and a cosine scheduler 
        # with warmup. You can use the AdamW Pytorch class and 
        # the get_cosine_schedule_with_warmup function. 
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.push_to_hub = push_to_hub
        self.repo_name = repo_name
        self.training_arguments = TrainingArguments(
            output_dir="./results",
            max_steps=1000,  # Mert: arbitrary numbers. Also, ``Overrides `num_train_epochs`.``
            # num_train_epochs=2,
            warmup_steps=300,
            weight_decay=0.01,
            gradient_accumulation_steps=10,
            do_train=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            report_to="none"
        )
        self.optimizer = AdamW(params=self.model.parameters(), lr=5e-5)
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=300,  # Mert: arbitrary numbers
            num_training_steps=1000  # Mert: arbitrary numbers
        )

    def train(self, tokenized_data):

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=tokenized_data,
            data_collator=self.data_collator,
            optimizers=(self.optimizer, self.scheduler)
        )

        trainer.train()
        self.save()

    def save(self):
        # TODO: Implement the LoRATrainer.save function.
        if self.push_to_hub:
            self.model.push_to_hub(self.repo_name)
            self.tokenizer.push_to_hub(self.repo_name)
