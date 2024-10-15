from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from training.base_train import BaseTrainer


class SupervisedTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3,
            batch_size=16, 
            output_dir='HW2-supervised',
            result_file='supervised_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        self.model = model
        self.tokenizer = tokenizer
        # TODO: set up the training arguments
        self.args = SFTConfig(
            dataset_text_field="text",
            output_dir=output_dir,
            num_train_epochs=num_epoch,
            max_seq_length=1024,
            auto_find_batch_size=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size
        )
        # TODO: set up the data collator to prepare the data for training. 
        # I suggest using the DataCollatorForCompletionOnlyLM data collator
        self.collator = DataCollatorForCompletionOnlyLM(
            response_template="Question:",
            instruction_template="\nAnswer:",
            tokenizer=tokenizer
        )

    def train(self, dataset):
        # TODO: Use the SFTTrainer to set up the training. 
        # Call the train method of the SFTTrainer class, 
        # and don't forget to push the model to the model hub.
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=self.collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"]
        )
        trainer.train()
        trainer.push_to_hub()


