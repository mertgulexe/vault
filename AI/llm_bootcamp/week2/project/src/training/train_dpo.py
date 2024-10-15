from training.base_train import BaseTrainer
from trl import DPOConfig, DPOTrainer
import copy


class MyDPOTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir='HW2-dpo',
            result_file='dpo_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        # TODO: Set the training arguments up with the DPOConfig
        self.args = DPOConfig(
            num_train_epochs=num_epoch,
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size
        )

    def train(self, dataset):
        # TODO:  Set the training up with the DPOTrainer.  
        # Call the train method of the DPOTrainer class, 
        # and don't forget to push the model to the model hub
        model_ref = copy.deepcopy(self.model)
        trainer = DPOTrainer(
            model=self.model,
            ref_model=model_ref,
            tokenizer=self.tokenizer,
            args=self.args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"]
        )
        trainer.push_to_hub(self.output_dir)