from training.base_train import BaseTrainer, login, HUGGINGFACE_TOKEN
from trl import RewardConfig, RewardTrainer, PPOConfig, PPOTrainer
from transformers import pipeline
from tqdm import tqdm
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardModelTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir="HW2-reward",
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir)

        self.model = model
        self.tokenizer = tokenizer
        # TODO: set up the training arguments
        self.args = RewardConfig(
            output_dir=output_dir,
            num_train_epochs=num_epoch,
            report_to="none",
            auto_find_batch_size=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_length=1024,
            remove_unused_columns=False
        )

    def train(self, tokenized_data):
        # TODO: Use the RewardTrainer to set up the training. 
        # Call the train method of the RewardTrainer class, 
        # and don't forget to push the model to the model hub.
        trainer = RewardTrainer(
            model=self.model,
            args=self.args,
            tokenizer=self.tokenizer,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"]
        )
        trainer.train()
        trainer.push_to_hub()
    

class RLHFTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=8, 
            output_dir="HW2-ppo",
            result_file="ppo_results.json"
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)

        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer
        # TODO: implement the training arguments with the PPOConfig. 
        self.args = PPOConfig(
            exp_name=output_dir,
            ppo_epochs=num_epoch,
            batch_size=batch_size,
            mini_batch_size=batch_size,
            remove_unused_columns=False
        )

    def _get_collator(self, data): 
        return dict((key, [d[key] for d in data]) for key in data[0])

    def train(self, tokenized_data):
        # TODO: implement the trainer with the PPOTrainer
        trainer = PPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.args,
            dataset=tokenized_data,
            data_collator=self._get_collator
        )
        # TODO: implement the generation_kwargs that will be used in the PPOTrainer.generate method 
        generation_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_prompt": False,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "max_new_tokens": 32
        }
        # TODO: Implement the reward_pipeline by using your reward model and pipeline function
        reward_pipeline = pipeline(
            task="text-classification",
            model="HW2-reward",
            device=DEVICE
        )
        t_dataloader = trainer.dataloader
        for epoch in tqdm(range(self.num_epoch), total=self.num_epoch):
            print("Epoch", epoch, "has started.")
            for batch in tqdm(t_dataloader, total=len(t_dataloader)): 
                query_tensors = batch["input_ids"]
                
                #### Get response from SFTModel
                # TODO: Generate the response_tensors from the query_tensors
                response_tensors = trainer.generate(
                    query_tensor=query_tensors,
                    **generation_kwargs
                )
                # TODO: Decode the response_tensors by using the tokenizer
                batch["response"] = [
                    self.tokenizer.decode(x.squeeze()) for x in response_tensors
                ]
            
                #### Compute reward score
                # TODO: Create the input text for the reward_pipeline by using batch["goal"] and batch["response"]
                texts = [
                    f"Question: {x}\nAnswer: {y}" for x, y in zip(batch["goal"], batch["response"])
                ]
                # TODO: Pass the input text to the reward_pipeline and extract the score output.
                rewards = reward_pipeline(texts)
                rewards = [torch.tensor(x["score"]) for x in rewards]
            
                #### Run PPO step 
                trainer.step(
                    queries=query_tensors,
                    responses=response_tensors,
                    scores=rewards

                )
                # TODO: Update the PPO model by using the query_tensors, response_tensors,  and rewards.
        self.save(trainer=trainer)

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub(self.output_dir)