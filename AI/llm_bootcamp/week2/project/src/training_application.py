# training imports
from training.train_rlhf import RewardModelTrainer, RLHFTrainer
from training.train_supervised import SupervisedTrainer
from training.train_dpo import MyDPOTrainer
from training.train_orpo import MyORPOTrainer

# data imports
from data.data_connection import DataConnector
from data.data_processing import DataProcessor

# model imports
from model.model_connection import Model


def run(training_type):
    # TODO: implement this function.
    trainer = None
    processed_data = None

    if training_type == "supervised":
        model, tokenizer = Model.get_model_for_LM(
            model_id="openai-community/gpt2"
        )
        data_processor = DataProcessor(tokenizer=tokenizer)
        unprocessed_data = DataConnector.get_data(path="ybisk/piqa")
        processed_data = data_processor.prepare_for_supervised_training(
            dataset=unprocessed_data
        )
        trainer = SupervisedTrainer(
            model=model,
            tokenizer=tokenizer,
            num_epoch=6,
            batch_size=64,
            output_dir="HW2-supervised",
            result_file="supervised_results.json"
        )
    elif training_type == "reward":
        model, tokenizer = Model.get_model_for_reward(
            model_id="HW2-supervised"
        )
        data_processor = DataProcessor(tokenizer=tokenizer)
        unprocessed_data = DataConnector.get_data(path="ybisk/piqa")
        processed_data = data_processor.prepare_for_reward_training(
            dataset=unprocessed_data
        )
        trainer = RewardModelTrainer(
            model=model,
            tokenizer=tokenizer,
            num_epoch=2,
            batch_size=16,
            output_dir="HW2-reward"
        )

    elif training_type == "ppo":
        model, tokenizer = Model.get_model_for_PPO(model_id="HW2-supervised")
        data_processor = DataProcessor(tokenizer=tokenizer)
        unprocessed_data = DataConnector.get_data(path="ybisk/piqa")
        processed_data = data_processor.prepare_for_ppo_training(
            dataset=unprocessed_data
        )
        trainer = RLHFTrainer(
            model=model,
            tokenizer=tokenizer,
            num_epoch=2,
            batch_size=32,
            output_dir="HW2-ppo"
        )

    elif training_type == "dpo":
        model, tokenizer = Model.get_model_for_LM(
            model_id="HW2-supervised"
        )
        data_processor = DataProcessor(tokenizer=tokenizer)
        unprocessed_data = DataConnector.get_data(path="ybisk/piqa")
        processed_data = data_processor.prepare_for_dpo_training(
            dataset=unprocessed_data
        )
        trainer = MyDPOTrainer(
            model=model,
            tokenizer=tokenizer,
            num_epoch=3,
            batch_size=128,
            output_dir="HW2-dpo"
        )
        # Gave these warnings:
        # ...python3.12/site-packages/trl/trainer/dpo_trainer.py:394: UserWarning: `max_length` is not set in the DPOConfig's init it will default to `512` by default, but you should do it yourself in the future.
        # ...python3.12/site-packages/trl/trainer/dpo_trainer.py:407: UserWarning: `max_prompt_length` is not set in the DPOConfig's init it will default to `128` by default, but you should do it yourself in the future.
        # ...python3.12/site-packages/trl/trainer/dpo_trainer.py:442: UserWarning: When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments we have set it for you, but you should do it yourself in the future.

    elif training_type == "orpo":
        model, tokenizer = Model.get_model_for_LM(
            model_id="openai-community/gpt2"
        )
        data_processor = DataProcessor(tokenizer=tokenizer)
        unprocessed_data = DataConnector.get_data(path="ybisk/piqa")
        processed_data = data_processor.prepare_for_orpo_training(
            dataset=unprocessed_data
        )
        trainer = MyORPOTrainer(
            model=model,
            tokenizer=tokenizer,
            num_epoch=3,
            batch_size=128,
            output_dir="HW2-orpo"
        )

    else: 
        raise NotImplemented
    print("Training has started!")
    trainer.train(processed_data)


if __name__ == "__main__":
    run("orpo")