class DataProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_for_supervised_training(self, dataset):
        # TODO: implement the method. 
        # The method should return one column with the text organized as such:
        # "Question: {goal}\nAnswer: {chosen_solution}"
        # This is because it is the way the instructions get presented 
        # to the models in the PIQA test of the Evaluation Harness. 
        # The goal represents the text of the original ''goal" column,
        #  and the chosen_solution is the solution that corresponds to the "label" column. 
        # The returned dataset doesn't need to be tokenized, 
        # but it should be in a DatasetDict format. 
        # You could decide to return a validation dataset as well 
        # to get better visibility during your training.
        processed_dataset = dataset.map(
            function=lambda x: {
                "text": f"Question: {x['goal']}\nAnswer: {x['label']}"
            },
            remove_columns=["goal", "label", "sol1", "sol2"]
        )
        processed_dataset = processed_dataset.train_test_split(test_size=0.1)
        return processed_dataset

    def _tokenize_for_reward_training(self, examples):
        # TODO: The RewardTrainer expects the data 
        # to be tokenized with very specific column names:
        # - "input_ids_chosen"
        # - "attention_mask_chosen"
        # - "input_ids_rejected"
        # - "attention_mask_rejected"
        # Implement the method to tokenize the data using the expected format for the RewardTrainer.
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": []
        }
        for chosen_label, sol1, sol2 in zip(examples["label"], examples["sol1"], examples["sol2"]):
            chosen_solution = f"sol{chosen_label+1}"  # label '0' refers to "sol1" and '1' refers to "sol2"
            if chosen_solution == "sol1":
                chosen = self.tokenizer(sol1)  # had to remove: `remove_unused_columns=False`
                rejected = self.tokenizer(sol2)  # had to remove: `remove_unused_columns=False`
            else:
                chosen = self.tokenizer(sol2)  # had to remove: `remove_unused_columns=False`
                rejected = self.tokenizer(sol1)  # had to remove: `remove_unused_columns=False`
            new_examples["input_ids_chosen"].append(chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(rejected["attention_mask"])
        return new_examples
    
    def prepare_for_reward_training(self, dataset):
        # TODO: Prepare the data: Now that we have a function that 
        # tokenizes the data, we need to prepare that data. 
        # The prepare_for_reward_training method should do the following things:
        # - create two new columns, "chosen" and "rejected," from the original data.
        # - tokenize those columns and return the tokenized data
        # The "chosen" column should have the following format:
        # "Question: {goal}\nAnswer: {chosen_solution}"
        # and the "rejected" column should have the following format:
        # "Question: {goal}\nAnswer: {rejected_solution}"
        processed_dataset = dataset.map(
            function=lambda e: {
                "chosen": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(e["label"])+1}"]}""",
                "rejected": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(not e["label"])+1}"]}"""
            }
        )
        processed_dataset = processed_dataset.train_test_split(test_size=0.1)
        tokenized_data = processed_dataset.map(
            function=self._tokenize_for_reward_training,
            batched=True
        )
        return tokenized_data
    
    def prepare_for_ppo_training(self, dataset):
        # TODO:  Implement the method. 
        # We just need to add the indicators "Question: {goal}\nAnswer: " 
        # and tokenize the resulting text.
        processed_data = dataset.map(
            function=lambda x: {
                "prompt": f"Question: {x['goal']}\nAnswer: "
            }
        )
        tokenized_data = processed_data.map(
            lambda x: self.tokenizer(
                x["prompt"],
                # truncation=True,
                # padding="max_length",
                # return_tensors="pt"
            )
        )
        # Set the format to PyTorch tensors
        tokenized_data.set_format(type="torch")
        return tokenized_data
    
    def prepare_for_dpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace DPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        processed_dataset = dataset.map(
            function=lambda e: {
                "chosen": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(e["label"])+1}"]}""",
                "rejected": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(not e["label"])+1}"]}"""
            }
        ).rename_column("goal", "prompt")
        processed_dataset = processed_dataset.train_test_split(test_size=0.1)
        return processed_dataset
    
    def prepare_for_orpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace ORPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        processed_dataset = dataset.map(
            function=lambda e: {
                "chosen": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(e["label"])+1}"]}""",
                "rejected": f"""Question: {e["goal"]}\nAnswer: {e[f"sol{int(not e["label"])+1}"]}"""
            }
        ).rename_column("goal", "prompt")
        processed_dataset = processed_dataset.train_test_split(test_size=0.1)
        return processed_dataset