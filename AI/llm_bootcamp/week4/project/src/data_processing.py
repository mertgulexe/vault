import torch
import math


PROMPT_INPUT = (
    "[INST] <<SYS>>\n"
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
    "<</SYS>> \n\n {instruction} \n{input} [/INST]"
)

PROMPT_NO_INPUT = "[INST]{instruction}[/INST]"

IGNORE_INDEX = -100


class DataProcessor:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def transform(self, dataset):
        # TODO: Implement DataProcessor.transform by mapping 
        # the tokenize_function to the dataset.
        tokenized_data = dataset.map(
            function=self.tokenize_function,
            batched=True,
            batch_size=32
        )
        return tokenized_data

    def create_prompt(self, example):
        # TODO: I provide two prompt templates: PROMPT_INPUT,  PROMPT_NO_INPUT. 
        # If the example has an input, use the PROMPT_INPUT and PROMPT_NO_INPUT otherwise.
        instruction, input_text = example
        if bool(input_text):
            return PROMPT_INPUT.format(
                instruction=instruction,
                input=input_text
            )
        return PROMPT_NO_INPUT.format(instruction=instruction)
        
    def create_target(self, output):
        # TODO: The target will just be the output column with 
        # the end-of-sequence token (available in the 
        # tokenizer: tokenizer.eos_token) appended to it.
        output = output + self.tokenizer.eos_token
        return output
    
    def tokenize_function(self, examples):
        # TODO:  iterate through the examples, and create the prompts 
        # and the targets by using the DataProcessor.create_prompt 
        # and DataProcessor.create_target functions.
        if isinstance(dict(examples), dict):  # Mert: Handle the non-batched case
            examples = [examples]
        prompts = [
            self.create_prompt((x["instruction"], x["input"])) for x in examples
        ]
        targets = [self.create_target(x["output"]) for x in examples]
        # TODO: Create the input_texts by iterating through the prompts 
        # and targets and concatenating them. For example, 
        # input_texts[0] = prompt[0] + targets[0].
        input_texts = [x+y for x, y in zip(prompts, targets)]

        # TODO: use the tokenizer to tokenize the input_texts and prompts. 
        # Truncate to the model_max_length but don't pad the sequence, 
        # as we are going to handle the padding in the data collator. 
        # Use return_tensors=None to return a Python list 
        # (we are going to change to PyTorch in the data collator as well)
        tokenized_inputs = self.tokenizer(
            text=input_texts,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors=None,
            padding="do_not_pad"
        )
        tokenized_prompts = self.tokenizer(
            text=prompts,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors=None,
            padding="do_not_pad"
        )

        input_ids = tokenized_inputs["input_ids"]
        labels = [ids.copy() for ids in input_ids]

        # TODO: For language modeling, the labels are the same as the inputs, 
        # but we are going to replace the tokens related to the prompt 
        # by the IGNORE_INDEX = -100. In labels, replace the token corresponding 
        # to the prompt by IGNORE_INDEX.
        for i in range(len(labels)):
            # Get the length of the prompt for each example
            len_prompt = len(tokenized_prompts["input_ids"][i])
            # Replace tokens for the prompt with IGNORE_INDEX
            labels[i][:len_prompt] = [IGNORE_INDEX] * len_prompt

        return dict(input_ids=input_ids, labels=labels)
    

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, group_size_ratio) -> None:
        self.tokenizer = tokenizer
        self.group_size_ratio = group_size_ratio

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]

        # TODO: input_ids is a batch of tokenized input texts. 
        # For that batch:
        # - find the maximum input sequence length: \(l\)
        # - find the smallest integer value \(L\) greater than \(l\) that is divisible by num_group.  
        # \(L\) is the target_length for that batch 
        # Find the max length in the batch
        max_length = max(list(map(len, input_ids)))
        # Calculate the group size
        group_size = math.floor(self.group_size_ratio * max_length)
        # Round up max_length to the next multiple of group_size
        target_length = math.ceil(max_length / group_size) * group_size
        
        # TODO: pad the input_ids with the padding token 
        # (tokenizer.pad_token_id) and the labels with the IGNORE_INDEX.
        input_ids = [
            self.pad_sequence(
                sequence=seq,
                target_length=target_length,
                pad_value=self.tokenizer.pad_token_id
            ) for seq in input_ids
        ]
        labels = [
            self.pad_sequence(
                sequence=seq,
                target_length=target_length,
                pad_value=IGNORE_INDEX
            ) for seq in labels
        ]
        
        # TODO: Make sure to convert the resulting data structures into a PyTorch tensor. 
        # If you have a list of PyTorch tensors, you can use the torch.stack 
        # function to convert, for example.
        input_ids = torch.stack(tensors=input_ids, dim=0)
        labels = torch.stack(tensors=labels, dim=0)
        
        # TODO: compute the attention_mask as a boolean tensor where True means 
        # input_ids is not equal to the padding token, and False means otherwise. 
        # You can use the torch.ne function to compute this.
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    
    def pad_sequence(self, sequence, target_length, pad_value=None):
        # TODO: pad_sequence should pad the input sequence on the right up to 
        # target_length with pad_value. It should return a PyTorch tensor.
        seq_tensor = torch.tensor(sequence)
        padding_tensor = torch.full(
            size=(target_length, ),
            fill_value=pad_value,
            dtype=seq_tensor.dtype
        )
        padding_tensor[0:len(sequence)] = seq_tensor

        return padding_tensor
