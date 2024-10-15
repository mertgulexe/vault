class DataProcessor:

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def transform(self, data):
        # TODO: Implement the DataProcessor. 
        # You need first to tokenize the data, 
        # and you can assume that the class has as an attribute 
        # the tokenizer related to the model you are going to use. 
        # Consider truncating or padding the resulting tokenized data. 
        # You need to make sure that the resulting data has the right 
        # column names for training: labels, input_ids, attention_mask. 
        # Make sure the resulting tensors are PyTorch tensors 
        # by using the set_format(type='torch') function.
        tokenized_data = data.map(
            function=self._tokenizer_fnc,
            batched=True,
            remove_columns=["text"]
        ).rename_column("label", "labels")
        tokenized_data.set_format(type="torch")
        return tokenized_data


    def _tokenizer_fnc(self, example):
        return self.tokenizer(
            example["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )