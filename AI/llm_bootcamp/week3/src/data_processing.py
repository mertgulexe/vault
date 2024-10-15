class DataProcessor:
    def __init__(self, tokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _tokenize_function(self, examples):
        return self.tokenizer(
            text=examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
    
    def transform(self, data):
        tokenized_data = data.map(
            function=self._tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        tokenized_data = tokenized_data.rename_column(
            "label", "labels"  # to conform with Pytorch API
        )
        tokenized_data.set_format(type="torch")
        return tokenized_data
