from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Model:

    @staticmethod
    def get_model(model_id: str, num_labels: int):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
