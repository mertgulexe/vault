from trl import AutoModelForCausalLMWithValueHead
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer
)


class Model:

    @staticmethod
    def get_model_for_LM(model_id):
        # TODO: implement the method to get the base model with a language modeling head. 
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            clean_up_tokenization_spaces=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
    
    @staticmethod
    def get_model_for_reward(model_id):
        # TODO: implement the method to get the reward model with a sequence classification head.
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            clean_up_tokenization_spaces=True
        ) 
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
    
    @staticmethod
    def get_model_for_PPO(model_id):
        # TODO: implement the method to get the PPO model with 
        # a language modeling head and a value head as well.
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            clean_up_tokenization_spaces=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer