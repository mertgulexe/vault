from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from attention import SparseAttention
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch



DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

MODEL_ID = 'meta-llama/Meta-Llama-3-8B'

QUANTIZATION_DICT = {
    'load_in_4bit': True,
    'bnb_4bit_compute_dtype': torch.bfloat16,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': "nf4",
}


class ModelLoader:

    def __init__(self, model_id=MODEL_ID, scaling_factor=2, group_size_ratio=1/4, quantization_dict=QUANTIZATION_DICT):
        self.model_id = model_id
        self.scaling_factor = scaling_factor
        self.group_size_ratio = group_size_ratio
        self.quantization_config = BitsAndBytesConfig(**quantization_dict)

    def load_and_prepare_model(self):
        # TODO: Implement ModelLoader.load_and_prepare_model by:
        # - Getting the config
        config = self.get_config(model_id=self.model_id)
        # - expanding the context
        config = self.expand_context(
            scaling_factor=self.scaling_factor, config=config
        )
        # - loading the model
        model = self.load_model(model_id=self.model_id, config=config)
        # - modifying the attentions
        model = self.modify_attention(model=model)
        # - loading the tokenizer
        tokenizer = self.load_tokenizer(
            model_id=self.model_id,
            scaling_factor=self.scaling_factor,
            config=config
        )
        # - resizing the tokenizer
        tokenizer = self.resize_tokenizer(tokenizer=tokenizer, model=model)
        # - adding the adapter
        model = self.add_adapter(model=model)
        return model, tokenizer

    def get_config(self, model_id):
        # TODO: Load the model configuration. 
        # With the transformers package, 
        # we can use the AutoConfig class to load the model config. 
        # This model config defines all the hyperparameters that can 
        # be passed to the model when we instantiate it. 
        # Once the config is loaded, we can modify it and then 
        # pass it to the model to change its structure.
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_id
        )
        return config
    
    def expand_context(self, scaling_factor, config):
        # TODO: modify the RoPE scaling factor
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        return config
    
    def load_model(self, model_id, config):
        # TODO: Implement the ModelLoader.load_model method. 
        # This method should return the model. 
        # You can use the AutoModelForCausalLM.from_pretrained method 
        # with the config that you modified above. 
        # Apply the quantization_config and the prepare_model_for_kbit_training 
        # function only if GPUs are available (torch.cuda.is_available()).
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config=config,
            quantization_config=self.quantization_config if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,  # Mert: See the comment first. Then read this:  ValueError: Passing along a `device_map` requires `low_cpu_mem_usage=True`
            device_map="auto"  # Mert: I added this line because it gave me this error: ``ValueError: weight is on the meta device, we need a `value` to put in on 0.`` Later, changed ``low_cpu_mem_usage=False`` to ``True``
        )
        model = prepare_model_for_kbit_training(
            model=model
        ) if torch.cuda.is_available() else model
        return model

    def modify_attention(self, model):
        # TODO: Implement ModelLoader.modify_attention by iterating 
        # through all the attention layers and replacing them with 
        # the new sparse attention we implemented.
        for layer_index in range(len(model.model.layers)):
            current_attn = model.model.layers[layer_index].self_attn
            attn_config = current_attn.config
            sparse_attn = SparseAttention(
                config=attn_config,
                layer_idx=layer_index
            )
            sparse_attn.load_state_dict(state_dict=current_attn.state_dict())
            model.model.layers[layer_index].self_attn = sparse_attn
        return model

    def add_adapter(self, model):
        # TODO: Use the LoraConfig class and the get_peft_model function to add a LoRA adapter. 
        # We are going to focus the fine-tuning on the attention layer parameters, 
        # so make sure to specify the target modules to be 
        ## target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # This is the way the Query, Key, Value matrices, and the final projection 
        # matrix are named in the attention layers in LLama: 
        # https://github.com/huggingface/transformers/blob/v4.34-release/src/transformers/models/llama/modeling_llama.py#L283. 
        # We are fine-tuning the model for language modeling, so don't forget to specify the task type.
        lora_configs = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = get_peft_model(
            model=model,
            peft_config=lora_configs,
            adapter_name="hw4_adapter",
            mixed=False
        )
        return model
    
    def resize_tokenizer(self, tokenizer, model):
        # TODO: Implement the method. We are going to add the potentially missing 
        # tokens in the tokenizer and model (Padding, End of Sequence, Beginning 
        # of Sequence, and Unknown tokens). Because we are using LoRA, 
        # the weights are going to be frozen, so we have to choose 
        # meaningful values when we initialize the new model weights when we add those tokens. 
        # We need to add the new related vectors in the embedding and the 
        # new related prediction vectors in the output layer. We are just 
        # going to compute the average of the existing vectors for the new vectors. 
        # The process is as follows:
        # - Add the new tokens to the tokenizer using the add_special_tokens method.
        # - Resize the token embedding of the model using the resize_token_embeddings method.
        # - Get the input embedding and the output embedding:
        # - Compute the average of the original vectors.
        # - Replace the values for the newly added vectors:
        new_special_tokens = {
            "pad_token": "[PAD]",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
        }
        NEW_TOKEN_SIZE = len(new_special_tokens)

        tokenizer.add_special_tokens(new_special_tokens)
        model.resize_token_embeddings(
            new_num_tokens=len(tokenizer)
        )
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[0:-NEW_TOKEN_SIZE].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[0:-NEW_TOKEN_SIZE].mean(dim=0, keepdim=True)

        input_embeddings[-NEW_TOKEN_SIZE:] = input_embeddings_avg  # happens inplace, i.e. modifies the model itself
        output_embeddings[-NEW_TOKEN_SIZE:] = output_embeddings_avg

        return tokenizer
    
    def load_tokenizer(self, model_id, scaling_factor, config):
        # TODO: In the config, we have the max_position_embeddings attribute 
        # that captures the original context size. When we load the tokenizer 
        # with the AutoTokenizer.from_pretrained method, 
        # we can use the model_max_length attribute to specify 
        # the context size as well. Load the tokenizer and specify the new 
        # context size by using the scaling factor. Make sure to specify 
        # the padding_size to be on the right.
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            config=config,
            padding_side="right"
        )
        config.max_position_embeddings = int(scaling_factor * config.max_position_embeddings)
        tokenizer.model_max_length = config.max_position_embeddings
        return tokenizer