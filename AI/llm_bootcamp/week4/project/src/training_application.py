from data_processing import DataCollatorForSupervisedDataset, DataProcessor
from model_loading import ModelLoader
from data_loading import DataLoader
from training import LoRATrainer


GROUP_SIZE_RATIO = 1/4

def run():
    # TODO: implement:
    # - get the data
    unprocessed_data = DataLoader.get_data()
    # - load the model
    model_init = ModelLoader()
    model, tokenizer = model_init.load_and_prepare_model()
    # - get the data collator
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        group_size_ratio=GROUP_SIZE_RATIO
    )
    # - get the trainer
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        group_size_ratio=GROUP_SIZE_RATIO
    )
    # - process the data
    data_processor = DataProcessor(tokenizer=tokenizer)
    processed_data = data_processor.transform(dataset=unprocessed_data)
    # - train the model
    trainer.train(tokenized_data=processed_data)
    print("Training has ended.")

if __name__ == '__main__':
    run()
    # Google Colab link:
    # https://colab.research.google.com/drive/1CvOqhUVw-WljsIv9ngXFe6W5nTQMt3Tb?usp=sharing
    # This notebook gives the error below as of 16 Sep 2024
    # Traceback (most recent call last):
    # File ".../src/training_application.py", line 36, in <module>
    #     run()
    # File ".../src/training_application.py", line 15, in run
    #     model, tokenizer = model_init.load_and_prepare_model()
    # File ".../src/model_loading.py", line 47, in load_and_prepare_model
    #     model = self.modify_attention(model=model)
    # File ".../src/model_loading.py", line 103, in modify_attention
    #     sparse_attn.load_state_dict(state_dict=current_attn.state_dict())
    # File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 2215, in load_state_dict
    #     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    # RuntimeError: Error(s) in loading state_dict for SparseAttention:
    #     Unexpected key(s) in state_dict: "q_proj.weight.absmax", "q_proj.weight.quant_map", "q_proj.weight.nested_absmax", "q_proj.weight.nested_quant_map", "q_proj.weight.quant_state.bitsandbytes__nf4", "k_proj.weight.absmax", "k_proj.weight.quant_map", "k_proj.weight.nested_absmax", "k_proj.weight.nested_quant_map", "k_proj.weight.quant_state.bitsandbytes__nf4", "v_proj.weight.absmax", "v_proj.weight.quant_map", "v_proj.weight.nested_absmax", "v_proj.weight.nested_quant_map", "v_proj.weight.quant_state.bitsandbytes__nf4", "o_proj.weight.absmax", "o_proj.weight.quant_map", "o_proj.weight.nested_absmax", "o_proj.weight.nested_quant_map", "o_proj.weight.quant_state.bitsandbytes__nf4". 
    #     size mismatch for q_proj.weight: copying a param with shape torch.Size([8388608, 1]) from checkpoint, the shape in current model is torch.Size([4096, 4096]).
    #     size mismatch for k_proj.weight: copying a param with shape torch.Size([2097152, 1]) from checkpoint, the shape in current model is torch.Size([1024, 4096]).
    #     size mismatch for v_proj.weight: copying a param with shape torch.Size([2097152, 1]) from checkpoint, the shape in current model is torch.Size([1024, 4096]).
    #     size mismatch for o_proj.weight: copying a param with shape torch.Size([8388608, 1]) from checkpoint, the shape in current model is torch.Size([4096, 4096]).
