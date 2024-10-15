from huggingface_hub import login
from evaluation.evaluate import Evaluator

# TODO: replace by yours
HUGGINGFACE_TOKEN = "hf_IdeLEteDThet0kEn5orrY"
REPO_NAME = 'project2-repo'


class BaseTrainer:

    def __init__(self, model, tokenizer, num_epoch=3, batch_size=8, output_dir='', result_file=''):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.result_file = result_file

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub()

    def evaluate(self):
        Evaluator.run(
            model_id='{}/{}'.format(REPO_NAME, self.output_dir),
            result_file=self.result_file
        )