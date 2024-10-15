from data_processing import DataProcessor
from data_connector import DataConnector
from training import Trainer
from model import Model


MODEL_ID = "gpt2"
DATA_PATH = "dair-ai/emotion"
DATA_NAME = None  # "20231101.en"
BATCH_SIZE = 32
NUM_EPOCHS = 2
MAX_LENGTH = 64
DATA_PERCENTAGE = 10
PUSH_TO_HUB = False

def run():
    data = DataConnector.get_data(
        data_path=DATA_PATH,
        dataset_name=DATA_NAME,
        split_perc=DATA_PERCENTAGE
    )
    print("--- Data has been fetched. ---")

    num_labels = len(data["train"].features["label"].names)
    model, tokenizer = Model.get_model(MODEL_ID, num_labels)
    print("--- Model has been loaded. ---")

    data_processor = DataProcessor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        push_to_hub=PUSH_TO_HUB
    )
    tokenized_data = data_processor.transform(data)
    print("--- Data has been processed. ---")

    trainer.train(tokenized_data)
    print("--- Training is finished. ---")

if __name__ == "__main__":
    run()
