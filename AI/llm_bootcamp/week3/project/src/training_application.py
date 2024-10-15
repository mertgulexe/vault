# environment imports
from dotenv import load_dotenv
import argparse
import os

# model and data imports
from data.data_connecting import DataConnector
from data.data_processing import DataProcessor
from ml.training import BasicTrainer, AcceleratedTrainer
from ml.model import Model


# model and data configs
load_dotenv()  # loads environment variables
MODEL_ID = "gpt2"
DATA_PATH = "dair-ai/emotion"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

# argument configs
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--training-type",
    "-tt",
    default="normal",
    type=str,
    help="Choose between 'accelerated' or 'normal' training.",
    required=False
)
arg_parser.add_argument(
    "--batch-size",
    "-bs",
    default=16,
    type=int,
    help="Batch size (int)",
    required=False
)
arg_parser.add_argument(
    "--num-epochs",
    "-epoch",
    default=2,
    type=int,
    help="Number of epochs to run (int).",
    required=False
)
arg_parser.add_argument(
    "--split-percentage",
    "-split",
    default=20,
    type=int,
    help="Dataset split percentage (int).",
    required=False
)
arg_parser.add_argument(
    "--push-to-hub",
    action=argparse.BooleanOptionalAction,  # "store_true",
    default=False,
    # type=bool,
    help="Push the model file to HuggingFace hub or not.",
    required=False
)
args = arg_parser.parse_args()

def run():
    # TODO: let's implement the main run function:
    # - Get the data
    # - Get the model
    # - Process the data
    # - Train the model with the data
    # - Save the model
    TrainingClass = BasicTrainer
    REPO_NAME = "project3-supervised-model"
    training_type = args.training_type

    if training_type == "accelerated":
        TrainingClass = AcceleratedTrainer
        REPO_NAME = "project3-supervised-distributed-model"
    elif training_type == "normal":
        pass
    else:
        raise NotImplemented(
            "No other training type beside 'accelerated' and 'normal' "
            "is implemented."
        )
    data_connector = DataConnector()
    unprocessed_data = data_connector.get_data(
        data_path=DATA_PATH,
        split_perc=args.split_percentage
    )
    model_obj = Model(
        model_id=MODEL_ID,
        num_labels=len(unprocessed_data["train"].features["label"].names)
    )
    model, tokenizer = model_obj.model, model_obj.tokenizer
    data_processor = DataProcessor(tokenizer=tokenizer, max_length=256)
    processed_data = data_processor.transform(data=unprocessed_data)
    trainer = TrainingClass(
        model=model,
        tokenizer=tokenizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hf_token=HUGGINGFACE_TOKEN,
        push_to_hub=args.push_to_hub,
        repo_name=REPO_NAME
    )
    trainer.train(tokenized_data=processed_data)


if __name__ == "__main__":
    # run(**vars(args))
    run()

    # You can run this file directly OR run this code on this working directory:
    # accelerate launch training_application.py --training-type "accelerated" 
