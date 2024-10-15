from evaluation.evaluate import Evaluator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--training-type",
    "-t",
    type=str,
    help="Training type ('base', 'supervised', 'ppo', 'dpo', 'orpo')",
    required=True
)
args = parser.parse_args()

def run(training_type):
    # TODO: implement the right model_id and result_file
    model_id = None
    result_file = "results"

    if training_type == 'base':
        model_id = "openai-community/gpt2"
    elif training_type == 'supervised':
        model_id = "gulmert89/HW2-supervised"
    elif training_type == 'ppo':
        model_id = "gulmert89/HW2-ppo"
    elif training_type == 'dpo':
        model_id = "gulmert89/HW2-dpo"
    elif training_type == 'orpo':
        model_id = "gulmert89/HW2-orpo"
    else: 
        raise NotImplemented
    
    Evaluator.run(
        model_id=model_id,
        result_file=result_file,
        device="cuda"
    )

if __name__ == '__main__':
    print("Evaluation has started.")
    run(args.training_type)
    print("Evaluation has ended.")