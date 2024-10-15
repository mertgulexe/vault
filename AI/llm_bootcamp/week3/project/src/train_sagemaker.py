from sagemaker.huggingface import HuggingFace
from dotenv import load_dotenv
import os

# TODO: Provide all the necessary information to the HuggingFace estimator to get it to run on Sagemaker:
# - Specify the source_dir. This is where the file that is running the training application is located.
# - Specify the entry_point. This is the file where is running the training application.
# - Specify the instance_type. Make sure you have requested enough quotas to use a specific machine (https://console.aws.amazon.com/servicequotas/home). Personally, I was using ml.p3.2xlarge, but there are cheaper machines. Make sure to check the cost of those (https://aws.amazon.com/sagemaker/pricing/)!!!
# - Specify the role. Check HW2, to generate a role.
# - Specify the image_uri. You can find the different Huggingface training containers 
# here: https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers. 
# Personally, I tend to use the following because it works: 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04
# - Specify the dependencies. You will most likely need two dependency files, 
# requirements.txt to install the additional packages not available in the docker image 
# and that your code may need, and the .env file where you can store the environment variables. 
# You can use the load_dotenv function to load the environment variables. 
# You will need to have your Huggingface and your Weight and Bias tokens.

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
ROLE = os.getenv("SAGEMAKER_ROLE")

huggingface_estimator = HuggingFace(
        source_dir=".",
        entry_point="training_application.py",
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        role=ROLE,
        dependencies=["../requirements.txt"],
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04",
        py_version="py310",  # Required unless ``image_uri`` is provided. (If ``image_uri`` is provided, set it to `None`, don't comment it out.)
        hyperparameters={
            "training_type": "accelerated",  # see the `args` in `training_application.py`
            "batch_size": 1024,
            "num_epochs": 8,
            "split_percentage": 100,
            "push_to_hub": True
        },
        environment={
            "HF_TOKEN": HUGGINGFACE_TOKEN,
            "WANDB_API_KEY": WANDB_API_KEY
        }
)
huggingface_estimator.fit()

# Mert: You can get other containers from:
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-training-containers
