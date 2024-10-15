from sagemaker.huggingface import HuggingFace
from training.base_train import HUGGINGFACE_TOKEN

role = 'YOUR SAGEMAKER ROLE'

huggingface_estimator = HuggingFace(
        entry_point='training_application.py',
        source_dir='.',
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        dependencies=['./requirements.txt'],
        image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
        py_version='py310',
        environment={"HF_TOKEN": HUGGINGFACE_TOKEN}
)

huggingface_estimator.fit()