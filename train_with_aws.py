from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

# Specify the SageMaker IAM role and output S3 path
role = "arn:aws:iam::108830828338:role/SageMakerFullAccess"
output_path = "s3://dedireformer/"

# Configure TensorBoard output
tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=output_path
)

# Create PyTorch Estimator
estimator = PyTorch(
    entry_point="train.py",  # Specify the entry point script
    source_dir=".",  # Specify the source directory (current directory)
    role=role,
    framework_version="1.12",  # Adjust to your PyTorch version
    py_version="py38",  # Specify the Python version
    instance_count=1,  # Number of instances for training
    instance_type="ml.p3.2xlarge",  # Instance type for training
    output_path=output_path,  # S3 path for output artifacts
    tensorboard_output_config=tensorboard_output_config,  # TensorBoard configuration
    hyperparameters={
        "model-type": "vir",  # Required argument for train_launcher.py
        "learning-rate": 0.001,  # Matches '--learning-rate'
        "epochs": 100,  # Matches '--epochs'
        "batch-size": 128,  # Matches '--batch-size'
        "tensorboard-dir": "/opt/ml/output/tensorboard/",  # Save TensorBoard logs
        "model-dir": "/opt/ml/model",  # Model directory
    },
)

# Start training
estimator.fit()
