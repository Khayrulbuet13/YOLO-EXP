from comet_ml import start
from comet_ml.integration.pytorch import log_model
import os
import json
import torch
from torchsummary import summary
import io
import argparse


class CometLogger:
    """
    Callback class for logging to Comet ML.
    Handles logging of hyperparameters, metrics, model architecture, and images.
    """
    def __init__(self, args, params):
        """
        Initialize Comet ML experiment and log hyperparameters.
        
        Args:
            args: Command line arguments
            params: Training parameters from YAML
        """
        # Store args for later use
        self.args = args
        
        # Load Comet ML settings from secrets.json
        secrets_path = os.path.join(os.path.dirname(__file__), 'secrets.json')
        try:
            with open(secrets_path, 'r') as f:
                secrets = json.load(f)
                api_key = secrets.get('comet_api_key')
                workspace = secrets.get('comet_workspace')
                project_name = secrets.get('comet_project_name', 'yoloexp')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load Comet ML settings from secrets.json: {e}")
            print("Using environment variables or defaults instead.")
            api_key = None  # Will use COMET_API_KEY environment variable
            workspace = None
            project_name = "yoloexp"
        
        # Start experiment
        self.experiment = start(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace
        )
        
        # Set experiment name based on save path
        self.experiment.set_name(f"YOLO-EXP-{args.save_path.split('/')[-1]}")
        
        # Log hyperparameters
        hyper_params = {
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "world_size": args.world_size,
        }
        
        # Add parameters from YAML
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                hyper_params[key] = value
        
        self.experiment.log_parameters(hyper_params)
        
        # Log YAML config file as artifact
        self.experiment.log_asset(args.yaml_file, file_name="config.yaml")
        
    def log_model_summary(self):
        """
        Log model summary file from the results directory.
        """
        # Check if model summary file exists in the save path
        summary_path = os.path.join(self.args.save_path, 'model_summary.txt')
        if os.path.exists(summary_path):
            try:
                # Read the summary file
                with open(summary_path, 'r') as f:
                    summary_text = f.read()
                
                # Log the summary text to Comet
                self.experiment.log_text(summary_text, metadata={"type": "model_summary"})
                
                # Log the file as an asset
                self.experiment.log_asset(summary_path, file_name="model_summary.txt")
                
                print(f"Successfully logged model summary from {summary_path}")
            except Exception as e:
                print(f"Failed to log model summary: {e}")
        else:
            print(f"Warning: Model summary file not found at {summary_path}")
    
    def log_batch_metrics(self, loss, lr, epoch, batch_idx, num_batches):
        """
        Log batch-level metrics during training.
        
        Args:
            loss: Training loss value
            lr: Current learning rate
            epoch: Current epoch
            batch_idx: Current batch index
            num_batches: Total number of batches per epoch
        """
        step = epoch * num_batches + batch_idx
        self.experiment.log_metrics({
            "train/loss": loss,
            "train/learning_rate": lr
        }, step=step)
    
    def log_epoch_metrics(self, epoch, map50, mean_ap, precision, recall, f1, phase="val"):
        """
        Log epoch-level metrics after validation or training.
        
        Args:
            epoch: Current epoch
            map50: mAP at IoU threshold 0.5
            mean_ap: Mean Average Precision across IoU thresholds
            precision: Precision value
            recall: Recall value
            f1: F1 score
            phase: Phase of logging, either 'train' or 'val'
        """
        self.experiment.log_metrics({
            f"{phase}/mAP@50": map50,
            f"{phase}/mAP": mean_ap,
            f"{phase}/precision": precision,
            f"{phase}/recall": recall,
            f"{phase}/f1": f1
        }, epoch=epoch)


    def log_image(self, image_path, name=None):
        """
        Log image to Comet ML.
        
        Args:
            image_path: Path to the image file
            name: Optional name for the image
        """
        self.experiment.log_image(image_path, name=name)
    
    def log_images_from_dir(self, dir_path, prefix="result"):
        """
        Log all images from a directory.
        
        Args:
            dir_path: Directory containing images
            prefix: Optional prefix to filter images
        """
        for filename in os.listdir(dir_path):
            if filename.startswith(prefix) and (filename.endswith('.png') or 
                                               filename.endswith('.jpg') or 
                                               filename.endswith('.jpeg')):
                image_path = os.path.join(dir_path, filename)
                self.experiment.log_image(image_path, name=filename)
    
    def log_source_code(self, directory):
        """
        Log source code files from a directory to Comet ML.
        
        Args:
            directory: Directory containing source code files
        """
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return
            
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and filepath.endswith('.py'):
                try:
                    # Log the file as an asset
                    self.experiment.log_asset(filepath, file_name=f"source/{filename}")
                    
                    # Also log the content as text for easier viewing
                    with open(filepath, 'r') as f:
                        content = f.read()
                    self.experiment.log_text(content, metadata={
                        "type": "source_code",
                        "filename": filename
                    })
                except Exception as e:
                    print(f"Failed to log source code for {filename}: {e}")
    
    def end(self):
        """End the experiment."""
        self.experiment.end()
