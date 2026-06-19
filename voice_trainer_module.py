#voice_trainer_module.py
import os
import time
import torch
import torchaudio
import logging
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from speechbrain.inference import SpeakerRecognition  # Updated import

class CUDAVoiceTrainer:
    def __init__(self, dataset_path, output_path, config_path=None, force_cpu=False):
        """
        Initialize the CUDA-enabled voice trainer
        dataset_path: Path to directory containing WAV files
        output_path: Path to save trained model and artifacts
        config_path: Optional path to custom YAML config file
        force_cpu: Force CPU usage even if CUDA is available
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.force_cpu = force_cpu
        self.progress_callback = None

        # Setup CUDA
        self.setup_cuda()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_path / 'training.log'),
                logging.StreamHandler()
            ]
        )

        self.setup_config(config_path)
        self.speaker_manager = None
        self.voice_samples = []
        self.embeddings = []

    def setup_cuda(self):
        """Setup CUDA environment"""
        if not self.force_cpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

            # Log CUDA information
            logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logging.info("CUDA not available. Using CPU.")

    def setup_config(self, config_path=None):
        """Setup training configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = {
                "model": "xtts",
                "epochs": 1000,
                "batch_size": 32 if self.device.type == "cuda" else 8,
                "eval_batch_size": 16 if self.device.type == "cuda" else 4,
                "use_phonemes": True,
                "phoneme_language": "en",
                "phoneme_cache_path": str(self.output_path / "phoneme_cache"),
                "precompute_num_workers": os.cpu_count() if self.device.type == "cuda" else 2,
                "optimizer": {
                    "name": "AdamW",
                    "params": {
                        "lr": 0.0003,
                        "weight_decay": 0.01
                    }
                },
                "cuda_optimization": {
                    "gradient_clip": 1.0,
                    "amp_level": 'O1' if self.device.type == "cuda" else None
                }
            }

        self.config = config_dict

    def initialize_speaker_manager(self):
        """Initialize the SpeakerRecognition model manually"""
        logging.info("Initializing SpeakerRecognition manually...")

        try:
            self.speaker_manager = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                run_opts={"device": self.device}
            )
            logging.info("SpeakerRecognition initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing SpeakerRecognition: {e}")
            raise ValueError("SpeakerManager encoder initialization failed.")

    def process_audio_files(self):
        """Process all audio files"""
        logging.info("Processing audio files...")
        audio_files = list(self.dataset_path.glob("*.wav"))

        for audio_file in tqdm(audio_files):
            try:
                waveform, sample_rate = torchaudio.load(str(audio_file))

                if sample_rate != 22050:
                    resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                self.voice_samples.append({
                    'audio': waveform.numpy().flatten(),
                    'sr': 22050,
                    'name': audio_file.stem,
                    'path': str(audio_file)
                })

            except Exception as e:
                logging.error(f"Error processing {audio_file.name}: {str(e)}")

        logging.info(f"Processed {len(self.voice_samples)} valid audio files")

    def extract_embeddings(self):
        """Extract speaker embeddings"""
        logging.info("Extracting speaker embeddings...")

        for sample in tqdm(self.voice_samples):
            try:
                embedding = self.speaker_manager.encode_batch(torch.tensor(sample['audio']).unsqueeze(0))
                self.embeddings.append({
                    'name': sample['name'],
                    'embedding': embedding.cpu().numpy().tolist(),
                    'path': sample['path']
                })
            except Exception as e:
                logging.error(f"Error extracting embedding for {sample['name']}: {str(e)}")
    
    def set_progress_callback(self, callback):
        """
        Set a callback function to report training progress
        The callback should accept (current_epoch, total_epochs, loss)
        """
        self.progress_callback = callback

    def train(self, epochs=None, progress_callback=None):
        """Run the training process"""
        try:
            # Set progress callback if provided
            if progress_callback:
                self.progress_callback = progress_callback
                
            logging.info(f"Starting training pipeline on {self.device.type.upper()}...")

            self.initialize_speaker_manager()
            self.process_audio_files()

            if not self.voice_samples:
                raise ValueError("No valid audio samples found")

            self.extract_embeddings()

            if not self.embeddings:
                raise ValueError("No embeddings could be extracted")

            if epochs:
                self.config["epochs"] = epochs
                
            # Simulate training epochs for demonstration
            # In a real implementation, you'd call your actual training code here
            total_epochs = self.config["epochs"]
            
            # Example of calling progress callback during training
            for epoch in range(total_epochs):
                # Simulate some work
                time.sleep(0.1)  # Remove in production, just for simulation
                
                # Calculate simulated loss (would be real loss in actual implementation)
                simulated_loss = 1.0 - (epoch / total_epochs) * 0.7
                
                # Log progress
                if epoch % 10 == 0 or epoch == total_epochs - 1:
                    logging.info(f"Epoch {epoch+1}/{total_epochs}, Loss: {simulated_loss:.4f}")
                
                # Call progress callback if set
                if self.progress_callback:
                    self.progress_callback(epoch + 1, total_epochs, simulated_loss)

            self.save_training_artifacts()
            logging.info(f"Training complete. Model trained for {self.config['epochs']} epochs.")

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
        
    def save_training_artifacts(self):
        """Save speaker embeddings and other artifacts"""
        logging.info("Saving training artifacts...")

        speakers_file = self.output_path / "speakers.json"
        with open(speakers_file, "w") as f:
            json.dump(self.embeddings, f)

        logging.info(f"Saved speaker embeddings for {len(self.embeddings)} voices to {speakers_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a voice model with CUDA support")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    trainer = CUDAVoiceTrainer(
        dataset_path=args.dataset,
        output_path=args.output,
        force_cpu=args.cpu
    )
    trainer.train(args.epochs)
