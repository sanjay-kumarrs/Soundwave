import os
import torch
import logging
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig

def create_xtts_config(output_path, device_type="cuda"):
    """Create a properly structured XttsConfig object"""
    config = XttsConfig()
    
    # Basic config
    config.model = "xtts"
    config.run_name = "custom_voice"
    config.run_description = "Custom voice training"
    config.epochs = 1000
    config.batch_size = 32 if device_type == "cuda" else 8
    config.eval_batch_size = 16 if device_type == "cuda" else 4
    config.mixed_precision = device_type == "cuda"
    config.run_eval = True
    config.test_delay_epochs = -1
    config.text_cleaner = "multilingual_cleaners"
    config.use_phonemes = True
    config.phoneme_language = "en"
    config.phoneme_cache_path = str(output_path / "phoneme_cache")
    config.precompute_num_workers = os.cpu_count() if device_type == "cuda" else 2
    
    # Model args - structure that works with XttsConfig
    config.model_args = {
        "use_speaker_embedding": True,
        "speaker_embedding_channels": 512,
        "use_d_vector_file": True,
        "d_vector_file": str(output_path / "speakers.json"),
        "use_memory_efficient_trainer": True
    }
    
    # Optimization config
    config.optimizer = "AdamW"
    config.optimizer_params = {
        "betas": [0.9, 0.999],
        "eps": 1e-7,
        "weight_decay": 0.01
    }
    config.lr = 0.0003
    
    # Add scheduler config
    config.scheduler = "NoamLR"
    config.scheduler_params = {
        "warmup_steps": 4000
    }
    
    return config

def main():
    """Fix the voice trainer module issues"""
    # Setup paths
    dataset_path = Path("D:/Main Project/rstt/dataset")
    output_path = Path("./trained_model_output")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Check CUDA
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device_type}")
    
    # Create proper config
    config = create_xtts_config(output_path, device_type)
    
    # Save config for reference
    config.save_json(output_path / "generated_config.json")
    logging.info(f"Created and saved proper XTTS config to {output_path / 'generated_config.json'}")
    
    # Print instructions
    print("\n" + "="*80)
    print("INSTRUCTIONS:")
    print("1. Please modify your voice_trainer_module.py to use this config structure")
    print("2. Edit your initialize_speaker_manager() method to use standard settings:")
    print("""
    def initialize_speaker_manager(self):
        ""Initialize the speaker manager with CUDA support""
        logging.info("Initializing speaker manager...")
        self.speaker_manager = SpeakerManager(
            encoder_model_path="speechbrain/spkrec-xvect-voxceleb",
            encoder_model_dim=512,
            use_cuda=self.device.type == "cuda"
        )
        if self.device.type == "cuda":
            self.speaker_manager.encoder = self.speaker_manager.encoder.to(self.device)
    """)
    print("=" * 80)

if __name__ == "__main__":
    main()