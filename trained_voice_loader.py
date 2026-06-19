#trained_voice_loader.py
import json
import torch
import numpy as np
from pathlib import Path

class TrainedVoiceLoader:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.speakers = self._load_speakers()
        
    def _load_speakers(self):
        speakers_file = self.model_path / "speakers.json"
        with open(speakers_file, "r") as f:
            return json.load(f)
        
    def get_voice_embedding(self, voice_name=None):
        """Get embedding by name or return the first one if name is None"""
        if voice_name:
            for speaker in self.speakers:
                if speaker['name'] == voice_name:
                    return np.array(speaker['embedding'])
        # Return first voice if no match or no name specified
        return np.array(self.speakers[0]['embedding'])
    
    def get_available_voices(self):
        """Return a list of available voice names"""
        return [speaker['name'] for speaker in self.speakers]