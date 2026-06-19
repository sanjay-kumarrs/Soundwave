#-----------------------------------------------
# Real Time Speech Predictor with VC - Improved
#----------------------------------------------
#rtvctrained.py
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import queue
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from threading import Thread
from pynput import keyboard
from TTS.api import TTS
import time
import os
import scipy.io.wavfile as wavfile
# Import the trained voice loader
from trained_voice_loader import TrainedVoiceLoader

class RealtimeSpeechPredictorWithCloning:
    def __init__(self, model_path, trained_voice_path=None, device_index=None):
        """Initialize the system with voice cloning capabilities"""
        self.model = Model(model_path)
        
        # Load GPT-2
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Initialize TTS with YourTTS model
        print("Loading TTS model...")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=torch.cuda.is_available())
        
        # Voice cloning parameters
        self.reference_audio = None
        self.reference_audio_path = None
        
        # Load trained voice if provided
        self.trained_voice_loader = None
        self.current_voice_name = None  # Track selected voice name
        self.voice_embedding = None  # Store the current voice embedding
        self.is_using_trained_voice = False
        
        if trained_voice_path and os.path.exists(trained_voice_path):
            try:
                print(f"Loading trained voices from: {trained_voice_path}")
                self.trained_voice_loader = TrainedVoiceLoader(trained_voice_path)
                available_voices = self.trained_voice_loader.get_available_voices()
                if available_voices:
                    self.current_voice_name = available_voices[0]
                    self.is_using_trained_voice = True
                    # Pre-load the voice embedding
                    self.voice_embedding = self.trained_voice_loader.get_voice_embedding(self.current_voice_name)
                    print(f"Available trained voices: {', '.join(available_voices)}")
                    print(f"Using voice: {self.current_voice_name}")
                else:
                    print("No trained voices found.")
            except Exception as e:
                print(f"Error loading trained voices: {str(e)}")
                print(f"Full error details: {repr(e)}")
        
        # Audio parameters
        self.samplerate = 16000
        self.blocksize = 8000
        self.device_index = device_index
        
        # Recognition variables
        self.q = queue.Queue()
        self.recognizer = KaldiRecognizer(self.model, self.samplerate)
        self.current_sentence = ""
        self.prediction_requested = False
        
        # Performance tracking
        self.last_prediction_time = 0

    def record_reference_voice(self, duration=30):
        """Record reference voice for cloning"""
        print(f"\n🎙️ Recording {duration} seconds of reference voice...")
        audio = sd.rec(int(duration * self.samplerate), 
                      samplerate=self.samplerate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        self.reference_audio = audio.flatten()
        
        # Save reference audio to a temporary WAV file
        self.reference_audio_path = "reference_voice.wav"
        wavfile.write(self.reference_audio_path, self.samplerate, self.reference_audio)
        print("✅ Reference voice recorded and saved!")
        self.is_using_trained_voice = False

    def predict_next_word(self, input_text):
        """Predict the next word given an input text using GPT-2."""
        try:
            # Tokenize input text and prepare input tensors
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Predict logits for the next token
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Get the top predictions
            top_tokens = torch.topk(next_token_logits, 5, dim=-1).indices[0]
            
            # Get the top prediction
            predicted_word = self.tokenizer.decode([top_tokens[0].item()]).strip()
            
            # Also get alternative predictions for display
            alt_predictions = [self.tokenizer.decode([token.item()]).strip() for token in top_tokens[1:]]
            
            return predicted_word, alt_predictions
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "", []

    def speak_text(self, text):
        """Speak text using appropriate voice method"""
        start_time = time.time()
        
        try:
            if self.is_using_trained_voice and self.trained_voice_loader:
                self.speak_text_with_trained_voice(text)
            else:
                if self.reference_audio_path is None:
                    raise ValueError("No reference audio available. Record reference voice first.")
                    
                # Generate audio with cloned voice
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.reference_audio_path,
                    language="en",
                )
                
                # Play the generated audio
                sd.play(np.array(audio), self.tts.synthesizer.output_sample_rate)
                sd.wait()
                
            processing_time = time.time() - start_time
            print(f"Speech generation took {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error in speech synthesis: {str(e)}")
            print("Falling back to default TTS...")
            try:
                audio = self.tts.tts(text=text, language="en")
                sd.play(np.array(audio), self.tts.synthesizer.output_sample_rate)
                sd.wait()
            except Exception as fallback_e:
                print(f"Fallback TTS also failed: {str(fallback_e)}")
    
    def speak_text_with_trained_voice(self, text):
        """Speak text using trained voice embeddings"""
        if not self.is_using_trained_voice or not self.trained_voice_loader:
            raise ValueError("No trained voice available")
        
        try:
            # Find the reference path for the selected voice
            reference_file = None
            
            # Search for the voice in speakers.json
            for speaker in self.trained_voice_loader.speakers:
                if speaker['name'] == self.current_voice_name:
                    reference_file = speaker.get('path')
                    break
            
            if not reference_file or not os.path.exists(reference_file):
                print(f"Reference file for voice {self.current_voice_name} not found or doesn't exist")
                print(f"Using fallback method with available reference file...")
                
                # Get any available reference file
                all_voices = self.trained_voice_loader.get_available_voices()
                for voice in all_voices:
                    for speaker in self.trained_voice_loader.speakers:
                        if speaker['name'] == voice and 'path' in speaker:
                            if os.path.exists(speaker['path']):
                                reference_file = speaker['path']
                                print(f"Using alternate reference file from voice: {voice}")
                                break
                    if reference_file:
                        break
            
            # If we still don't have a valid reference file, raise an error
            if not reference_file or not os.path.exists(reference_file):
                raise ValueError("No valid reference files found for any trained voice")
                
            print(f"Generating speech with voice: {self.current_voice_name}")
            print(f"Using reference file: {reference_file}")
            
            # Generate audio with the selected trained voice
            audio = self.tts.tts(
                text=text,
                speaker_wav=reference_file,
                language="en",
            )
            
            # Play the generated audio
            sd.play(np.array(audio), self.tts.synthesizer.output_sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"Error generating speech with trained voice: {str(e)}")
            # Provide more detailed error information
            import traceback
            traceback.print_exc()
            
            # Fallback to standard synthesis if there's an error
            print("Falling back to default TTS...")
            audio = self.tts.tts(text=text, language="en")
            sd.play(np.array(audio), self.tts.synthesizer.output_sample_rate)
            sd.wait()

    def switch_voice(self, voice_name):
        """Switch to a different trained voice"""
        if not self.trained_voice_loader:
            print("No trained voices available")
            return False
            
        available_voices = self.trained_voice_loader.get_available_voices()
        if voice_name in available_voices:
            self.current_voice_name = voice_name
            self.voice_embedding = self.trained_voice_loader.get_voice_embedding(voice_name)
            self.is_using_trained_voice = True
            print(f"Switched to voice: {voice_name}")
            return True
        else:
            print(f"Voice '{voice_name}' not found. Available voices: {', '.join(available_voices)}")
            return False

    def on_press(self, key):
        """Handle keyboard input"""
        try:
            if key == keyboard.Key.enter:
                self.prediction_requested = True
            elif key == keyboard.Key.space and self.prediction_requested:
                # Space during prediction allows regenerating prediction
                self.prediction_requested = True
            # Add voice switching capability with number keys
            elif hasattr(key, 'char') and self.trained_voice_loader:
                try:
                    key_num = int(key.char)
                    available_voices = self.trained_voice_loader.get_available_voices()
                    if 1 <= key_num <= len(available_voices):
                        self.current_voice_name = available_voices[key_num-1]
                        self.voice_embedding = self.trained_voice_loader.get_voice_embedding(self.current_voice_name)
                        print(f"\n🎤 Switched to voice: {self.current_voice_name}")
                except (ValueError, IndexError):
                    pass
        except AttributeError:
            pass

    def callback(self, indata, frames, time, status):
        """Callback function for the audio stream"""
        if status:
            print(status)
        self.q.put(bytes(indata))
        
    def process_audio(self):
        """Process audio from the queue and perform recognition"""
        try:
            data = self.q.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    return text, True
                return "", True
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "")
                return text, False
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return "", False

    def start_listening(self):
        """Start the real-time STT system with prediction and TTS integration"""
        try:
            # Start keyboard listener in a separate thread
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()

            with sd.RawInputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                device=self.device_index,
                dtype="int16",
                channels=1,
                callback=self.callback
            ):
                print("\n🎤 Started listening... Speak and press Enter when you want a prediction!")
                
                # Additional instructions for voice switching
                if self.trained_voice_loader and self.trained_voice_loader.get_available_voices():
                    voices = self.trained_voice_loader.get_available_voices()
                    print("\n💡 Voice controls:")
                    for i, voice in enumerate(voices):
                        print(f"   Press {i+1}: Switch to voice '{voice}'")
                        
                print("\n💡 Speak your sentence, then press Enter to get the prediction\n")
                
                while True:
                    text, is_final = self.process_audio()
                    # Update current sentence
                    if text:
                        self.current_sentence = text
                        print(f"\r🎯 Current: {self.current_sentence}", end="", flush=True)
                    
                    # Handle prediction request
                    if self.prediction_requested and self.current_sentence:
                        # Don't process predictions too quickly (debounce)
                        current_time = time.time()
                        if current_time - self.last_prediction_time < 1.0:
                            self.prediction_requested = False
                            continue
                            
                        self.last_prediction_time = current_time
                        
                        # Get prediction and alternatives
                        predicted_word, alt_predictions = self.predict_next_word(self.current_sentence)
                        complete_sentence = f"{self.current_sentence} {predicted_word}"
                        print("\n\n✨ Complete sentence with prediction:")
                        print(f"→ {complete_sentence}")
                        
                        # Show alternative predictions
                        if alt_predictions:
                            print("\n🔄 Alternative predictions:")
                            for i, alt in enumerate(alt_predictions):
                                print(f"   {i+1}. {self.current_sentence} {alt}")
                        
                        # Speak the complete sentence
                        print("\n🔊 Speaking prediction...")
                        self.speak_text(complete_sentence)
                        
                        print("\n🎤 Listening for new sentence...")
                        self.current_sentence = ""
                        self.prediction_requested = False
                        
        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
            listener.stop()
        except Exception as e:
            print(f"❌ Error in audio stream: {str(e)}")
            import traceback
            traceback.print_exc()
            listener.stop()

def list_audio_devices():
    """List available audio devices"""
    devices = sd.query_devices()
    print("\n📋 Available input devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"   Index {i}: {device['name']} (channels: {device['max_input_channels']})")
    print()

if __name__ == "__main__":
    print("🔊 Real-Time Speech Predictor with Voice Cloning 🎙️")
    print("====================================================")
    
    # Get model paths with error handling
    default_model_path = "D:/Main Project/rstt/vosk-model-small-en-us-0.15"
    model_path = input(f"Enter path to Vosk model [default: {default_model_path}]: ").strip()
    if not model_path:
        model_path = default_model_path
    
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model path {model_path} does not exist. Please check the path.")
        sys.exit(1)
        
    default_trained_voice_path = "./trained_model_output"
    trained_voice_path = input(f"Enter path to trained voice models [default: {default_trained_voice_path}]: ").strip()
    if not trained_voice_path:
        trained_voice_path = default_trained_voice_path
    
    if not os.path.exists(trained_voice_path):
        print(f"⚠️ Warning: Trained voice path {trained_voice_path} does not exist.")
        trained_voice_path = None
    
    # List audio devices
    list_audio_devices()
    
    # Select audio device
    device_index = None
    device_input = input("Select input device index (leave blank for default): ").strip()
    if device_input:
        try:
            device_index = int(device_input)
        except ValueError:
            print("Invalid device index, using default.")
    
    # Choose option: trained model or live recording
    use_trained = input("Use trained voice model? (y/n) [default: y]: ").lower().strip()
    use_trained = use_trained != 'n'  # Default to yes if empty or not 'n'
    
    try:
        if use_trained and trained_voice_path:
            # Initiate with trained voice path
            print("\n🔄 Initializing with trained voice models...")
            predictor = RealtimeSpeechPredictorWithCloning(
                model_path=model_path,
                trained_voice_path=trained_voice_path,
                device_index=device_index
            )
            predictor.start_listening()
        else:
            # Use live recording approach
            print("\n🔄 Initializing with live voice recording...")
            predictor = RealtimeSpeechPredictorWithCloning(
                model_path=model_path,
                device_index=device_index
            )
            predictor.record_reference_voice()  # Record reference audio
            predictor.start_listening()
    except Exception as e:
        print(f"❌ Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()