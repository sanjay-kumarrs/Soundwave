from flask import Flask, request, jsonify
import shutil
from datetime import datetime
import subprocess
import base64
import os
import tempfile
import soundfile as sf
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from TTS.api import TTS
from vosk import Model, KaldiRecognizer
import json
import time
import threading
import psutil
import shutil
from flask_socketio import SocketIO, emit
import pathlib
import logging
from werkzeug.utils import secure_filename
import re
import sched  # Import the sched module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Remove explicit async_mode to let Flask-SocketIO auto-detect the best mode
# simple-websocket package is used for WebSocket support with threading mode
socketio = SocketIO(app, cors_allowed_origins='*', ping_timeout=60, ping_interval=25)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'}), 200

# --- Path Configuration ---
BASE_DIR = pathlib.Path(__file__).parent.absolute()
VOSK_MODEL_DIR = BASE_DIR / "vosk-model-small-en-us-0.15"
TRAINED_VOICE_DIR = BASE_DIR / "trained_model_output"
DATASET_DIR = BASE_DIR / "dataset_samples"
TEMP_DIR = BASE_DIR / "temp_files"

TRAINED_VOICE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Initialization ---
logger.info("Loading models...")

# Load Vosk model
if not VOSK_MODEL_DIR.exists():
    logger.error(f"Vosk model path not found: {VOSK_MODEL_DIR}")
    vosk_model = None
else:
    try:
        vosk_model = Model(str(VOSK_MODEL_DIR))
        logger.info("Vosk model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Vosk model: {e}")
        vosk_model = None

# Load GPT-2 model
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    logger.info("GPT-2 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load GPT-2 model: {e}")
    gpt2_tokenizer = None
    gpt2_model = None

# Load TTS model
try:
    tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/your_tts",
        progress_bar=False,
        gpu=torch.cuda.is_available()
    )
    logger.info(f"TTS model loaded successfully (GPU: {torch.cuda.is_available()})")
except Exception as e:
    logger.error(f"Failed to load TTS model: {e}")
    tts_model = None

# --- Voice Management ---
available_voices = []
current_voice_name = None
reference_voice_map = {}

def validate_filename(filename):
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name or safe_name.startswith('.'):
        return None
    return safe_name

def get_secure_temp_dir():
    temp_dir = TEMP_DIR / f"session_{int(time.time())}_{os.getpid()}" # Add PID for more uniqueness
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def load_trained_voices():
    global available_voices, reference_voice_map
    available_voices = []
    reference_voice_map = {}

    logger.info("load_trained_voices() function called")
    logger.info(f"Loading trained voices from directory: {TRAINED_VOICE_DIR}")
    speakers_file = TRAINED_VOICE_DIR / "speakers.json"

    if speakers_file.exists():
        try:
            with open(speakers_file, 'r') as f:
                speakers_data = json.load(f)
                if isinstance(speakers_data, list):
                    speakers = speakers_data
                else:
                    logger.warning("speakers.json does not contain a list of speakers.")
                    speakers = []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding speakers.json: {e}")
            speakers = []

        unique_voice_names = set() # Use a set to track names easily
        for speaker in speakers:
            name = speaker.get('name')
            path = speaker.get('path')
            if name and path:
                logger.debug(f"Processing speaker: Name='{name}', Path='{path}'")
                ref_path = pathlib.Path(path)
                # Check if path is absolute or relative to TRAINED_VOICE_DIR
                if not ref_path.is_absolute():
                    ref_path = TRAINED_VOICE_DIR / name / ref_path.name # Assume path is relative to voice dir

                if ref_path.exists() and TRAINED_VOICE_DIR in ref_path.parents:
                    if name not in unique_voice_names:
                        available_voices.append(name)
                        reference_voice_map[name] = str(ref_path)
                        unique_voice_names.add(name)
                        logger.info(f"Voice '{name}' loaded successfully: {ref_path}")
                    else:
                        logger.warning(f"Duplicate voice name '{name}' found in speakers.json. Skipping.")
                else:
                    logger.warning(f"Reference audio for '{name}' not found or invalid path: '{ref_path}'")
            else:
                logger.warning(f"Speaker entry missing 'name' or 'path': {speaker}")

    else:
        logger.warning("speakers.json not found in trained_voice_path")

    logger.info(f"Available voices after loading: {available_voices}")


# --- Training Thread ---
class TrainingProcessThread(threading.Thread):
    def __init__(self, dataset_path, output_path, voice_name, epochs=50, use_cpu=False):
        threading.Thread.__init__(self)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.voice_name = voice_name
        self.epochs = epochs
        self.use_cpu = use_cpu
        self.success = False
        self.error = None
        self.daemon = True

    def run(self):
        try:
            from voice_trainer_module import CUDAVoiceTrainer

            voice_output_dir = TRAINED_VOICE_DIR / self.voice_name
            voice_output_dir.mkdir(exist_ok=True) # Should already exist, but double check
            self.output_path = str(voice_output_dir)

            trainer = CUDAVoiceTrainer(
                dataset_path=self.dataset_path,
                output_path=self.output_path,
                force_cpu=self.use_cpu
            )

            def progress_callback(current_epoch, total_epochs, loss=None):
                progress = round((current_epoch / total_epochs) * 100)
                status_msg = f"Training voice model '{self.voice_name}': {progress}% complete"
                if loss is not None:
                    status_msg += f" (loss: {loss:.4f})"

                socketio.emit('training_progress', {
                    'progress': progress,
                    'status': status_msg,
                    'voice_name': self.voice_name
                })
                logger.info(f"Training progress: {progress}% (loss: {loss if loss is not None else 'N/A'})")

            trainer.train(self.epochs, progress_callback=progress_callback)
            self.success = True

            logger.info(f"Emitting training_progress 100% for '{self.voice_name}'")
            socketio.emit('training_progress', {
                'progress': 100,
                'status': f"Training for '{self.voice_name}' complete!",
                'voice_name': self.voice_name
            })
            logger.info(f"Emitting training_complete for '{self.voice_name}'")
            socketio.emit('training_complete', {
                'success': True,
                'voice_name': self.voice_name,
                'status': f"Voice model '{self.voice_name}' trained successfully!"
            })
            logger.info(f"training_complete event emitted successfully for '{self.voice_name}'")

            # Update speakers.json
            speakers_file = TRAINED_VOICE_DIR / "speakers.json"
            if speakers_file.exists():
                try:
                    with open(speakers_file, 'r') as f:
                        speakers = json.load(f)

                    found = False
                    for speaker in speakers:
                        if speaker.get('name') == self.voice_name:
                            speaker['training_status'] = 'complete'
                            speaker['training_end'] = datetime.now().isoformat()
                            # Update reference path if needed (usually the first sample)
                            first_sample_path = next((p for p in voice_output_dir.glob("*.wav")), None)
                            if first_sample_path:
                                speaker['path'] = str(first_sample_path.relative_to(voice_output_dir)) # Store relative path
                            found = True
                            break

                    if not found: # Should not happen if entry was added before training
                         logger.warning(f"Could not find speaker '{self.voice_name}' in speakers.json to mark as complete.")

                    with open(speakers_file, 'w') as f:
                        json.dump(speakers, f, indent=4)

                    load_trained_voices() # Reload voices after update
                except Exception as e:
                    logger.error(f"Error updating speakers.json after training: {e}")
            else:
                 logger.error("speakers.json not found after training completion.")


        except Exception as e:
            self.error = str(e)
            logger.error(f"Training error for voice '{self.voice_name}': {e}", exc_info=True)
            socketio.emit('training_error', {
                'error': f"Training failed for '{self.voice_name}': {str(e)}",
                'voice_name': self.voice_name
            })


# --- STT Endpoint ---
@app.route('/stt', methods=['POST'])
def stt():
    if vosk_model is None:
        return jsonify({'error': 'Vosk model not loaded'}), 500

    audio_data_base64 = request.json.get('audio_data')
    if not audio_data_base64:
        return jsonify({'error': 'No audio data provided'}), 400

    temp_dir = None
    try:
        temp_dir = get_secure_temp_dir()
        request_id = f"{os.getpid()}_{int(time.time())}"
        tmp_audio_path = temp_dir / f"audio_{request_id}.wav"

        try:
            audio_data = base64.b64decode(audio_data_base64)
            with open(tmp_audio_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"Received audio of size: {len(audio_data)} bytes, saved to {tmp_audio_path}")
        except Exception as e:
            logger.error(f"Failed to decode or save audio: {e}")
            return jsonify({'error': 'Invalid audio data format'}), 400

        try:
            data, samplerate = sf.read(tmp_audio_path)
            if len(data.shape) > 1 and data.shape[1] > 1:
                 data = np.mean(data, axis=1) # Convert to mono by averaging channels
            if samplerate != 16000:
                try:
                    import librosa
                    data = librosa.resample(y=data.astype(np.float32), orig_sr=samplerate, target_sr=16000)
                    samplerate = 16000
                except ImportError:
                    logger.warning("librosa not installed, cannot resample audio. STT might be inaccurate.")
                except Exception as resample_e:
                     logger.error(f"Resampling failed: {resample_e}")
                     return jsonify({'error': 'Audio resampling failed'}), 500

            # Ensure data is in 16-bit PCM format if needed by Vosk
            if data.dtype != np.int16:
                 data = (data * 32767).astype(np.int16)

            sf.write(tmp_audio_path, data, 16000, subtype='PCM_16') # Ensure PCM_16 subtype
        except Exception as e:
            logger.error(f"Audio processing/resampling failed: {e}", exc_info=True)
            return jsonify({'error': 'Audio processing failed'}), 500

        recognizer = KaldiRecognizer(vosk_model, 16000)
        transcribed_text = ""

        with open(tmp_audio_path, 'rb') as wf:
            while True:
                data = wf.read(4096)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcribed_text += result.get("text", "") + " "
                # else:
                #     partial_result = json.loads(recognizer.PartialResult())
                #     logger.debug(f"Partial result: {partial_result.get('partial')}")

            final_result = json.loads(recognizer.FinalResult())
            transcribed_text += final_result.get("text", "")

        logger.info(f"Transcribed text: {transcribed_text.strip()}")
        return jsonify({'text': transcribed_text.strip()})

    except Exception as e:
        logger.error(f"STT Error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Speech-to-text processing failed', 'details': str(e)}), 500
    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


# --- Prediction Endpoint (REVISED & COMPLETE) ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives text input, uses GPT-2 to predict the top 3 most likely next words,
    and returns them as a JSON list.
    """
    if gpt2_model is None or gpt2_tokenizer is None:
        logger.error("Predict endpoint called but GPT-2 model/tokenizer not loaded.")
        return jsonify({'error': 'GPT-2 model not loaded'}), 503 # Service Unavailable

    text = request.json.get('text')
    if not text or not isinstance(text, str):
        logger.warning(f"Predict endpoint called with invalid text input: {text}")
        return jsonify({'error': 'No valid text provided'}), 400

    logger.info(f"Received prediction request for text: '{text}'")

    try:
        # Tokenize the input text
        input_ids = gpt2_tokenizer.encode(text, return_tensors="pt")

        # Ensure model and inputs are on the correct device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        gpt2_model.to(device) # Ensure model is on the device

        # Get model predictions without calculating gradients
        with torch.no_grad():
            outputs = gpt2_model(input_ids)
            # Logits for the *next* token prediction (shape: [batch_size, vocab_size])
            next_token_logits = outputs.logits[:, -1, :]

        # Get the top K token IDs and their probabilities/logits (we only need IDs here)
        top_k = 3
        # Use softmax to get probabilities if needed, but topk works on logits directly
        # probs = torch.softmax(next_token_logits, dim=-1)
        top_k_results = torch.topk(next_token_logits, k=top_k, dim=-1)

        predicted_token_ids = top_k_results.indices[0].tolist() # Get list of top K token IDs

        # Decode the token IDs into words safely
        predicted_words = [] # Initialize as empty list for safety
        for token_id in predicted_token_ids:
            try:
                # Decode the single token ID
                # Use clean_up_tokenization_spaces=True to handle spacing better
                decoded_word = gpt2_tokenizer.decode([token_id], clean_up_tokenization_spaces=True).strip()

                # Add only if the decoded word is not empty after stripping
                if decoded_word:
                    predicted_words.append(decoded_word)
                else:
                    logger.debug(f"Decoded token ID {token_id} resulted in an empty string.")

            except Exception as decode_e:
                # Log if decoding a specific token fails, but continue with others
                logger.warning(f"Could not decode token ID {token_id}: {decode_e}")

        # Safety check: Ensure predicted_words is definitely a list
        if not isinstance(predicted_words, list):
             logger.error(f"Critical error: predicted_words is not a list after decoding! Type: {type(predicted_words)}, Value: {predicted_words}. Defaulting to empty list.")
             predicted_words = []

        # Optional: Add a placeholder if decoding resulted in an empty list?
        # if not predicted_words and predicted_token_ids: # Check if we had tokens but failed to decode all
        #     logger.warning("Decoding resulted in an empty list, adding placeholder.")
        #     predicted_words = ["<?>"] # Or any suitable placeholder

        logger.info(f"Successfully predicted {len(predicted_words)} words for input '{text}': {predicted_words}")

        # Prepare the successful response data
        response_data = {'predictions': predicted_words}
        logger.debug(f"Data being returned (before jsonify): {response_data}")

        # Return the list of predicted words with a 200 OK status
        return jsonify(response_data), 200

    except Exception as e:
        # Log the full exception traceback for debugging
        logger.error(f"Prediction Error processing text '{text}': {e}", exc_info=True)

        # Prepare the error response data
        error_response = {
            'error': 'Prediction failed due to an internal server error.',
            'details': str(e) # Provide a general error detail, traceback is in server logs
        }
        logger.error(f"Returning error response: {error_response}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify(error_response), 500

# --- TTS Endpoint ---
@app.route('/tts', methods=['POST'])
def tts():
    if tts_model is None:
        return jsonify({'error': 'TTS model not loaded'}), 500

    logger.info(f"TTS request received: {request.json}")
    text = request.json.get('text')
    voice_type = request.json.get('voice_type', 'trained') # Default to trained
    reference_audio_base64 = request.json.get('reference_audio')
    trained_voice_name = request.json.get('trained_voice_name')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    temp_dir = None
    try:
        temp_dir = get_secure_temp_dir()
        request_id = f"{os.getpid()}_{int(time.time())}"
        speaker_wav = None
        tmp_ref_audio_path = None

        if voice_type == 'cloned':
            if not reference_audio_base64:
                return jsonify({'error': 'Reference audio required for cloned voice'}), 400

            try:
                reference_audio_bytes = base64.b64decode(reference_audio_base64)
                tmp_ref_audio_path = temp_dir / f"ref_{request_id}.wav"
                with open(tmp_ref_audio_path, 'wb') as f:
                    f.write(reference_audio_bytes)
                speaker_wav = str(tmp_ref_audio_path)
                logger.info(f"Using cloned voice with reference: {speaker_wav}")
            except Exception as e:
                logger.error(f"Failed to process reference audio: {e}")
                return jsonify({'error': 'Invalid reference audio format'}), 400

        elif voice_type == 'trained':
            if not trained_voice_name:
                # If no specific voice name, try using the current default
                global current_voice_name
                trained_voice_name = current_voice_name
                if not trained_voice_name:
                    return jsonify({'error': 'Trained voice name required, and no default set'}), 400
                logger.info(f"Using default trained voice: {trained_voice_name}")

            speaker_wav = reference_voice_map.get(trained_voice_name)
            if not speaker_wav:
                 logger.error(f"Reference audio path not found in map for trained voice: {trained_voice_name}")
                 return jsonify({'error': f'Reference audio path not found for trained voice: {trained_voice_name}'}), 400
            if not os.path.exists(speaker_wav):
                logger.error(f"Reference audio file does not exist for trained voice '{trained_voice_name}': {speaker_wav}")
                return jsonify({'error': f'Reference audio file not found for trained voice: {trained_voice_name}'}), 400
            logger.info(f"Using trained voice '{trained_voice_name}' with reference: {speaker_wav}")

        else:
            return jsonify({'error': 'Invalid voice_type. Must be "cloned" or "trained"'}), 400

        # Generate speech
        logger.info(f"Generating TTS with text: '{text[:50]}...' using {voice_type} voice")

        # Ensure model is on correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model.to(device)

        # Call TTS function
        # Note: Coqui TTS API might change; parameters may need adjustment
        audio_output = tts_model.tts(
            text=text,
            speaker_wav=speaker_wav,
            language="en" # Assuming English, adjust if multilingual needed
        )

        # Save output audio
        tmp_output_audio_path = temp_dir / f"output_{request_id}.wav"
        sf.write(str(tmp_output_audio_path), np.array(audio_output), tts_model.synthesizer.output_sample_rate)

        # Convert to base64
        with open(tmp_output_audio_path, 'rb') as audio_file:
            audio_base64_output = base64.b64encode(audio_file.read()).decode('utf-8')

        return jsonify({'audio_data': audio_base64_output})

    except Exception as e:
        logger.error(f"TTS Error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Text-to-speech generation failed', 'details': str(e)}), 500
    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


# --- Record Reference Voice Endpoint ---
@app.route('/record_reference', methods=['POST'])
def record_reference():
    audio_data_base64 = request.json.get('audio_data')
    if not audio_data_base64:
        return jsonify({'error': 'No audio data provided'}), 400

    # Note: This endpoint saves a *temporary* reference for immediate use.
    # It's not intended for permanent storage or training datasets.
    temp_dir = None
    try:
        audio_data = base64.b64decode(audio_data_base64)
        # Use the main temp dir for these transient references
        temp_dir = get_secure_temp_dir()
        reference_audio_path = temp_dir / f"user_reference_{int(time.time())}.wav"

        with open(reference_audio_path, 'wb') as f:
            f.write(audio_data)

        # Return the *base64* data of the saved file, so the client can use it directly
        # without needing the server path (which might not be accessible or persistent)
        with open(reference_audio_path, 'rb') as f_read:
            saved_audio_base64 = base64.b64encode(f_read.read()).decode('utf-8')

        logger.info(f"Temporary reference voice recorded successfully at {reference_audio_path}")
        return jsonify({
            'message': 'Reference voice recorded for immediate use.',
            'reference_audio_base64': saved_audio_base64 # Send back the base64
        })

    except Exception as e:
        logger.error(f"Record Reference Error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save reference voice', 'details': str(e)}), 500
    finally:
        # Since this is for immediate use, we might not clean up immediately,
        # relying on the periodic cleanup task. Or clean up after a short delay?
        # For now, rely on periodic cleanup. If 'temp_dir' was created, it will be cleaned eventually.
        pass


# --- List Trained Voices Endpoint ---
@app.route('/list_voices', methods=['GET'])
def list_voices():
    global available_voices
    logger.info("GET /list_voices endpoint hit")
    load_trained_voices() # Reload to ensure list is up-to-date
    logger.info(f"Returning available voices: {available_voices}")
    return jsonify({'voices': available_voices})


# --- Set Current Trained Voice Endpoint ---
@app.route('/set_voice', methods=['POST'])
def set_voice():
    global current_voice_name, available_voices
    voice_name = request.json.get('voice_name')
    if not voice_name:
        return jsonify({'error': 'No voice name provided'}), 400

    # Ensure the voice list is current
    load_trained_voices()

    if voice_name not in available_voices:
        logger.warning(f"Attempted to set non-existent voice: {voice_name}. Available: {available_voices}")
        return jsonify({'error': f'Voice "{voice_name}" not found in available voices'}), 404 # Use 404 Not Found

    current_voice_name = voice_name
    logger.info(f"Current trained voice set to: {current_voice_name}")
    return jsonify({'message': f'Default trained voice set to {voice_name}'})


# --- Dataset Management ---
@app.route('/list_dataset_samples', methods=['GET'])
def list_dataset_samples():
    try:
        if not DATASET_DIR.exists():
            logger.warning(f"Dataset directory not found: {DATASET_DIR}")
            return jsonify({'samples': []})

        samples = [f.name for f in DATASET_DIR.iterdir() if f.is_file() and f.name.lower().endswith('.wav')]
        logger.info(f"Found {len(samples)} dataset samples in {DATASET_DIR}")
        return jsonify({'samples': sorted(samples)}) # Sort for consistent order
    except Exception as e:
        logger.error(f"Error listing dataset samples: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list dataset samples', 'details': str(e)}), 500


@app.route('/add_dataset_sample', methods=['POST'])
def add_dataset_sample():
    sample_name = request.json.get('sample_name')
    audio_data_base64 = request.json.get('audio_data')

    if not sample_name or not audio_data_base64:
        return jsonify({'error': 'Sample name and audio data are required'}), 400

    try:
        # Validate and sanitize filename
        safe_sample_name = validate_filename(sample_name)
        if not safe_sample_name:
            return jsonify({'error': 'Invalid sample name provided'}), 400

        # Ensure .wav extension
        if not safe_sample_name.lower().endswith('.wav'):
            safe_sample_name += '.wav'

        # Ensure dataset directory exists
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

        audio_data = base64.b64decode(audio_data_base64)
        sample_path = DATASET_DIR / safe_sample_name

        # Optional: Check if file already exists? Overwrite or return error?
        if sample_path.exists():
            logger.warning(f"Dataset sample '{safe_sample_name}' already exists. Overwriting.")
            # return jsonify({'error': f'Sample "{safe_sample_name}" already exists'}), 409 # Conflict

        with open(sample_path, 'wb') as f:
            f.write(audio_data)

        logger.info(f"Dataset sample added: {safe_sample_name}")
        return jsonify({
            'message': f'Sample "{safe_sample_name}" added successfully',
            'sample_name': safe_sample_name # Return the sanitized name
        })
    except Exception as e:
        logger.error(f"Error adding dataset sample: {e}", exc_info=True)
        return jsonify({'error': 'Failed to add dataset sample', 'details': str(e)}), 500


@app.route('/delete_dataset_sample', methods=['POST'])
def delete_dataset_sample():
    sample_name = request.json.get('sample_name')

    if not sample_name:
        return jsonify({'error': 'Sample name is required'}), 400

    try:
        safe_sample_name = validate_filename(sample_name)
        if not safe_sample_name:
            return jsonify({'error': 'Invalid sample name provided'}), 400

        # Ensure .wav extension for matching
        if not safe_sample_name.lower().endswith('.wav'):
             safe_sample_name += '.wav'

        sample_path = DATASET_DIR / safe_sample_name

        if sample_path.exists() and sample_path.is_file():
             # Double check it's within the dataset dir (security)
            if DATASET_DIR not in sample_path.parents:
                 logger.error(f"Attempt to delete file outside dataset directory: {sample_path}")
                 return jsonify({'error': 'Invalid file path'}), 400

            os.remove(sample_path)
            logger.info(f"Dataset sample deleted: {safe_sample_name}")
            return jsonify({'message': f'Sample "{safe_sample_name}" deleted successfully'})
        else:
            logger.warning(f"Dataset sample not found for deletion: {sample_path}")
            return jsonify({'error': f'Sample "{safe_sample_name}" not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting dataset sample: {e}", exc_info=True)
        return jsonify({'error': 'Failed to delete dataset sample', 'details': str(e)}), 500


# --- Train Voice Model Endpoint ---
@app.route('/train_voice_model', methods=['POST'])
def train_voice_model():
    logger.info(f"Train voice model request data: {request.json}")
    try:
        epochs = request.json.get('epochs', 50)
        force_cpu = request.json.get('force_cpu', False)
        voice_name_raw = request.json.get('voice_name')

        if not voice_name_raw or not voice_name_raw.strip():
             return jsonify({'error': 'Voice name is required for training'}), 400

        # Sanitize voice name (allow letters, numbers, underscore, hyphen)
        voice_name = re.sub(r'[^\w\-]+', '_', voice_name_raw.strip())
        if not voice_name: # If sanitization results in empty string
             return jsonify({'error': 'Invalid voice name after sanitization'}), 400

        logger.info(f"Sanitized voice name: {voice_name}")

        # Check resources
        free_space = shutil.disk_usage(str(BASE_DIR)).free / (1024**3) # Check space in base dir
        if free_space < 2:
            return jsonify({'error': f'Insufficient disk space. Available: {free_space:.1f}GB, Required: ~2GB'}), 400

        if not force_cpu and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                total_mem_gb = props.total_memory / (1024**3)
                # Estimate required memory (highly variable, adjust as needed)
                required_gpu_mem_gb = 1.5
                if total_mem_gb < required_gpu_mem_gb:
                     logger.warning(f"Low total GPU memory ({total_mem_gb:.1f}GB). Training might fail.")
                # Check currently available memory
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                cached = torch.cuda.memory_reserved(0) / (1024**3)
                available_gpu_memory = total_mem_gb - (allocated + cached) # Rough estimate
                if available_gpu_memory < required_gpu_mem_gb:
                    logger.warning(f"Insufficient *currently available* GPU memory ({available_gpu_memory:.1f}GB). Trying anyway...")
                    # Don't block, let Coqui TTS handle memory errors if they occur during training
                    # return jsonify({'error': f'Insufficient GPU memory. Available: {available_gpu_memory:.1f}GB, Required: {required_gpu_mem_gb}GB. Consider using CPU mode.'}), 400
            except Exception as e:
                logger.warning(f"Could not accurately check GPU memory: {e}")

        # Check dataset samples
        samples = list(DATASET_DIR.glob("*.wav"))
        if not samples:
            return jsonify({'error': 'No dataset samples found in dataset_samples directory.'}), 400
        if len(samples) < 3:
             logger.warning(f'Only {len(samples)} samples found. Training quality may be low.')
             # Allow proceeding, but maybe return a warning in the response?

        # Check if voice name already exists in TRAINED_VOICE_DIR
        voice_output_dir = TRAINED_VOICE_DIR / voice_name
        if voice_output_dir.exists():
             # Option 1: Return error
             return jsonify({'error': f"A trained voice named '{voice_name}' already exists. Choose a different name or delete the existing one."}), 409 # Conflict
             # Option 2: Allow overwrite (potentially dangerous)
             # logger.warning(f"Overwriting existing trained voice: {voice_name}")
             # shutil.rmtree(voice_output_dir) # Uncomment with caution!

        voice_output_dir.mkdir(parents=True, exist_ok=True) # Create dir

        # Copy dataset samples to the voice-specific training dir
        copied_samples = []
        for sample_path in samples:
            try:
                dest_path = voice_output_dir / sample_path.name
                shutil.copy2(sample_path, dest_path)
                copied_samples.append(dest_path)
            except Exception as copy_e:
                 logger.error(f"Failed to copy sample {sample_path.name} to {voice_output_dir}: {copy_e}")
                 # Clean up potentially partially created dir and return error
                 shutil.rmtree(voice_output_dir, ignore_errors=True)
                 return jsonify({'error': f"Failed to prepare training data for '{voice_name}'."}), 500

        if not copied_samples:
             shutil.rmtree(voice_output_dir, ignore_errors=True)
             return jsonify({'error': 'Failed to copy any dataset samples for training.'}), 500

        # Update speakers.json *before* starting the thread
        speakers_file = TRAINED_VOICE_DIR / "speakers.json"
        speakers = []
        if speakers_file.exists():
            try:
                with open(speakers_file, 'r') as f:
                    speakers = json.load(f)
                    if not isinstance(speakers, list): speakers = [] # Handle invalid json
            except Exception as e:
                logger.error(f"Error reading speakers.json before training: {e}. Starting with empty list.")
                speakers = []

        # Check for existing entry with the same name (should be handled by dir check, but belt-and-suspenders)
        if any(s.get('name') == voice_name for s in speakers):
             logger.error(f"Voice name '{voice_name}' already exists in speakers.json despite directory check passing.")
             shutil.rmtree(voice_output_dir, ignore_errors=True) # Cleanup dir
             return jsonify({'error': f"Voice name '{voice_name}' conflict in speakers.json."}), 409

        # Add placeholder entry
        reference_audio_path_relative = copied_samples[0].relative_to(voice_output_dir) # Use first copied sample as ref, store relative path
        speakers.append({
            'name': voice_name,
            'path': str(reference_audio_path_relative), # Store relative path
            'samples_count': len(copied_samples),
            'training_status': 'in_progress',
            'training_start': datetime.now().isoformat(),
            'training_end': None, # Add placeholder
            'epochs_requested': epochs # Store requested epochs
        })

        try:
            with open(speakers_file, 'w') as f:
                json.dump(speakers, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write initial entry to speakers.json for '{voice_name}': {e}")
            shutil.rmtree(voice_output_dir, ignore_errors=True) # Cleanup dir
            return jsonify({'error': 'Failed to update voice registry before training.'}), 500

        # Start training in a background thread
        training_thread = TrainingProcessThread(
            dataset_path=str(voice_output_dir), # Path containing the copied samples
            output_path=str(voice_output_dir), # Coqui output goes here too
            voice_name=voice_name,
            epochs=epochs,
            use_cpu=force_cpu or not torch.cuda.is_available()
        )
        training_thread.start()

        logger.info(f"Voice model '{voice_name}' training started with {len(copied_samples)} samples, output to {voice_output_dir}")
        return jsonify({
            'message': f'Voice model "{voice_name}" training started with {len(copied_samples)} samples.',
            'voice_name': voice_name,
            'status': 'training_started',
            'connect_to_socket': True # Tell client to listen for socket events
        })
    except Exception as e:
        logger.error(f"Error starting voice model training: {e}", exc_info=True)
        # Attempt cleanup if voice_output_dir was created
        if 'voice_output_dir' in locals() and voice_output_dir.exists():
            shutil.rmtree(voice_output_dir, ignore_errors=True)
            logger.info(f"Cleaned up directory {voice_output_dir} after training start failure.")
        # Attempt cleanup of speakers.json entry if added
        if 'speakers_file' in locals() and speakers_file.exists() and 'voice_name' in locals():
             try:
                 with open(speakers_file, 'r') as f:
                     speakers = json.load(f)
                 speakers_filtered = [s for s in speakers if s.get('name') != voice_name]
                 if len(speakers_filtered) < len(speakers):
                     with open(speakers_file, 'w') as f:
                         json.dump(speakers_filtered, f, indent=4)
                     logger.info(f"Removed entry for '{voice_name}' from speakers.json after training start failure.")
             except Exception as cleanup_e:
                 logger.error(f"Failed to cleanup speakers.json entry for '{voice_name}': {cleanup_e}")

        return jsonify({'error': 'Failed to start voice model training', 'details': str(e)}), 500


# --- Socket.IO event handlers ---
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected to Socket.IO: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected from Socket.IO: {request.sid}")

@socketio.on('check_training_status')
def handle_check_training_status(data):
    try:
        voice_name = data.get('voice_name')
        if not voice_name:
            emit('training_status_error', {'error': 'No voice name provided'})
            return

        speakers_file = TRAINED_VOICE_DIR / "speakers.json"
        if speakers_file.exists():
            try:
                with open(speakers_file, 'r') as f:
                    speakers = json.load(f)

                for speaker in speakers:
                    if speaker.get('name') == voice_name:
                        status = speaker.get('training_status', 'unknown')
                        emit('training_status', {
                            'voice_name': voice_name,
                            'status': status,
                            'progress': 100 if status == 'complete' else (0 if status == 'in_progress' else -1), # Simple progress
                            'message': f"Status for '{voice_name}': {status}"
                        }, room=request.sid) # Emit only to the requesting client
                        return

                emit('training_status_error', {'error': f'Voice "{voice_name}" not found'}, room=request.sid)
            except Exception as e:
                logger.error(f"Error reading speakers.json for status check: {e}")
                emit('training_status_error', {'error': f'Error checking training status: {str(e)}'}, room=request.sid)
        else:
            emit('training_status_error', {'error': 'Speakers file not found'}, room=request.sid)
    except Exception as e:
        logger.error(f"Error in handle_check_training_status: {e}", exc_info=True)
        emit('training_status_error', {'error': 'Internal server error during status check'}, room=request.sid)


@socketio.on('error')
def handle_error(e):
    logger.error(f"Socket.IO error from client {request.sid}: {e}")


# --- Periodic Cleanup Task ---
TEMP_FILE_TTL_SECONDS = 3600 # 1 hour

def cleanup_temp_files():
    logger.info("Running periodic cleanup of temp files...")
    cleaned_files = 0
    cleaned_dirs = 0
    try:
        now = time.time()
        if not TEMP_DIR.exists():
            logger.info("Temp directory does not exist, skipping cleanup.")
            return

        for item in TEMP_DIR.iterdir():
            try:
                item_stat = item.stat()
                item_age = now - item_stat.st_mtime

                if item.is_file() and item_age > TEMP_FILE_TTL_SECONDS:
                    os.remove(item)
                    logger.debug(f"Cleaned up old temp file: {item}")
                    cleaned_files += 1
                elif item.is_dir():
                    # Clean up directories older than TTL *if* they are empty
                    # Or force delete older directories regardless of content? Let's be safer first.
                    if item_age > TEMP_FILE_TTL_SECONDS * 2: # Longer TTL for dirs
                       try:
                           # Check if dir is empty first (safer)
                           # if not any(item.iterdir()):
                           #    os.rmdir(item)
                           #    logger.debug(f"Removed empty old temp directory: {item}")
                           #    cleaned_dirs += 1
                           # Force remove old directories (use with caution)
                           shutil.rmtree(item)
                           logger.debug(f"Force-removed old temp directory: {item}")
                           cleaned_dirs +=1
                       except OSError as e:
                           logger.warning(f"Could not remove temp directory {item}: {e}")
            except FileNotFoundError:
                logger.debug(f"Temp item {item} already removed.")
            except Exception as e:
                logger.warning(f"Error processing temp item {item}: {e}")

        logger.info(f"Temp cleanup finished. Removed {cleaned_files} files, {cleaned_dirs} directories.")

    except Exception as e:
        logger.error(f"Error during temp file cleanup task: {e}", exc_info=True)


# Use sched for periodic task
cleanup_scheduler = sched.scheduler(time.time, time.sleep)
cleanup_interval_seconds = 3600 # Run cleanup every hour

def schedule_next_cleanup():
    """Schedules the cleanup task."""
    cleanup_scheduler.enter(cleanup_interval_seconds, 1, run_cleanup_task)
    logger.info(f"Scheduled next temp cleanup in {cleanup_interval_seconds} seconds.")

def run_cleanup_task():
    """Wrapper function to run cleanup and reschedule."""
    try:
        with app.app_context(): # Ensure app context for potential Flask extensions used indirectly
            cleanup_temp_files()
    except Exception as e:
         logger.error(f"Exception in cleanup task execution: {e}", exc_info=True)
    finally:
        # Always reschedule, even if the current run failed
        schedule_next_cleanup()

def start_cleanup_scheduler_thread():
    """Starts the scheduler in a separate daemon thread."""
    schedule_next_cleanup() # Schedule the first run
    thread = threading.Thread(target=cleanup_scheduler.run, daemon=True, name="TempCleanupScheduler")
    thread.start()
    logger.info("Temp file cleanup scheduler thread started.")


# --- Initialize on Startup ---
load_trained_voices()
if available_voices:
    current_voice_name = available_voices[0]
    logger.info(f"Default trained voice set to: {current_voice_name}")
else:
    logger.warning("No trained voices loaded on startup. Default voice not set.")

# Start periodic cleanup task in a background thread
start_cleanup_scheduler_thread()


if __name__ == '__main__':
    logger.info("Starting Flask server with SocketIO...")

    # Ensure necessary directories exist
    TRAINED_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Use environment variable for debug mode or rely on Flask's default behavior
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'

    # Use allow_unsafe_werkzeug=True only if necessary and understand the risks,
    # especially in production. It's often needed for debugging reloader with SocketIO.
    # For production, use a proper WSGI server like gunicorn or uwsgi with gevent/eventlet workers.
    run_kwargs = {'app': app, 'host': '0.0.0.0', 'port': 5000, 'debug': debug_mode}
    if debug_mode:
        # run_kwargs['allow_unsafe_werkzeug'] = True # Enable if reloader issues occur
        run_kwargs['use_reloader'] = True # Standard Flask reloader
    else:
        run_kwargs['use_reloader'] = False

    logger.info(f"Running SocketIO server with kwargs: {run_kwargs}")
    socketio.run(**run_kwargs)