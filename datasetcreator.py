#-----Dataset Creation----
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import threading
import time

class VoiceDatasetRecorder:
    def __init__(self, output_dir="dataset", sample_rate=22050):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.recording = False
        self.current_recording = None
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Voice Dataset Recorder")
        self.root.geometry("400x300")
        
        # Recording duration entry
        duration_frame = tk.Frame(self.root)
        duration_frame.pack(pady=10)
        tk.Label(duration_frame, text="Recording Duration (seconds):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="5")
        tk.Entry(duration_frame, textvariable=self.duration_var, width=5).pack(side=tk.LEFT)
        
        # Sample name entry
        name_frame = tk.Frame(self.root)
        name_frame.pack(pady=10)
        tk.Label(name_frame, text="Sample Name:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        tk.Entry(name_frame, textvariable=self.name_var, width=20).pack(side=tk.LEFT)
        
        # Record button
        self.record_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to record")
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(pady=10)
        
        # Recordings list
        tk.Label(self.root, text="Recorded Samples:").pack(pady=5)
        self.recordings_list = tk.Listbox(self.root, width=40, height=8)
        self.recordings_list.pack(pady=5)
        self.update_recordings_list()
        
    def update_recordings_list(self):
        """Update the list of recorded samples"""
        self.recordings_list.delete(0, tk.END)
        for file in sorted(self.output_dir.glob("*.wav")):
            self.recordings_list.insert(tk.END, file.name)
            
    def record_audio(self, duration):
        """Record audio for specified duration"""
        self.current_recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording:
            try:
                duration = float(self.duration_var.get())
                name = self.name_var.get().strip()
                
                if not name:
                    messagebox.showerror("Error", "Please enter a sample name")
                    return
                    
                # Start recording in a separate thread
                self.recording = True
                self.record_button.config(text="Recording...", state=tk.DISABLED)
                self.status_var.set("Recording in progress...")
                
                def record_thread():
                    self.record_audio(duration)
                    
                    # Save the recording
                    output_path = self.output_dir / f"{name}.wav"
                    wavfile.write(str(output_path), self.sample_rate, self.current_recording)
                    
                    # Update UI
                    self.root.after(0, self.recording_completed)
                
                threading.Thread(target=record_thread).start()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid duration")
                
    def recording_completed(self):
        """Update UI after recording is complete"""
        self.recording = False
        self.record_button.config(text="Start Recording", state=tk.NORMAL)
        self.status_var.set("Recording saved successfully")
        self.update_recordings_list()
        self.name_var.set("")  # Clear the name field
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    recorder = VoiceDatasetRecorder()
    recorder.run()