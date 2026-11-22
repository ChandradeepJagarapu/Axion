MODEL_NAME = "qwen2.5-coder:3b"
OLLAMA_HOST = "http://localhost:11434"

# Audio normalization (applied to Whisper's 16 kHz mono float input).
AUDIO_SAMPLE_RATE = 16000
AUDIO_NORMALIZATION_ENABLED = True
AUDIO_TARGET_RMS_DBFS = -20.0
AUDIO_MAX_GAIN_DB = 20.0
AUDIO_MAX_ATTENUATION_DB = -12.0
AUDIO_PEAK_LIMIT = 0.95
AUDIO_SILENCE_PEAK_THRESHOLD = 1e-6
AUDIO_MIN_RMS_FOR_BOOST = 1e-4
AUDIO_TARGET_TOLERANCE_DB = 0.5

# Audio noise reduction (applied before normalization, to Whisper's 16 kHz
# mono float input).  Pure-NumPy STFT spectral gating with a stationary
# noise estimate taken from the quietest noise-like frames.
#
# Opt-in for now: real-world A/B validation (WER on noisy speech, continuous
# speech and clean speech) is still outstanding.  Set AXION_AUDIO_DENOISE=1
# to enable it; AXION_AUDIO_DENOISE=0 keeps it off.
AUDIO_NOISE_REDUCTION_ENABLED = False
AUDIO_DENOISE_SAMPLE_RATE = 16000
AUDIO_DENOISE_FRAME_SAMPLES = 512      # 32 ms at 16 kHz
AUDIO_DENOISE_HOP_SAMPLES = 128        # 8 ms at 16 kHz (75% overlap)
AUDIO_DENOISE_NOISE_PERCENTILE = 15.0
AUDIO_DENOISE_OVER_SUBTRACTION = 2.5
AUDIO_DENOISE_SPECTRAL_FLOOR_DB = -18.0
AUDIO_DENOISE_FREQ_SMOOTHING_BINS = 3
AUDIO_DENOISE_TIME_SMOOTHING_FRAMES = 3
AUDIO_DENOISE_MIN_NOISE_DBFS = -60.0
AUDIO_DENOISE_MIN_CONTRAST_DB = 6.0
AUDIO_DENOISE_MIN_NOISE_FLATNESS = 0.20
AUDIO_DENOISE_MIN_SNR_DB = 10.0
AUDIO_DENOISE_HIGHPASS_HZ = 60.0

# Voice activity detection (applied after denoise, before normalization).
# Opt-in until real-world trimming/WER validation is complete; set
# AXION_AUDIO_VAD=1 to enable it.  Detection is energy based with an adaptive
# noise floor, and extraction always keeps a contiguous first-to-last span.
AUDIO_VAD_ENABLED = False
AUDIO_VAD_SAMPLE_RATE = 16000
AUDIO_VAD_FRAME_MS = 25.0
AUDIO_VAD_HOP_MS = 10.0
AUDIO_VAD_NOISE_PERCENTILE = 20.0
AUDIO_VAD_SPEECH_MARGIN_DB = 8.0
AUDIO_VAD_HYSTERESIS_DB = 3.0
AUDIO_VAD_MIN_SPEECH_DBFS = -60.0
AUDIO_VAD_MIN_SPEECH_MS = 150.0
AUDIO_VAD_MIN_SILENCE_MS = 300.0
AUDIO_VAD_PRE_ROLL_MS = 200.0
AUDIO_VAD_POST_ROLL_MS = 250.0
AUDIO_VAD_MIN_TOTAL_SPEECH_MS = 200.0
AUDIO_VAD_MIN_TRIM_MS = 300.0

# Boundary-presence guard: protects quiet leading/trailing words that fall
# below the main speech threshold but remain audible above the noise floor.
# A lower percentile is used so speech cannot contaminate the reference.
AUDIO_VAD_BOUNDARY_PERCENTILE = 5.0
AUDIO_VAD_BOUNDARY_MARGIN_DB = 3.0
AUDIO_VAD_BOUNDARY_MIN_DBFS = -70.0
AUDIO_VAD_BOUNDARY_MIN_ACTIVITY_MS = 80.0
AUDIO_VAD_BOUNDARY_LINK_MS = 500.0

# Conservative confidence gates: return the original audio when detections
# are all very short or when the buffer has too little dynamic contrast to
# separate speech from noise reliably.
AUDIO_VAD_MIN_CONFIDENT_SPEECH_MS = 300.0
AUDIO_VAD_MIN_CONTRAST_DB = 6.0

# Safety guard: do not trim when detected speech occupies most of the buffer,
# because the adaptive noise floor may then have been estimated from speech
# itself.  Value is a fraction of the original duration.
AUDIO_VAD_MAX_SPEECH_RATIO = 0.70
