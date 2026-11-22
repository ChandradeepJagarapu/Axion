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
