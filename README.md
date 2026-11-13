# AXION – AI-Powered Voice Assistant

AXION is a Python-based personal voice assistant designed to make computer interaction easier through voice commands.

It combines speech recognition, wake-word detection, text-to-speech, and command processing to provide a voice-driven assistant experience.

The goal of AXION is to create a practical, extensible desktop assistant that can understand spoken commands, respond naturally, and help users perform everyday tasks.

## ✨ Features

- **Wake-Word Detection:** Activate the assistant using supported phrases:
  - Hey AXION
  - Hello AXION
  - Hey Action
  - Hello Action
- **Speech Recognition:** Converts spoken commands into text using OpenAI Whisper.
- **Voice Responses:** Responds using text-to-speech.
- **Command Processing:** Processes recognized commands and routes them to relevant functionality.
- **Reminders:** Supports reminder-related commands.
- **Memory:** Includes assistant memory functionality.
- **Web Search:** Supports web search functionality.
- **Weather:** Supports weather-related queries.
- **System Interaction:** Includes system-control functionality.
- **Conversation:** Supports interactive voice-based conversations.

## 🛠️ Technology Stack

| Technology           | Purpose                                  |
| -------------------- | ---------------------------------------- |
| Python               | Core programming language                |
| OpenAI Whisper       | Speech recognition                       |
| Text-to-Speech (TTS) | Voice responses                          |
| SoundDevice          | Microphone audio capture                 |
| PowerShell           | Windows microphone recording integration |
| WSL2 / Debian        | Development environment                  |

## 🏗️ Project Architecture

AXION follows a modular architecture, separating speech recognition, wake-word detection, command processing, and voice responses.

```text
AXION/
│
├── axion/
│   ├── main.py          # Main application entry point
│   ├── stt.py           # Speech-to-text processing
│   ├── tts.py           # Text-to-speech
│   ├── wake_word.py     # Wake-word detection
│   └── ...              # Other assistant modules
│
├── .venv/               # Python virtual environment
├── requirements.txt     # Project dependencies (if included)
└── README.md
```

*The architecture above is illustrative. Update it to match the actual repository structure before publishing.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later (use the version supported by your dependencies)
- Microphone
- Internet connection for online features
- Debian / Linux or a supported Windows and WSL2 setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AXION.git
cd AXION
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Linux / Debian / WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

If the repository contains `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Run AXION

```bash
python -m axion.main
```

### Run with wake-word mode

```bash
AXION_MIC_DEVICE=1 AXION_WAKE_WORD=1 AXION_WAKE_DEBUG=1 python -m axion.main
```

## 🎙️ Voice Interaction

AXION listens for its supported wake phrases before processing the following spoken command.

Example:

```text
User: Hello Action, what is the time?
AXION: [Provides the current time]
```

Another example:

```text
User: Hello Action, what is the weather?
AXION: [Processes the weather request]
```

## 🧪 Development and Testing

Run the existing test suite using:

```bash
pytest
```

Run Python compilation checks:

```bash
python -m compileall axion
```

## 🗺️ Roadmap

The following improvements are planned:

- [ ] Improve speech recognition accuracy in noisy environments.
- [ ] Implement optional audio gain normalization.
- [ ] Improve wake-word detection reliability.
- [ ] Add more voice-controlled system operations.
- [ ] Add modular command extensions.
- [ ] Improve conversation context and memory.
- [ ] Add support for additional languages.
- [ ] Add a graphical user interface.
- [ ] Improve error handling and voice interaction feedback.

## 🔒 Privacy

AXION uses microphone input for voice interaction. Review the implementation and any external services before using it with sensitive information.

Do not commit API keys, access tokens, personal recordings, or other sensitive information to GitHub.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run relevant tests.
5. Submit a pull request.

## 📌 Project Status

AXION is an actively developed personal voice assistant. Core voice interaction and assistant functionality have been developed, while additional improvements are planned.

## 👩‍💻 Author

**Chandradeep Jagarapu**

Developed as a personal AI and Python project.

---

**AXION – Your voice, your assistant.**
