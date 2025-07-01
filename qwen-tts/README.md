# Qwen-TTS Sichuan Dialect Demo

This project demonstrates the Qwen-TTS model's ability to generate speech with Sichuan dialect accent. Based on the [Qwen blog post](https://qwenlm.github.io/blog/qwen-tts/), this demo showcases the "Sunny" voice which specializes in Sichuan dialect.

## Features

- 🎭 **Sichuan Dialect Support**: Uses the "Sunny" voice for authentic Sichuan accent
- 📝 **Multiple Sample Texts**: Includes traditional rhymes, stories, and daily conversations
- 🎵 **Audio Playback**: Optional real-time audio playback using pygame
- 📁 **File Management**: Organized output directory structure
- 🛡️ **Error Handling**: Comprehensive error handling and user feedback
- 🔧 **Easy Setup**: Simple installation and configuration

## Prerequisites

- Python 3.7 or higher
- DASHSCOPE_API_KEY (Qwen API key)
- Internet connection for API calls

## Installation

1. **Clone or download this project**
   ```bash
   cd qwen-tts
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   ```bash
   export DASHSCOPE_API_KEY='your_api_key_here'
   ```
   
   Or add it to your shell profile:
   ```bash
   echo 'export DASHSCOPE_API_KEY="your_api_key_here"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Usage

Run the demo:
```bash
python qwen_tts_sichuan_demo.py
```

The program will:
1. Process multiple Sichuan dialect text samples
2. Generate audio files using Qwen-TTS
3. Save files to the `output/` directory
4. Optionally play audio files (if pygame is available)

## Sample Texts

The demo includes four Sichuan dialect samples:

1. **Traditional Sichuan Rhyme** (胖娃胖嘟嘟...)
   - A traditional children's rhyme about a chubby child

2. **Sichuan Story** (他一辈子的使命...)
   - A story about perseverance in Sichuan dialect

3. **Sichuan Daily Life** (今天天气巴适得很...)
   - Daily conversation about weather and hotpot

4. **Sichuan Greeting** (老乡，你从哪儿来嘛...)
   - Traditional Sichuan greeting and invitation to chat

## Output

Generated audio files are saved in the `output/` directory:
```
qwen-tts/
├── output/
│   ├── sichuan_sample_01.wav
│   ├── sichuan_sample_02.wav
│   ├── sichuan_sample_03.wav
│   └── sichuan_sample_04.wav
├── qwen_tts_sichuan_demo.py
├── requirements.txt
└── README.md
```

## API Information

This demo uses the Qwen-TTS API with the following specifications:
- **Model**: `qwen-tts-latest` (or `qwen-tts-2025-05-22`)
- **Voice**: `Sunny` (Sichuan dialect)
- **Language Support**: Chinese with Sichuan accent
- **API Provider**: Alibaba Cloud (DashScope)

## Features from Qwen Blog

Based on the [Qwen-TTS blog post](https://qwenlm.github.io/blog/qwen-tts/):

- ✅ **Human-level naturalness** and expressiveness
- ✅ **Automatic prosody adjustment** based on input text
- ✅ **Emotional inflections** and pacing control
- ✅ **3 Chinese dialects** support (Pekingese, Shanghainese, Sichuanese)
- ✅ **7 Chinese-English bilingual voices** available

## Performance Metrics

According to the Qwen blog, Qwen-TTS achieves excellent performance on the SeedTTS-Eval benchmark:
- Low Word Error Rate (WER)
- High Similarity (SIM) scores
- Human-level naturalness

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   DASHSCOPE_API_KEY environment variable not set
   ```
   Solution: Set your API key as described in the installation section.

2. **Import Error for dashscope**
   ```
   Error: dashscope package not found
   ```
   Solution: Install dependencies with `pip install -r requirements.txt`

3. **Audio Playback Not Working**
   ```
   Warning: pygame not available
   ```
   Solution: Install pygame with `pip install pygame`

4. **Network Timeout**
   ```
   Download failed: timeout
   ```
   Solution: Check your internet connection and try again.

### Getting API Key

1. Visit [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
2. Sign up or log in to your account
3. Navigate to the API key management section
4. Create a new API key for Qwen-TTS
5. Copy the key and set it as an environment variable

## Contributing

Feel free to:
- Add more Sichuan dialect samples
- Improve error handling
- Add support for other dialects
- Enhance the audio playback features

## License

This project is for educational and demonstration purposes. Please refer to the [Qwen-TTS terms of service](https://qwenlm.github.io/blog/qwen-tts/) for commercial usage.

## References

- [Qwen-TTS Blog Post](https://qwenlm.github.io/blog/qwen-tts/)
- [DashScope Documentation](https://help.aliyun.com/zh/dashscope/)
- [Qwen Model Hub](https://huggingface.co/Qwen)

## Acknowledgments

- Qwen Team for developing the TTS model
- Alibaba Cloud for providing the API service
- The Sichuan dialect community for preserving the dialect 