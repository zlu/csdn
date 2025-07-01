"""
Configuration file for Qwen-TTS Sichuan Dialect Demo
===================================================

This file contains configuration settings that can be easily modified
to customize the demo behavior.
"""

# API Configuration
MODEL_NAME = "qwen-tts-latest"  # or "qwen-tts-2025-05-22"
VOICE_NAME = "Sunny"  # Sichuan dialect voice

# Available voices from Qwen blog:
# - Cherry: Chinese-English bilingual
# - Ethan: Chinese-English bilingual  
# - Chelsie: Chinese-English bilingual
# - Serena: Chinese-English bilingual
# - Dylan: Pekingese dialect
# - Jada: Shanghainese dialect
# - Sunny: Sichuanese dialect

# Output Configuration
OUTPUT_DIR = "output"
AUDIO_FORMAT = "wav"

# Audio Playback Configuration
AUTO_PLAY = False  # Set to True to automatically play audio without asking
PLAYBACK_TIMEOUT = 30  # Maximum seconds to wait for audio playback

# Network Configuration
REQUEST_TIMEOUT = 30  # Timeout for API requests in seconds
DOWNLOAD_TIMEOUT = 30  # Timeout for audio downloads in seconds

# Sample Texts Configuration
SAMPLE_TEXTS = [
    {
        "title": "Traditional Sichuan Rhyme",
        "text": "胖娃胖嘟嘟，骑马上成都，成都又好耍。胖娃骑白马，白马跳得高。胖娃耍关刀，关刀耍得圆。胖娃吃汤圆。",
        "description": "A traditional Sichuan children's rhyme about a chubby child",
        "filename": "sichuan_rhyme.wav"
    },
    {
        "title": "Sichuan Story",
        "text": "他一辈子的使命就是不停地爬哟，爬到大海头上去，不管有好多远！",
        "description": "A Sichuan dialect story about perseverance",
        "filename": "sichuan_story.wav"
    },
    {
        "title": "Sichuan Daily Life",
        "text": "今天天气巴适得很，我们一起去吃火锅嘛！那个味道简直不摆了，安逸得很！",
        "description": "Daily conversation in Sichuan dialect about weather and hotpot",
        "filename": "sichuan_daily.wav"
    },
    {
        "title": "Sichuan Greeting",
        "text": "老乡，你从哪儿来嘛？要不要一起摆龙门阵？",
        "description": "Sichuan dialect greeting and invitation to chat",
        "filename": "sichuan_greeting.wav"
    },
    {
        "title": "Sichuan Food Culture",
        "text": "川菜就是巴适，麻婆豆腐、回锅肉、宫保鸡丁，样样都安逸！",
        "description": "Sichuan dialect about food culture",
        "filename": "sichuan_food.wav"
    }
]

# Custom sample texts can be added here
CUSTOM_TEXTS = [
    # Add your custom Sichuan dialect texts here
    # {
    #     "title": "Your Custom Title",
    #     "text": "Your custom Sichuan dialect text here",
    #     "description": "Description of your custom text",
    #     "filename": "custom_sample.wav"
    # }
]

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SHOW_PROGRESS = True  # Show progress indicators during synthesis

# Error Handling Configuration
MAX_RETRIES = 3  # Maximum number of retries for failed API calls
RETRY_DELAY = 2  # Delay between retries in seconds

# Performance Configuration
BATCH_SIZE = 1  # Number of texts to process in parallel (keep at 1 for API limits) 