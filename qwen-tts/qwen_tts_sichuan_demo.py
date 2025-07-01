#!/usr/bin/env python3
"""
Qwen-TTS Sichuan Dialect Demo
=============================

This program demonstrates the Qwen-TTS model's ability to generate speech
with Sichuan dialect accent. Based on the Qwen blog post:
https://qwenlm.github.io/blog/qwen-tts/

The program showcases:
1. Text-to-speech synthesis with Sichuan accent
2. Multiple sample texts in Sichuan dialect
3. Audio file generation and playback
4. Error handling and API key management

Author: Generated based on Qwen-TTS documentation
Date: 2025
"""

import os
import sys
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional

try:
    import dashscope
except ImportError:
    print("Error: dashscope package not found. Please install it with:")
    print("pip install dashscope")
    sys.exit(1)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Audio playback will be disabled.")
    print("Install with: pip install pygame")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not available. .env files will not be loaded.")
    print("Install with: pip install python-dotenv")


class QwenTTSSichuanDemo:
    """Demo class for Qwen-TTS with Sichuan dialect."""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.model = "qwen-tts-latest"
        self.voice = "Sunny"  # Sichuan dialect voice
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Sample texts in Sichuan dialect from the blog
        self.sichuan_samples = [
            {
                "title": "Traditional Sichuan Rhyme",
                "text": "胖娃胖嘟嘟，骑马上成都，成都又好耍。胖娃骑白马，白马跳得高。胖娃耍关刀，关刀耍得圆。胖娃吃汤圆。",
                "description": "A traditional Sichuan children's rhyme about a chubby child"
            },
            {
                "title": "Sichuan Story",
                "text": "他一辈子的使命就是不停地爬哟，爬到大海头上去，不管有好多远！",
                "description": "A Sichuan dialect story about perseverance"
            },
            {
                "title": "Sichuan Daily Life",
                "text": "今天天气巴适得很，我们一起去吃火锅嘛！那个味道简直不摆了，安逸得很！",
                "description": "Daily conversation in Sichuan dialect about weather and hotpot"
            },
            {
                "title": "Sichuan Greeting",
                "text": "老乡，你从哪儿来嘛？要不要一起摆龙门阵？",
                "description": "Sichuan dialect greeting and invitation to chat"
            }
        ]
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file."""
        if not DOTENV_AVAILABLE:
            return
        
        # Try to load from current directory
        env_files = [".env", "../.env", "../../.env"]
        
        for env_file in env_files:
            env_path = Path(env_file)
            if env_path.exists():
                print(f"📄 Loading environment from: {env_path.absolute()}")
                load_dotenv(env_path)
                return
        
        print("ℹ️ No .env file found in current or parent directories")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variable or .env file."""
        # First, try to load from .env file
        self._load_env_file()
        
        # Then check environment variables
        api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if not api_key:
            # Try alternative environment variable names
            alternative_keys = [
                "DASHSCOPE_API_KEY",
                "QWEN_API_KEY", 
                "QWEN_TTS_API_KEY",
                "ALIBABA_API_KEY"
            ]
            
            for key in alternative_keys:
                api_key = os.getenv(key)
                if api_key:
                    print(f"✅ Found API key using environment variable: {key}")
                    break
        
        if not api_key:
            raise EnvironmentError(
                "DASHSCOPE_API_KEY not found in environment variables or .env file.\n"
                "Please set your API key using one of these methods:\n"
                "1. Environment variable: export DASHSCOPE_API_KEY='your_api_key_here'\n"
                "2. .env file: Create a .env file with DASHSCOPE_API_KEY=your_api_key_here\n"
                "3. Alternative names: QWEN_API_KEY, QWEN_TTS_API_KEY, or ALIBABA_API_KEY"
            )
        
        # Validate API key format
        if not api_key.startswith("sk-"):
            print("⚠️ Warning: API key doesn't start with 'sk-'. This might not be a valid DashScope API key.")
        
        print(f"✅ API key loaded successfully (starts with: {api_key[:8]}...)")
        return api_key
    
    def synthesize_speech(self, text: str, filename: str) -> Optional[str]:
        """
        Synthesize speech using Qwen-TTS with Sichuan accent.
        
        Args:
            text: Text to synthesize
            filename: Output filename
            
        Returns:
            Path to the generated audio file or None if failed
        """
        try:
            print(f"🎤 Synthesizing: {text[:50]}...")
            
            response = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
                model=self.model,
                api_key=self.api_key,
                text=text,
                voice=self.voice,
            )
            
            # Check response validity
            if response is None:
                raise RuntimeError("API call returned None response")
            
            if response.output is None:
                raise RuntimeError("API call failed: response.output is None")
            
            if not hasattr(response.output, 'audio') or response.output.audio is None:
                raise RuntimeError("API call failed: response.output.audio is None or missing")
            
            audio_url = response.output.audio["url"]
            
            # Download the audio file
            save_path = self.output_dir / filename
            self._download_audio(audio_url, save_path)
            
            print(f"✅ Audio saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            print(f"❌ Speech synthesis failed: {e}")
            return None
    
    def _download_audio(self, audio_url: str, save_path: Path) -> None:
        """Download audio file from URL."""
        try:
            print(f"📥 Downloading audio from: {audio_url}")
            resp = requests.get(audio_url, timeout=30)
            resp.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(resp.content)
                
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")
    
    def play_audio(self, audio_path: str) -> None:
        """Play audio file using pygame."""
        if not PYGAME_AVAILABLE:
            print("🔇 Audio playback not available (pygame not installed)")
            return
        
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
        finally:
            pygame.mixer.quit()
    
    def run_demo(self) -> None:
        """Run the complete Sichuan dialect demo."""
        print("🎭 Qwen-TTS Sichuan Dialect Demo")
        print("=" * 50)
        print(f"Model: {self.model}")
        print(f"Voice: {self.voice} (Sichuan dialect)")
        print(f"Output directory: {self.output_dir}")
        print()
        
        generated_files = []
        
        for i, sample in enumerate(self.sichuan_samples, 1):
            print(f"\n📝 Sample {i}: {sample['title']}")
            print(f"Description: {sample['description']}")
            print(f"Text: {sample['text']}")
            print("-" * 40)
            
            filename = f"sichuan_sample_{i:02d}.wav"
            audio_path = self.synthesize_speech(sample['text'], filename)
            
            if audio_path:
                generated_files.append(audio_path)
                
                # Ask user if they want to play the audio
                if PYGAME_AVAILABLE:
                    play_choice = input("🎵 Play audio? (y/n): ").lower().strip()
                    if play_choice in ['y', 'yes']:
                        print("🔊 Playing audio...")
                        self.play_audio(audio_path)
                        print("✅ Audio playback completed")
            
            print()
        
        # Summary
        print("📊 Demo Summary")
        print("=" * 50)
        print(f"Total samples processed: {len(self.sichuan_samples)}")
        print(f"Successfully generated: {len(generated_files)}")
        print(f"Output directory: {self.output_dir}")
        
        if generated_files:
            print("\n📁 Generated files:")
            for file_path in generated_files:
                print(f"  - {file_path}")
        
        print("\n🎉 Demo completed!")


def main():
    """Main function to run the demo."""
    try:
        demo = QwenTTSSichuanDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 