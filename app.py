import torch
import transformers
from diffusers import DiffusionPipeline, StableDiffusionXLControlNetPipeline, DPMSolverMultistepScheduler
from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import io
import time
import re
import threading
from controlnet_aux import OpenposeDetector  # Fixed import
import speech_recognition as sr
from flask import Response, stream_with_context
import tempfile
import wave

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__, static_folder='static')

# Create necessary folders
os.makedirs('static/generated', exist_ok=True)
os.makedirs('static/characters', exist_ok=True)
os.makedirs('static/fonts', exist_ok=True)

# Global dict to track generation progress
generation_progress = {}
generation_results = {}

class ComicGenerator:
    def __init__(self):
        # GPU optimization configurations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        print("Initializing GPU context...")
        self.device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        
        print(f"Using device: {self.device}")
        self.dtype = torch.float16  # Force FP16 for all operations

        # Load SDXL base model
        print("Loading Stable Diffusion XL base model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        
        # Load Refiner with memory management
        self.use_refiner = True
        try:
            print("Loading SDXL refiner...")
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=self.dtype,
                variant="fp16",
                use_safetensors=True
            )
            self.refiner.enable_xformers_memory_efficient_attention()
            self.refiner.enable_model_cpu_offload()
        except Exception as e:
            print(f"Could not load refiner: {e}")
            self.use_refiner = False

        # Character data storage
        self.character_concepts = {}
        
        # ControlNet setup
        self.use_controlnet = True
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
            
            print("Initializing ControlNet...")
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-openpose-sdxl-1.0",
                torch_dtype=self.dtype
            )
            
            self.controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=controlnet,
                torch_dtype=self.dtype,
                variant="fp16",
                use_safetensors=True
            )
            self.controlnet_pipe.enable_xformers_memory_efficient_attention()
            self.controlnet_pipe.enable_model_cpu_offload()
            self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        except Exception as e:
            print(f"ControlNet disabled: {e}")
            self.use_controlnet = False

        # Initialize panel-specific narrative generator
        try:
            print("Loading text generation model for narratives...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.narrative_model = None
            self.narrative_tokenizer = None
            
            # Instead of loading a full model which would consume too much GPU memory
            # We'll use a narrative generation approach that builds from the prompt
            # If you want to load a small model for text generation, uncomment below:
            '''
            self.narrative_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.narrative_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            self.narrative_model = self.narrative_model.to("cpu")  # Keep on CPU to save VRAM
            '''
        except Exception as e:
            print(f"Narrative generation will use rule-based fallback: {e}")

        print(f"VRAM usage after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # Load or create comic font
        self.font_path = self._ensure_comic_font()
    
    def _ensure_comic_font(self):
        """Make sure we have a comic font for the text bubbles"""
        # Default system fonts that might be available (platform-dependent)
        font_options = [
            "static/fonts/comic.ttf",  # Check if we already downloaded it
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux
            "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",  # MacOS
            "C:\\Windows\\Fonts\\comic.ttf"  # Windows
        ]
        
        for font_path in font_options:
            if os.path.exists(font_path):
                return font_path
        
        # If no font found, use default (will be system dependent)
        print("No specific comic font found, will use default")
        return None

    def _extract_key_traits(self, description):
        """Helper to extract visual traits from character description"""
        traits = {}
        patterns = {
            'hair': r'(blonde|brunette|black|red|blue|green|purple) hair',
            'eyes': r'(blue|green|brown|hazel|amber) eyes',
            'build': r'(slender|muscular|athletic|petite|stocky) build'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                traits[key] = match.group(1)
        return traits

    def generate_character_embedding(self, character_name, character_description):
        """Generate character concept with VRAM optimization"""
        try:
            torch.cuda.empty_cache()
            base_prompt = f"{character_name}: {character_description}"
            quality_elements = "masterpiece, best quality, highly detailed, sharp focus"
            style_elements = "comic book style, professional illustration, crisp lines, vibrant colors"
            
            full_prompt = f"{quality_elements}, {base_prompt}, {style_elements}"
            negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, mutated, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, blurry, duplicate, multiplied, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, poorly drawn, gross proportions, text, watermark, signature"
            
            char_dir = f"static/characters/{character_name.replace(' ', '_')}"
            os.makedirs(char_dir, exist_ok=True)
            
            pose_prompts = [
                "portrait, face closeup, detailed facial features",
                "full body, standing pose, full view",
                "action pose, dynamic composition"
            ]
            
            for i, pose in enumerate(pose_prompts):
                full_pose_prompt = f"{full_prompt}, {pose}"
                n_steps = 40
                high_noise_frac = 0.8
                
                if self.use_refiner:
                    image = self.pipe(
                        prompt=full_pose_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=n_steps,
                        denoising_end=high_noise_frac,
                        output_type="latent",
                        width=768,
                        height=768,
                        guidance_scale=7.0
                    ).images
                    
                    image = self.refiner(
                        prompt=full_pose_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=n_steps,
                        denoising_start=high_noise_frac,
                        image=image,
                    ).images[0]
                else:
                    image = self.pipe(
                        prompt=full_pose_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,
                        width=768,
                        height=768,
                        guidance_scale=7.0
                    ).images[0]
                
                image.save(f"{char_dir}/pose_{i}.png")
                torch.cuda.empty_cache()

            traits = self._extract_key_traits(character_description)
            self.character_concepts[character_name] = {
                'name': character_name,
                'base_prompt': full_prompt,
                'traits': traits,
                'ref_images': [f"{char_dir}/pose_{i}.png" for i in range(len(pose_prompts))]
            }

            portrait = Image.open(f"{char_dir}/pose_0.png")
            portrait.save(f"static/characters/{character_name.replace(' ', '_')}.png")
            
            if self.use_controlnet:
                try:
                    for i, image_path in enumerate(self.character_concepts[character_name]['ref_images']):
                        pose_image = Image.open(image_path)
                        control_map = self.pose_detector(pose_image)
                        control_map.save(f"{char_dir}/pose_control_{i}.png")
                    
                    self.character_concepts[character_name]['pose_controls'] = [
                        f"{char_dir}/pose_control_{i}.png" for i in range(len(pose_prompts))
                    ]
                except Exception as e:
                    print(f"Could not generate pose controls: {e}")
            
            return True
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def generate_panel(self, prompt, characters=None, pose_reference=None, width=768, height=512):
        """Generate panel with reduced resolution"""
        try:
            torch.cuda.empty_cache()
            start_time = time.time()
            
            quality_elements = "masterpiece, best quality, highly detailed, sharp focus"
            style_elements = "comic book style, professional illustration, crisp lines, vibrant colors"
            negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, mutated, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, blurry, duplicate, multiplied, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, poorly drawn, gross proportions, text, watermark, signature"
            
            character_descriptions = []
            if characters:
                for character in characters:
                    if character in self.character_concepts:
                        char_info = self.character_concepts[character]
                        traits_str = ", ".join([f"{k}: {v}" for k, v in char_info['traits'].items()])
                        character_descriptions.append(f"{char_info['name']}: {traits_str}")
            
            content_prompt = prompt
            character_block = f", featuring {', '.join(character_descriptions)}" if character_descriptions else ""
                    
            full_prompt = f"{quality_elements}, {content_prompt}{character_block}, {style_elements}"
            
            n_steps = 30
            high_noise_frac = 0.8
            
            if pose_reference and self.use_controlnet:
                try:
                    if isinstance(pose_reference, str):
                        pose_image = Image.open(pose_reference)
                    else:
                        pose_image = pose_reference
                    
                    control_image = self.pose_detector(pose_image)
                    
                    image = self.controlnet_pipe(
                        prompt=full_prompt,
                        negative_prompt=negative_prompt,
                        image=control_image,
                        num_inference_steps=n_steps,
                        guidance_scale=7.0,
                        width=width,
                        height=height
                    ).images[0]
                except Exception as e:
                    print(f"ControlNet failed: {e}")
                    image = self._generate_with_base_refiner(full_prompt, negative_prompt, n_steps, high_noise_frac, width, height)
            else:
                image = self._generate_with_base_refiner(full_prompt, negative_prompt, n_steps, high_noise_frac, width, height)
            
            # Don't add border yet - we'll do that after adding narrative
            print(f"Panel generated in {time.time() - start_time:.2f}s")
            return image, content_prompt
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def _generate_with_base_refiner(self, prompt, negative_prompt, n_steps, high_noise_frac, width, height):
        """Memory-optimized generation"""
        try:
            if self.use_refiner:
                latents = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=n_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    width=width,
                    height=height,
                    guidance_scale=7.0
                ).images
                
                image = self.refiner(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    num_inference_steps=n_steps,
                    denoising_start=high_noise_frac,
                    image=latents
                ).images[0]
            else:
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=n_steps,
                    width=width,
                    height=height,
                    guidance_scale=7.0
                ).images[0]
            return image
        except Exception as e:
            torch.cuda.empty_cache()
            raise e

    def generate_narrative_text(self, panel_prompt, panel_index, total_panels):
        """Generate appropriate narrative text for the panel based on context and prompt"""
        # Extract the core narrative elements from the prompt
        # Remove technical terms and keep the storytelling elements
        narrative_elements = re.sub(r'establishing shot|comic book style|key dramatic moment|introduction|resolution', '', panel_prompt)
        narrative_elements = narrative_elements.replace(',,', ',').strip(' ,')
        
        # Contextual narrative generation based on panel position
        if panel_index == 0:  # First panel
            intro_phrases = [
                "Our story begins as",
                "It all started when",
                "The adventure begins with",
                "We find our hero as",
                "The scene opens on"
            ]
            intro = intro_phrases[hash(panel_prompt) % len(intro_phrases)]
            
            # Extract action/setting from prompt
            main_elements = narrative_elements.split(',')
            if len(main_elements) > 2:
                setting = main_elements[1].strip()
                narrative = f"{intro} {setting}."
            else:
                narrative = f"{intro} {narrative_elements}."
        
        elif panel_index == total_panels - 1:  # Last panel
            ending_phrases = [
                "Finally,",
                "In the end,",
                "The story concludes as",
                "At last,",
                "The adventure ends with"
            ]
            ending = ending_phrases[hash(panel_prompt) % len(ending_phrases)]
            narrative = f"{ending} {narrative_elements}."
            
        else:  # Middle panels
            transition_phrases = [
                "Meanwhile,",
                "Suddenly,",
                "Then,",
                "As events unfold,",
                "The story continues as"
            ]
            transition = transition_phrases[hash(panel_prompt) % len(transition_phrases)]
            narrative = f"{transition} {narrative_elements}."
        
        # Clean up the text
        narrative = re.sub(r'\s+', ' ', narrative)  # Remove multiple spaces
        narrative = re.sub(r',\s*,', ',', narrative)  # Remove double commas
        narrative = narrative.strip()
        
        # Keep it reasonably short for a comic panel
        if len(narrative) > 120:
            narrative = narrative[:117] + "..."
            
        return narrative

    def add_narrative_bubble(self, image, narrative_text):
        """Add a narrative bubble to the image"""
        # Create a copy of the image to draw on
        img_with_narrative = image.copy()
        draw = ImageDraw.Draw(img_with_narrative)
        width, height = img_with_narrative.size
        
        # Configuration for the narrative bubble
        bubble_margin = 20
        bubble_padding = 10
        bubble_height = 0  # We'll calculate this based on text
        bubble_width = width - (bubble_margin * 2)
        
        # Set up the font
        try:
            if self.font_path:
                font_size = 18
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
                font_size = 12
        except Exception as e:
            print(f"Font loading error: {e}")
            font = ImageFont.load_default()
            font_size = 12
        
        # Calculate text wrapping and height
        max_chars_per_line = bubble_width // (font_size // 2)  # Approximate width of a character
        words = narrative_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
                
        if current_line:
            lines.append(current_line)
            
        # Calculate bubble height based on number of lines
        line_height = font_size + 4
        bubble_height = (len(lines) * line_height) + (bubble_padding * 2)
        
        # Draw the bubble at the top of the panel
        bubble_coords = [
            bubble_margin, 
            bubble_margin, 
            width - bubble_margin, 
            bubble_margin + bubble_height
        ]
        
        # Draw bubble background with opacity
        bubble_background = Image.new('RGBA', (bubble_width, bubble_height), (255, 255, 255, 180))
        img_with_narrative.paste(bubble_background, (bubble_margin, bubble_margin), bubble_background)
        
        # Draw bubble border
        draw.rectangle(bubble_coords, outline=(0, 0, 0), width=2)
        
        # Draw the text
        text_y = bubble_margin + bubble_padding
        for line in lines:
            draw.text((bubble_margin + bubble_padding, text_y), line, fill=(0, 0, 0), font=font)
            text_y += line_height
            
        return img_with_narrative

    def add_comic_border(self, image):
        border_size = 5
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        draw.rectangle([(0, 0), (width-1, height-1)], outline="black", width=border_size)
        return image

    def generate_story_panels(self, story_prompt, num_panels=3, characters=None, session_id=None):
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CUDA cleanup error: {e}")

        if not story_prompt:
            return []
        
        panel_prompts = self.story_to_panel_prompts(story_prompt, num_panels)
    
        panels = []
        last_panel = None

        for i, panel_prompt in enumerate(panel_prompts):
            if session_id:
                generation_progress[session_id] = int((i / len(panel_prompts)) * 100)

            pose_reference = None
            if i > 0 and last_panel and characters and self.use_controlnet:
                pose_reference = last_panel
            elif i == 0 and characters and self.use_controlnet:
                for character in characters:
                    if character in self.character_concepts and 'pose_controls' in self.character_concepts[character]:
                        pose_reference = self.character_concepts[character]['pose_controls'][2]
                        break
    
        # Generate the panel image
            image, prompt_content = self.generate_panel(panel_prompt, characters, pose_reference)
        
        # Add border to the raw panel
            image_with_border = self.add_comic_border(image.copy())
        
        # Generate narrative text based on the panel content
            narrative_text = self.generate_narrative_text(panel_prompt, i, len(panel_prompts))
        
        # Create a larger canvas with space for the narrative above
            width, height = image_with_border.size
            margin = 20
        
        # Calculate narrative text height
            try:
                if self.font_path:
                    font_size = 18
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    font = ImageFont.load_default()
                    font_size = 12
            except:
                font = ImageFont.load_default()
                font_size = 12
            
        # Simple word wrapping
            max_chars_per_line = (width - margin*2) // (font_size // 2)
            words = narrative_text.split()
            lines = []
            current_line = ""
        
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) <= max_chars_per_line:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
                
            if current_line:
                lines.append(current_line)
            
            line_height = font_size + 4
            text_height = len(lines) * line_height + margin * 2
        
        # Create the combined image
            combined = Image.new('RGB', (width, height + text_height), (255, 255, 255))
        
        # Draw the narrative text
            draw = ImageDraw.Draw(combined)
            text_y = margin
            for line in lines:
                draw.text((margin, text_y), line, fill=(0, 0, 0), font=font)
                text_y += line_height
            
        # Add a separator line
            draw.line([(0, text_height-1), (width, text_height-1)], fill=(0, 0, 0), width=2)
        
        # Paste the bordered panel below the narrative
            combined.paste(image_with_border, (0, text_height))
        
            panels.append(combined)
            last_panel = image  # Use the image without narrative for pose reference
    
            if session_id:
                generation_progress[session_id] = int(((i + 1) / len(panel_prompts))) * 100

        return panels
    def story_to_panel_prompts(self, story_prompt, num_panels):
        if "Panel" in story_prompt or "panel" in story_prompt:
            panels = re.split(r'Panel \d+:|panel \d+:', story_prompt)
            panels = [p.strip() for p in panels if p.strip()]
            
            if len(panels) > num_panels:
                panels = panels[:num_panels]
                
            if len(panels) < num_panels:
                last_panel = panels[-1] if panels else story_prompt
                remaining = num_panels - len(panels)
                for i in range(remaining):
                    if i == remaining - 1:
                        panels.append(f"{last_panel}, conclusion, resolution of the story")
                    else:
                        panels.append(f"{last_panel}, continuation, progressing the action")
            
            return panels
        
        if num_panels <= 1:
            return [f"{story_prompt}, key dramatic moment, comic book style"]
        
        narrative_beats = [
            f"{story_prompt}, establishing shot, introduction, beginning of the story",
            f"{story_prompt}, character introduction, setting the scene",
            f"{story_prompt}, rising action, conflict emerges",
            f"{story_prompt}, confrontation, peak action moment",
            f"{story_prompt}, climax, most dramatic moment",
            f"{story_prompt}, resolution, conclusion of the story"
        ]
        
        if num_panels == 2:
            return [narrative_beats[0], narrative_beats[-1]]
        elif num_panels == 3:
            return [narrative_beats[0], narrative_beats[2], narrative_beats[-1]]
        elif num_panels == 4:
            return [narrative_beats[0], narrative_beats[1], narrative_beats[3], narrative_beats[-1]]
        else:
            return narrative_beats[:min(num_panels, len(narrative_beats))]

# Initialize generator with GPU context
with torch.cuda.device(0):
    comic_generator = ComicGenerator()

@app.route('/')
def index():
    return send_from_directory('templates', 'index_audio.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)



@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    """
    Endpoint to handle speech recognition from an audio file
    Returns the transcribed text
    """
    if 'audio' not in request.files:
        print("No audio file in request")
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    allowed_extensions = {'wav', 'mp3', 'ogg', 'flac'}
    extension = audio_file.filename.rsplit('.', 1)[1].lower() if '.' in audio_file.filename else ''
    
    if extension not in allowed_extensions:
        return jsonify({'error': f'File type not allowed. Please upload {", ".join(allowed_extensions)}'}), 400
    
    # Save the audio file temporarily to ensure correct format
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{extension}')
    audio_file.save(temp_file.name)
    temp_file.close()
    
    # Create a recognizer instance
    recognizer = sr.Recognizer()
    
    try:
        # Convert the uploaded audio to text
        with sr.AudioFile(temp_file.name) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio_data)
            
        # Clean up temp file
        os.unlink(temp_file.name)
        
        print(f"Recognized text: {text}")
        return jsonify({
            'success': True,
            'text': text
        })
    except sr.UnknownValueError:
        print("Speech could not be recognized")
        os.unlink(temp_file.name)
        return jsonify({
            'success': False,
            'error': 'Speech could not be recognized'
        }), 400
    except sr.RequestError as e:
        print(f"RequestError: {e}")
        os.unlink(temp_file.name)
        return jsonify({
            'success': False,
            'error': f'Could not request results from speech recognition service: {e}'
        }), 500
    except Exception as e:
        print(f"Error processing audio: {e}")
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return jsonify({
            'success': False,
            'error': f'Error processing audio: {e}'
        }), 500
@app.route('/process_audio_file', methods=['POST'])
def process_audio_file():
    """
    Endpoint dedicated to processing WAV audio files
    Performs validation on WAV format before processing
    """
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio_file']
    
    # Check if the file is a WAV file
    if not audio_file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Please upload a WAV file'}), 400
    
    try:
        # Save the file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_file.name)
        temp_file.close()
        
        # Validate WAV format
        try:
            with wave.open(temp_file.name, 'rb') as wav_file:
                # Get WAV parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                
                print(f"WAV file: channels={channels}, sample_width={sample_width}, frame_rate={frame_rate}")
                
                # You can add additional validation here if needed
        except Exception as e:
            return jsonify({'error': f'Invalid WAV file: {str(e)}'}), 400
        
        # Create a recognizer instance
        recognizer = sr.Recognizer()
        
        # Process the audio file
        with sr.AudioFile(temp_file.name) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio_data)
        
        # Clean up
        os.unlink(temp_file.name)
        
        return jsonify({
            'success': True,
            'text': text
        })
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/process_blob', methods=['POST'])
def process_blob():
    """
    Process audio blob data from the browser
    """
    if 'audio_blob' not in request.files:
        print("No audio blob in request")
        return jsonify({'error': 'No audio blob provided'}), 400
    
    audio_blob = request.files['audio_blob']
    
    try:
        # Create a temporary file with proper WAV format handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_blob.save(temp_file.name)
        
        # Convert to proper WAV format if needed
        with wave.open(temp_file.name, 'rb') as wav_file:
            # Validate WAV file parameters
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Audio file must be mono (1 channel) and 16-bit PCM format")
    
    # Create a recognizer instance
        recognizer = sr.Recognizer()
    
        try:
        # Convert the uploaded audio to text
            with sr.AudioFile(temp_file.name) as source:
            # Adjust for ambient noise and record
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)
            
            # Use Google's speech recognition
                text = recognizer.recognize_google(audio_data)
        
        # Clean up temp file
            os.unlink(temp_file.name)
        
            print(f"Recognized text from blob: {text}")
            return jsonify({
                'success': True,
                'text': text
            })
        except:
            return jsonify({'error': f'Invalid WAV file: {str(e)}'}), 400
    except Exception as e:
        print(f"Error processing audio blob: {e}")
        try:
            os.unlink(temp_file.name)
        except:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Simplified version of real-time listening
@app.route('/listen', methods=['POST'])
def listen():
    """
    Simplified version of speech recognition that works better with web applications
    """
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            
            return jsonify({
                'success': True,
                'text': text
            })
    except Exception as e:
        print(f"Error in listen: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_character', methods=['POST'])
def generate_character():
    data = request.json
    character_name = data.get('name', '')
    character_description = data.get('description', '')
    
    if not character_name or not character_description:
        return jsonify({'success': False, 'error': 'Name and description required'}), 400
    
    try:
        success = comic_generator.generate_character_embedding(character_name, character_description)
        characters = list(comic_generator.character_concepts.keys())
        
        return jsonify({
            'success': success,
            'message': f"Character {character_name} created",
            'character_image': f"/static/characters/{character_name.replace(' ', '_')}.png",
            'all_characters': characters
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_characters', methods=['GET'])
def get_characters():
    characters = list(comic_generator.character_concepts.keys())
    return jsonify({'characters': characters})

@app.route('/generate_comic', methods=['POST'])
def generate_comic():
    data = request.json
    story_prompt = data.get('prompt', '')
    selected_characters = data.get('characters', [])
    num_panels = min(int(data.get('num_panels', 3)), 6)
    
    if not story_prompt:
        return jsonify({'success': False, 'error': 'Story prompt required'}), 400
    
    try:
        session_id = str(int(time.time()))
        generation_progress[session_id] = 0
        
        def generate_in_background():
            try:
                panels = comic_generator.generate_story_panels(
                    story_prompt, 
                    num_panels=num_panels,
                    characters=selected_characters,
                    session_id=session_id
                )
                
                panel_images = []
                for i, panel in enumerate(panels):
                    buffered = io.BytesIO()
                    panel.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    panel_filename = f"panel_{session_id}_{i}.png"
                    panel.save(f"static/generated/{panel_filename}")
                    
                    panel_images.append({
                        'data': img_str,
                        'filename': panel_filename
                    })
                
                generation_results[session_id] = {
                    'success': True,
                    'panels': panel_images,
                    'message': f"Generated {len(panels)} comic panels with narratives"
                }
                generation_progress[session_id] = 100
                
            except Exception as e:
                generation_results[session_id] = {
                    'success': False,
                    'error': str(e)
                }
                generation_progress[session_id] = -1
        
        thread = threading.Thread(target=generate_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f"Generation started for {num_panels} panels with narratives"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_progress/<session_id>', methods=['GET'])
def check_progress(session_id):
    progress = generation_progress.get(session_id, 0)
    
    if progress == 100 and session_id in generation_results:
        return jsonify({
            'progress': progress,
            'complete': True,
            'results': generation_results[session_id]
        })
    elif progress == -1 and session_id in generation_results:
        return jsonify({
            'progress': -1,
            'complete': True,
            'error': generation_results[session_id].get('error', 'Unknown error')
        })
    else:
        return jsonify({
            'progress': progress,
            'complete': False
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
