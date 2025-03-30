import os
import re
import json
import time
import uuid
import asyncio
import pygame
import nltk
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from rank_bm25 import BM25Okapi
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, messagebox
import threading

nltk.download('punkt', quiet=True)

KEYWORD = "جيل"
ACTIVATION_SOUND_PATH = 'Alarm.mp3' 
RESPONSES_DIR = 'responses'
UNKNOWN_QUESTIONS_FILE = 'unknown_questions.json'

config_lock = threading.Lock()
stats_lock = threading.Lock()

DEFAULT_CONFIG = {
    'thresholds': {
        'activation_confidence': 0.65,
        'response_confidence': 0.45,
        'similarity': 0.35
    },
    'timeouts': {
        'activation_phrase': 5,
        'question_listen': 60
    }
}

CONFIG = DEFAULT_CONFIG.copy()

stats = {
    'total_activations': 0,
    'total_responses': 0,
    'total_unknown': 0
}

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return [token.lower() for token in tokens]

questions = [
    'خدمات الشركه؟',
    'الخدمات؟',
    'مرحبا',
    'ما هيا شركة جي تك',
    'عرف عن الشركة',
    'من صنعك؟',
    'من قام بصنعك؟',
    'كيف حالك؟',
    'فروع الشركه؟',
    'تاريخ الشركة',
    'من انت؟',
    'عرف بنفسك',
    'تاريخ اليوم',
    'عرفني عن نفسك',
    'تكلم عن نفسك',
]

answers = [
    'الخدمات التي نقدمها تشمل البرمجة وتصميم المواقع.',
    'نقدم خدمات متعددة مثل الدعم الفني والتدريب.',
    'مرحبا! كيف يمكنني مساعدتك اليوم؟',
    'شركة جي تك هي شركة متخصصة في تكنولوجيا المعلومات.',
    'شركة جي تك هي شركة متخصصة في تقديم الحلول التقنية.',
    'تم تطويري بواسطة فريق من المهندسين في شركة جي تك.',
    'فريق من المهندسين في شركة جي تك قام بتطويري.',
    'أنا بخير، شكراً لك! كيف يمكنني مساعدتك؟',
    'لدينا فروع في العديد من المدن، يمكنك زيارة موقعنا لمزيد من التفاصيل.',
    'تأسست الشركة في عام 2020.',
    'أنا مساعد افتراضي، كيف يمكنني مساعدتك؟',
    'أنا مساعد افتراضي تم تطويره لمساعدتك في الحصول على المعلومات.',
    'اليوم هو: {}'.format(time.strftime('%Y-%m-%d')),
    'أنا مساعد افتراضي تم تطويره لمساعدتك.',
    'أنا مساعد افتراضي، وأعمل على تزويدك بالمعلومات التي تحتاجها.',
]

vectorizer = TfidfVectorizer()
vectorizer.fit(questions)
processed_questions = [preprocess(q) for q in questions]
bm25 = BM25Okapi(processed_questions)

class DialogueManager:
    def __init__(self):
        self.active = False
        self.paused = False
    
    def activate(self):
        self.active = True
        with stats_lock:
            stats['total_activations'] += 1
    
    def deactivate(self):
        self.active = False
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False

dialogue_manager = DialogueManager()

@lru_cache(maxsize=100)
def generate_speech(text, lang='ar'):
    filename = f'response_{uuid.uuid4().hex}.mp3'
    filepath = os.path.join(RESPONSES_DIR, filename)
    tts = gTTS(text=text, lang=lang)
    tts.save(filepath)
    return filepath

async def play_audio(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)
    pygame.mixer.quit()

def stop_audio():
    pygame.mixer.music.stop()

async def listen_for_activation(recognizer, source):
    print(f"في انتظار الكلمة المفتاحية ({KEYWORD})...")
    try:
        with config_lock:
            timeout = CONFIG['timeouts']['activation_phrase']
        audio = recognizer.listen(source, timeout=timeout)
        text = recognizer.recognize_google(audio, language='ar-EG').lower()
        return KEYWORD.lower() in text
    except sr.UnknownValueError:
        return False
    except sr.WaitTimeoutError:
        return False

def get_best_match(user_input):
    tokens = preprocess(user_input)
    scores = bm25.get_scores(tokens)
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]

async def process_question(text):
    best_idx, score = get_best_match(text)
    with config_lock:
        threshold = CONFIG['thresholds']['response_confidence']
    if score > threshold:
        response = answers[best_idx].format(time.strftime('%Y-%m-%d'))
        with stats_lock:
            stats['total_responses'] += 1
        return response
    return None

async def voice_interaction_loop():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            if dialogue_manager.paused:
                await asyncio.sleep(1)
                continue
            
            activated = await listen_for_activation(r, source)
            
            if activated:
                dialogue_manager.activate()
                await play_audio(ACTIVATION_SOUND_PATH)
                
                start_time = time.time()
                with config_lock:
                    timeout = CONFIG['timeouts']['question_listen']
                
                while time.time() - start_time < timeout:
                    try:
                        print("استمع للسؤال...")
                        audio = r.listen(source, timeout=timeout - (time.time() - start_time))
                        text = r.recognize_google(audio, language='ar-EG')
                        
                        if response := await process_question(text):
                            await play_audio(generate_speech(response))
                        else:
                            await play_audio(generate_speech("لم أفهم السؤال، سيتم تسجيله للتحسين"))
                            with stats_lock:
                                stats['total_unknown'] += 1
                            with open(UNKNOWN_QUESTIONS_FILE, 'a', encoding='utf-8') as f:
                                json.dump({'question': text, 'timestamp': time.time()}, f, ensure_ascii=False)
                                f.write('\n')
                    
                    except sr.UnknownValueError:
                        await play_audio(generate_speech("لم أسمع السؤال بوضوح"))
                    except sr.WaitTimeoutError:
                        break
                
                dialogue_manager.deactivate()

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("لوحة التحكم")
        self.current_theme = 'light'
        self.create_widgets()
        self.update_stats()
    
    def create_widgets(self):
        self.stats_frame = ttk.LabelFrame(self.root, text="إحصائيات النظام")
        self.stats_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.activations_label = ttk.Label(self.stats_frame, text="إجمالي التفعيلات: 0")
        self.activations_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.responses_label = ttk.Label(self.stats_frame, text="إجمالي الإجابات: 0")
        self.responses_label.grid(row=1, column=0, padx=5, pady=5)
        
        self.unknown_label = ttk.Label(self.stats_frame, text="الأسئلة غير المعروفة: 0")
        self.unknown_label.grid(row=2, column=0, padx=5, pady=5)
        
        # الإعدادات
        self.settings_frame = ttk.LabelFrame(self.root, text="الإعدادات")
        self.settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.act_conf_label = ttk.Label(self.settings_frame, text="حد ثقة التفعيل:")
        self.act_conf_label.grid(row=0, column=0, padx=5, pady=5)
        self.act_conf_slider = ttk.Scale(self.settings_frame, from_=0, to=1, value=CONFIG['thresholds']['activation_confidence'])
        self.act_conf_slider.grid(row=0, column=1, padx=5, pady=5)
        self.act_conf_value = ttk.Label(self.settings_frame, text=f"{CONFIG['thresholds']['activation_confidence']:.2f}")
        self.act_conf_value.grid(row=0, column=2, padx=5, pady=5)
        
        self.res_conf_label = ttk.Label(self.settings_frame, text="حد ثقة الإجابة:")
        self.res_conf_label.grid(row=1, column=0, padx=5, pady=5)
        self.res_conf_slider = ttk.Scale(self.settings_frame, from_=0, to=1, value=CONFIG['thresholds']['response_confidence'])
        self.res_conf_slider.grid(row=1, column=1, padx=5, pady=5)
        self.res_conf_value = ttk.Label(self.settings_frame, text=f"{CONFIG['thresholds']['response_confidence']:.2f}")
        self.res_conf_value.grid(row=1, column=2, padx=5, pady=5)
        
        self.sim_label = ttk.Label(self.settings_frame, text="حد التشابه:")
        self.sim_label.grid(row=2, column=0, padx=5, pady=5)
        self.sim_slider = ttk.Scale(self.settings_frame, from_=0, to=1, value=CONFIG['thresholds']['similarity'])
        self.sim_slider.grid(row=2, column=1, padx=5, pady=5)
        self.sim_value = ttk.Label(self.settings_frame, text=f"{CONFIG['thresholds']['similarity']:.2f}")
        self.sim_value.grid(row=2, column=2, padx=5, pady=5)
        
        self.act_timeout_label = ttk.Label(self.settings_frame, text="مهلة التفعيل (ثواني):")
        self.act_timeout_label.grid(row=3, column=0, padx=5, pady=5)
        self.act_timeout_entry = ttk.Entry(self.settings_frame)
        self.act_timeout_entry.insert(0, str(CONFIG['timeouts']['activation_phrase']))
        self.act_timeout_entry.grid(row=3, column=1, padx=5, pady=5)
        
        self.quest_timeout_label = ttk.Label(self.settings_frame, text="مهلة السؤال (ثواني):")
        self.quest_timeout_label.grid(row=4, column=0, padx=5, pady=5)
        self.quest_timeout_entry = ttk.Entry(self.settings_frame)
        self.quest_timeout_entry.insert(0, str(CONFIG['timeouts']['question_listen']))
        self.quest_timeout_entry.grid(row=4, column=1, padx=5, pady=5)
        
        self.apply_button = ttk.Button(self.settings_frame, text="تطبيق الإعدادات", command=self.apply_settings)
        self.apply_button.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.reset_button = ttk.Button(self.settings_frame, text="إعادة الإعدادات الافتراضية", command=self.reset_settings)
        self.reset_button.grid(row=5, column=2, pady=10)
        
        self.theme_button = ttk.Button(self.root, text="تبديل الوضع الداكن", command=self.toggle_theme)
        self.theme_button.grid(row=2, column=0, pady=10)
        
        self.stop_button = ttk.Button(self.root, text="إيقاف الإجابة الحالية", command=stop_audio)
        self.stop_button.grid(row=3, column=0, pady=10)
        
        self.pause_button = ttk.Button(self.root, text="إيقاف مؤقت", command=self.pause_system)
        self.pause_button.grid(row=4, column=0, pady=10)
        
        self.resume_button = ttk.Button(self.root, text="استئناف", command=self.resume_system)
        self.resume_button.grid(row=5, column=0, pady=10)
    
    def update_stats(self):
        with stats_lock:
            self.activations_label.config(text=f"إجمالي التفعيلات: {stats['total_activations']}")
            self.responses_label.config(text=f"إجمالي الإجابات: {stats['total_responses']}")
            self.unknown_label.config(text=f"الأسئلة غير المعروفة: {stats['total_unknown']}")
        self.root.after(1000, self.update_stats)
    
    def apply_settings(self):
        try:
            new_act_conf = float(self.act_conf_slider.get())
            new_res_conf = float(self.res_conf_slider.get())
            new_sim = float(self.sim_slider.get())
            new_act_timeout = int(self.act_timeout_entry.get())
            new_quest_timeout = int(self.quest_timeout_entry.get())
            
            with config_lock:
                CONFIG['thresholds']['activation_confidence'] = new_act_conf
                CONFIG['thresholds']['response_confidence'] = new_res_conf
                CONFIG['thresholds']['similarity'] = new_sim
                CONFIG['timeouts']['activation_phrase'] = new_act_timeout
                CONFIG['timeouts']['question_listen'] = new_quest_timeout
            
            self.act_conf_value.config(text=f"{new_act_conf:.2f}")
            self.res_conf_value.config(text=f"{new_res_conf:.2f}")
            self.sim_value.config(text=f"{new_sim:.2f}")
            
            messagebox.showinfo("نجاح", "تم تطبيق الإعدادات بنجاح!")
        except Exception as e:
            messagebox.showerror("خطأ", f"خطأ في الإعدادات: {str(e)}")
    
    def reset_settings(self):
        with config_lock:
            CONFIG.update(DEFAULT_CONFIG)
        
        self.act_conf_slider.set(CONFIG['thresholds']['activation_confidence'])
        self.res_conf_slider.set(CONFIG['thresholds']['response_confidence'])
        self.sim_slider.set(CONFIG['thresholds']['similarity'])
        self.act_timeout_entry.delete(0, tk.END)
        self.act_timeout_entry.insert(0, str(CONFIG['timeouts']['activation_phrase']))
        self.quest_timeout_entry.delete(0, tk.END)
        self.quest_timeout_entry.insert(0, str(CONFIG['timeouts']['question_listen']))
        
        self.act_conf_value.config(text=f"{CONFIG['thresholds']['activation_confidence']:.2f}")
        self.res_conf_value.config(text=f"{CONFIG['thresholds']['response_confidence']:.2f}")
        self.sim_value.config(text=f"{CONFIG['thresholds']['similarity']:.2f}")
        
        messagebox.showinfo("نجاح", "تم إعادة الإعدادات إلى الوضع الافتراضي!")
    
    def toggle_theme(self):
        if self.current_theme == 'light':
            self.root.tk_setPalette(background='#2d2d2d', foreground='#ffffff')
            self.current_theme = 'dark'
        else:
            self.root.tk_setPalette(background='SystemButtonFace', foreground='SystemWindowText')
            self.current_theme = 'light'
    
    def pause_system(self):
        dialogue_manager.pause()
        messagebox.showinfo("إيقاف مؤقت", "تم إيقاف النظام مؤقتًا.")
    
    def resume_system(self):
        dialogue_manager.resume()
        messagebox.showinfo("استئناف", "تم استئناف النظام.")

if __name__ == "__main__":
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    print("=== النظام يعمل، قل 'جيل' لتفعيله ===")
    

    voice_thread = threading.Thread(target=asyncio.run, args=(voice_interaction_loop(),), daemon=True)
    voice_thread.start()
    
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()