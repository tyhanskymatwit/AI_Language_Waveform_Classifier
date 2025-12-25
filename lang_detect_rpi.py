"""
50-Language Detection Training
"""
import numpy as np
import librosa
import soundfile as sf
import matplotlib
# Use non-interactive backend so plotting does not block or require a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time
from pathlib import Path

# CONFIGURATION

SAMPLE_RATE = 16000
N_MELS = 128
MAX_LENGTH = 128
SAMPLES_PER_WORD = 10


MIN_WORD_FOLDERS = 20
MIN_TOTAL_CLIPS = 200
TOP_K_LANGUAGES = 10
LANGUAGES = [
    'ar', 'br', 'ca', 'cnh', 'cs', 'cv', 'cy', 'de', 'el', 'en',
    'eo', 'es', 'et', 'eu', 'fa', 'fr', 'fy-NL', 'ga-IE', 'ha', 'ia',
    'id', 'it', 'ka', 'ky', 'lt', 'lv', 'mn', 'mt', 'nl', 'pl',
    'pt', 'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sk', 'sl',
    'sv-SE', 'tr', 'tt', 'uk', 'zh-CN'
]

# Full language names
LANGUAGE_NAMES = {
    'ar': 'Arabic',
    'br': 'Breton',
    'ca': 'Catalan',
    'cnh': 'Hakha Chin',
    'cs': 'Czech',
    'cv': 'Chuvash',
    'cy': 'Welsh',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'eo': 'Esperanto',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fr': 'French',
    'fy-NL': 'Frisian (Netherlands)',
    'ga-IE': 'Irish (Ireland)',
    'ha': 'Hausa',
    'ia': 'Interlingua',
    'id': 'Indonesian',
    'it': 'Italian',
    'ka': 'Georgian',
    'ky': 'Kyrgyz',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mn': 'Mongolian',
    'mt': 'Maltese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'rm-sursilv': 'Sursilvan',
    'rm-vallader': 'Vallader',
    'ro': 'Romanian',
    'ru': 'Russian',
    'rw': 'Kinyarwanda',
    'sah': 'Sakha',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sv-SE': 'Swedish (Sweden)',
    'tr': 'Turkish',
    'tt': 'Tatar',
    'uk': 'Ukrainian',
    'zh-CN': 'Chinese (China)'
}

# FEATURE EXTRACTION

def extract_features_from_audio(audio, sr):
    """Extract mel-spectrogram features from audio array"""
    try:
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=N_MELS,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        # Pad or truncate to fixed length
        if mel_spec_db.shape[1] < MAX_LENGTH:
            pad_width = MAX_LENGTH - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :MAX_LENGTH]
        
        return mel_spec_db
    except:
        return None

# DATASET LOADING

def load_mlcommons_dataset(base_dir, max_samples_per_lang=10000, samples_per_word=3, languages=None):
    """
    Load MLCommons dataset from the specified directory structure
    
    Parameters:
    - base_dir: path to 'tyhanskym/mlcommons_data' (parent of 'audio' folder)
    - max_samples_per_lang: maximum audio clips per language (to balance dataset)
    - languages: list of language codes to use (default: all 50)
    - samples_per_word: Number of audio files to take from each word folder
    
    Structure:
    base_dir/
      audio/
        en/
          clips/
            word1/
              clip1.opus
              clip2.opus
            word2/
              ...
        it/
          clips/
            ...
        ... (there's no way I can do this for millions of words but I guess we'll find out)
      alignments/
      splits/
      *.json
    """
    
    if languages is None:
        languages = LANGUAGES

    samples_per_word = int(samples_per_word)

    print("=" * 70)
    print("Loading MLCommons Multilingual Spoken Words Dataset")
    print(f"   Base directory: {base_dir}")
    print(f"   Languages: {len(languages)}")
    print(f"   Max samples per language: {max_samples_per_lang} (cap on samples per language; languages with fewer clips than this will be removed before loading)")
    print(f"   Samples per word folder: {samples_per_word}")
    print("=" * 70)
    
    audio_dir = os.path.join(base_dir, 'audio')
    
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    X = []
    y = []
    
    for lang_idx, lang_code in enumerate(languages):
        lang_dir = os.path.join(audio_dir, lang_code, 'clips')
        
        if not os.path.exists(lang_dir):
            print(f"Warning: {lang_code} not found at {lang_dir}")
            continue
        
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        print(f"\nProcessing {lang_name} ({lang_code})... [{lang_idx+1}/{len(languages)}]")
        
        # Iterate all word folders and sample up to `samples_per_word` files per word
        word_dirs = [os.path.join(lang_dir, d) for d in os.listdir(lang_dir)
                     if os.path.isdir(os.path.join(lang_dir, d))]
        total_word_folders = len(word_dirs)
        print(f"   Found {total_word_folders} word folders (processing all folders; sampling {samples_per_word} files per word)")
        
        successful = 0
        for word_dir in tqdm(word_dirs, desc=f"  Processing words for {lang_code}"):
            # collect .opus in this word folder
            files = glob.glob(os.path.join(word_dir, '*.opus'))
            if not files:
                continue
            # sample up to samples_per_word from this word folder
            if len(files) > samples_per_word:
                files = list(np.random.choice(files, samples_per_word, replace=False))
            for audio_file in files:
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
                    
                    # Pad/truncate to exactly 1 second
                    target_length = SAMPLE_RATE
                    if len(audio) < target_length:
                        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                    else:
                        audio = audio[:target_length]
                    
                    # Extract features
                    features = extract_features_from_audio(audio, sr)
                    if features is not None:
                        X.append(features)
                        y.append(lang_idx)
                        successful += 1   
                except Exception:
                    continue
            # stop if we've reached max for this language
            if max_samples_per_lang is not None and successful >= max_samples_per_lang:
                break
        print(f"   Successfully loaded {successful} clips for {lang_name}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    print("\n" + "=" * 70)
    print(f"Dataset loaded successfully!")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features shape: {X.shape}")
    print(f"   Languages: {len(np.unique(y))}")
    print("=" * 70)
    
    return X, y, languages

# MODEL ARCHITECTURE


def filter_languages_by_data(base_dir, languages, min_word_folders=MIN_WORD_FOLDERS, min_total_clips=MIN_TOTAL_CLIPS, top_k=None):
    """Return a filtered list of languages that meet minimum data thresholds.

    Scans `audio/<lang>/clips` and counts word folders and total .opus clips.
    Languages that do not meet both thresholds are excluded. If `top_k` is
    provided, the function will select the top-k languages by `total_clips`.
    """
    audio_dir = os.path.join(base_dir, 'audio')
    selected = []
    stats = {}

    for lang in languages:
        lang_clips_dir = os.path.join(audio_dir, lang, 'clips')
        if not os.path.exists(lang_clips_dir):
            stats[lang] = {'word_folders': 0, 'total_clips': 0}
            continue

        # Count word folders
        word_folders = [d for d in os.listdir(lang_clips_dir)
                        if os.path.isdir(os.path.join(lang_clips_dir, d))]
        total_clips = 0
        for wd in word_folders:
            files = glob.glob(os.path.join(lang_clips_dir, wd, '*.opus'))
            total_clips += len(files)

        stats[lang] = {'word_folders': len(word_folders), 'total_clips': total_clips}

        if len(word_folders) >= min_word_folders and total_clips >= min_total_clips:
            selected.append(lang)

    # Print a short summary
    print("\nLanguage data stats (lang: word_folders / total_clips):")
    for l, s in stats.items():
        print(f"  {l}: {s['word_folders']} / {s['total_clips']}")

    # If top_k is set, reduce to the top-k languages by total_clips
    if top_k is not None and len(selected) > top_k:
        # sort selected by total_clips desc and keep top_k
        selected = sorted(selected, key=lambda l: stats[l]['total_clips'], reverse=True)[:top_k]
        print(f"\nSelecting top {top_k} languages by available clips")

    removed = [l for l in languages if l not in selected]
    if removed:
        print(f"\nRemoving {len(removed)} languages due to insufficient data or top-k selection: {removed}")
    else:
        print("\nAll languages meet the minimum data thresholds (and top-k selection if used).")

    return selected

def build_model(input_shape=(N_MELS, MAX_LENGTH, 1), num_classes=50):
    """Build CNN model for language classification"""
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Conv Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# VISUALIZATION

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    fig_filename = PLOTS_DIR / f"training_history_{int(time.time())}.png"
    fig.savefig(fig_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_filename}")

def plot_confusion_matrix(y_true, y_pred, languages, top_n=20):
    """Plot confusion matrix (show only top N languages for readability)"""
    cm = confusion_matrix(y_true, y_pred)
    
    # If too many languages, show only most common ones
    if len(languages) > top_n:
        # Find top N most common languages in test set
        unique, counts = np.unique(y_true, return_counts=True)
        top_indices = unique[np.argsort(counts)[-top_n:]]
        
        # Filter confusion matrix
        mask = np.isin(y_true, top_indices) & np.isin(y_pred, top_indices)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Remap indices
        idx_map = {old: new for new, old in enumerate(sorted(top_indices))}
        y_true_mapped = np.array([idx_map[val] for val in y_true_filtered])
        y_pred_mapped = np.array([idx_map[val] for val in y_pred_filtered])
        
        cm = confusion_matrix(y_true_mapped, y_pred_mapped)
        language_labels = [LANGUAGE_NAMES.get(languages[i], languages[i]) for i in sorted(top_indices)]
        title = f'Confusion Matrix (Top {top_n} Languages)'
    else:
        language_labels = [LANGUAGE_NAMES.get(l, l) for l in languages]
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=language_labels, yticklabels=language_labels,
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig = plt.gcf()
    fig_filename = PLOTS_DIR / f"confusion_matrix_{int(time.time())}.png"
    fig.savefig(fig_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_filename}")

def visualize_sample_spectrograms(X, y, languages, num_samples=8):
    """Visualize sample mel-spectrograms from different languages"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    
    # Get random samples from different languages
    unique_langs = np.unique(y)
    selected_langs = np.random.choice(unique_langs, min(num_samples, len(unique_langs)), replace=False)
    
    for i, lang_idx in enumerate(selected_langs):
        if i >= num_samples:
            break
        
        # Get random sample from this language
        indices = np.where(y == lang_idx)[0]
        idx = np.random.choice(indices)
        
        axes[i].imshow(X[idx, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        lang_name = LANGUAGE_NAMES.get(languages[lang_idx], languages[lang_idx])
        axes[i].set_title(f'{lang_name} ({languages[lang_idx]})')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Mel Frequency')
    
    plt.tight_layout()
    fig_filename = PLOTS_DIR / f"spectrograms_{int(time.time())}.png"
    fig.savefig(fig_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_filename}")

def plot_class_distribution(y, languages):
    """Plot distribution of samples across languages"""
    unique, counts = np.unique(y, return_counts=True)
    lang_names = [LANGUAGE_NAMES.get(languages[i], languages[i]) for i in unique]
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    lang_names = [lang_names[i] for i in sorted_indices]
    counts = counts[sorted_indices]
    
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), lang_names, rotation=90)
    plt.xlabel('Language')
    plt.ylabel('Number of Samples')
    plt.title(f'Dataset Distribution Across {len(languages)} Languages')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig_filename = PLOTS_DIR / f"class_distribution_{int(time.time())}.png"
    fig.savefig(fig_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {fig_filename}")

# MAIN TRAINING PIPELINE

print("=" * 70)
print("LANGUAGE DETECTION MODEL TRAINING")
print("Using MLCommons Multilingual Spoken Words Dataset")
print("=" * 70)

# STEP 1: Configure paths
print("\nSTEP 1: Configuration")
print("-" * 70)

BASE_DIR = '/home/tyhanskym/mlcommons_data'

MAX_SAMPLES_PER_LANG = 10000  # Increase to allow more samples per language

print(f"Base directory: {BASE_DIR}")
print(f"Samples per language: {MAX_SAMPLES_PER_LANG}")

# Filter languages that don't meet minimum data requirements
selected_languages = filter_languages_by_data(
    BASE_DIR,
    LANGUAGES,
    min_word_folders=MIN_WORD_FOLDERS,
    min_total_clips=MIN_TOTAL_CLIPS,
    top_k=TOP_K_LANGUAGES,
)
if not selected_languages:
    raise RuntimeError("No languages meet the minimum data thresholds. Lower thresholds or add data.")
print(f"Using {len(selected_languages)} languages: {selected_languages}")

# Create output directory for plots so scripts can save figures non-interactively
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load Dataset
print("\nSTEP 2: Loading Dataset")
print("-" * 70)
print("This may take 10-30 minutes depending on your system...")

X, y, active_languages = load_mlcommons_dataset(
    BASE_DIR,
    max_samples_per_lang=MAX_SAMPLES_PER_LANG,
    samples_per_word=SAMPLES_PER_WORD,
    languages=selected_languages
)

# Visualize distribution
plot_class_distribution(y, active_languages)

# Visualize sample spectrograms
print("\nVisualizing sample spectrograms...")
visualize_sample_spectrograms(X, y, active_languages)

# STEP 3: Split Dataset
print("\nSTEP 3: Splitting Dataset")
print("-" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# STEP 4: Build Model
print("\nSTEP 4: Building Model")
print("-" * 70)
num_classes = len(active_languages)
model = build_model(num_classes=num_classes)
print(model.summary())

# STEP 5: Train Model
print("\nSTEP 5: Training Model")
print("-" * 70)
print("This may take 1-3 hours depending on your system...")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7
)

best_model_filename = f'best_model_{num_classes}lang.h5'
checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_filename,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,  # Increase if you have more GPU memory
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# STEP 6: Evaluate Model
print("\nSTEP 6: Evaluating Model")
print("-" * 70)

plot_training_history(history)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Top-5 accuracy (useful for 50 classes)
y_pred_probs = model.predict(X_test)
top5_correct = 0
for i in range(len(y_test)):
    if y_test[i] in np.argsort(y_pred_probs[i])[-5:]:
        top5_correct += 1
top5_accuracy = top5_correct / len(y_test)
print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")

# Classification report
print("\nClassification Report:")
target_names = [LANGUAGE_NAMES.get(l, l) for l in active_languages]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix (top 20 languages for readability)
plot_confusion_matrix(y_test, y_pred, active_languages, top_n=20)

# STEP 7: Save Model
print("\nSTEP 7: Saving Model")
print("-" * 70)
model_save_filename = f'language_detector_{num_classes}lang.h5'
model.save(model_save_filename)

# Save language mapping
import pickle
with open('language_mapping.pkl', 'wb') as f:
    pickle.dump({
        'languages': active_languages,
        'language_names': LANGUAGE_NAMES
    }, f)

print(f"Model saved as '{model_save_filename}'")
print("Language mapping saved as 'language_mapping.pkl'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nDataset Info:")
print(f"  Source: MLCommons Multilingual Spoken Words")
print(f"  License: CC-BY 4.0")
print(f"  Total clips used: {len(X)}")
print(f"  Languages: {len(active_languages)}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Top-5 Accuracy: {top5_accuracy*100:.2f}%")
print("=" * 70)
