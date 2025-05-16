import os
import sqlite3
import hashlib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Concatenate, BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix # <--- Повертаємо імпорт

# --- Конфігурація ---
DB_FILE = 'ipso_database_binary.db'
MESSAGES_PATH = 'data/messages/messages.jsonl'
TRAIN_CSV_PATH = 'data/train.csv'
TEST_CSV_PATH = 'data/test.csv'
MODEL_WEIGHTS_DIR = 'model_weights_binary'
MODEL_WEIGHTS_FILE = os.path.join(MODEL_WEIGHTS_DIR, 'ipso_rnn_binary_lstm_30k.weights.h5')
TOKENIZER_FILE = 'tokenizer_binary_30k.json'
LABEL_ENCODER_FILE = 'label_encoder_binary.pkl'

# Параметри надійності
INITIAL_RELIABILITY = 1.0
RELIABILITY_DECREMENT = 0.02
RELIABILITY_INCREMENT = 0.01
MIN_RELIABILITY = 0.1

# Параметри моделі
VOCAB_SIZE = 30000
MAX_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.4
EPOCHS = 15
BATCH_SIZE = 64

# Бінарні класи
SAFE_CONTENT_LABEL = "SAFE_CONTENT"
IPSO_LABEL = "IPSO"
BINARY_LABELS = [SAFE_CONTENT_LABEL, IPSO_LABEL]

# --- Функції для роботи з БД---
def init_db(db_path):
    """Ініціалізує БД, створює таблиці sources та processed_messages."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir): os.makedirs(db_dir)
    conn = sqlite3.connect(db_path, check_same_thread=False); cursor = conn.cursor()
    try:
        # Таблиця джерел
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                reliability REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_id ON sources(source_id)')

        # Таблиця для оброблених повідомлень
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_hash TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit(); print(f"БД '{db_path}' ініціалізована.")
    except sqlite3.Error as e: print(f"Помилка ініціалізації БД: {e}"); conn.close(); return None
    return conn

def get_source_reliability(conn, source_id):
    if source_id is None: return INITIAL_RELIABILITY
    if conn is None: return INITIAL_RELIABILITY
    try:
        cursor = conn.cursor(); cursor.execute('SELECT reliability FROM sources WHERE source_id = ?', (str(source_id),)); result = cursor.fetchone()
        if result: return result[0]
        else: cursor.execute('INSERT OR IGNORE INTO sources (source_id, reliability) VALUES (?, ?)', (str(source_id), INITIAL_RELIABILITY)); conn.commit(); # print(f"[DB] Додано/знайдено джерело {source_id}..."); return INITIAL_RELIABILITY
    except sqlite3.Error as e: print(f"[DB] Помилка (get): {e}"); return INITIAL_RELIABILITY
    return INITIAL_RELIABILITY # Повертаємо на випадок помилки

def update_source_reliability(conn, source_id, is_ipso):
    if source_id is None: return None
    if conn is None: return None
    try:
        cursor = conn.cursor(); cursor.execute('SELECT reliability FROM sources WHERE source_id = ?', (str(source_id),)); result = cursor.fetchone()
        current_reliability = result[0] if result else INITIAL_RELIABILITY
        new_reliability = current_reliability
        if is_ipso: new_reliability = max(MIN_RELIABILITY, current_reliability - RELIABILITY_DECREMENT); change_type = "зменшено"
        else: new_reliability = min(INITIAL_RELIABILITY, current_reliability + RELIABILITY_INCREMENT); change_type = "збільшено"
        if abs(new_reliability - current_reliability) > 1e-9:
            cursor.execute('UPDATE sources SET reliability = ?, last_updated = CURRENT_TIMESTAMP WHERE source_id = ?', (new_reliability, str(source_id))); conn.commit()
            print(f"[DB] Надійність {source_id} {change_type} до {new_reliability:.3f}")
        return new_reliability
    except sqlite3.Error as e: print(f"[DB] Помилка (update): {e}"); return None


# --- Функції для обробки тексту---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\-_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Завантаження та підготовка даних ---
def load_and_preprocess_data(messages_path, train_path, test_path, binary_labels_list, vocab_size_limit):
    print("Завантаження даних...")
    try:
        with open(messages_path, 'r', encoding='utf-8') as f: messages_data = [json.loads(line) for line in f if line.strip()]
        print(f"Завантажено {len(messages_data)} повідомлень.")
    except Exception as e: print(f"ПОМИЛКА читання {messages_path}: {e}"); return None, None, None, None, None, None
    df_messages = pd.DataFrame(messages_data)
    required_msg_cols = ['text', 'source_id', 'message_id', 'published_at']
    if not all(col in df_messages.columns for col in required_msg_cols): print(f"ПОМИЛКА: відсутні колонки в {messages_path}"); return None, None, None, None, None, None
    df_messages['published_at'] = pd.to_datetime(df_messages['published_at'], unit='s', errors='coerce'); df_messages = df_messages.dropna(subset=['text', 'source_id', 'message_id']); print(f"Повідомлень після dropna: {len(df_messages)}")
    try:
        df_train_sources = pd.read_csv(train_path); df_test_sources = pd.read_csv(test_path)
        print(f"Завантажено {len(df_train_sources)} трен. і {len(df_test_sources)} тест. джерел.")
    except Exception as e: print(f"ПОМИЛКА читання CSV: {e}"); return None, None, None, None, None, None
    if 'source_id' not in df_train_sources.columns or 'source_category' not in df_train_sources.columns: print(f"ПОМИЛКА: У {train_path} відсутні колонки."); return None, None, None, None, None, None
    if 'source_id' not in df_test_sources.columns: print(f"ПОМИЛКА: У {test_path} відсутня 'source_id'."); return None, None, None, None, None, None
    df_sources = pd.concat([df_train_sources, df_test_sources], ignore_index=True).drop_duplicates(subset=['source_id'])
    df_sources['label'] = df_sources['source_category'].apply(lambda x: SAFE_CONTENT_LABEL if x == SAFE_CONTENT_LABEL else (IPSO_LABEL if pd.notna(x) else None))
    print("Об'єднання даних...")
    df = pd.merge(df_messages, df_sources[['source_id', 'label']], on='source_id', how='inner')
    df = df.dropna(subset=['label'])
    print(f"Повідомлень після об'єднання/dropna(label): {len(df)}")
    if df.empty: print("ПОМИЛКА: Дані порожні."); return None, None, None, None, None, None
    print("Очищення тексту...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"Після очищення тексту: {len(df)}")
    if df.empty: print("ПОМИЛКА: Дані порожні після очищення."); return None, None, None, None, None, None
    print("\nРозподіл бінарних класів у даних:"); print(df['label'].value_counts()); print("-" * 30)
    print("Кодування міток...")
    label_encoder = LabelEncoder(); label_encoder.fit(binary_labels_list)
    y = label_encoder.transform(df['label'])
    print(f"Мітки закодовано: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")
    print("Токенізація...")
    tokenizer = Tokenizer(num_words=vocab_size_limit, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    X_seq = tokenizer.texts_to_sequences(df['cleaned_text'])
    effective_vocab_size = tokenizer.num_words if tokenizer.num_words else (len(tokenizer.word_index) + 1)
    print(f"Ефективний розмір словника: {effective_vocab_size}")
    print("Паддинг..."); X_pad = pad_sequences(X_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    print("Розділення даних...")
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)
    print("-" * 30); print(f"Підготовка даних завершена: Навч={len(X_train)}, Валід={len(X_val)}"); print("-" * 30)
    return X_train, X_val, y_train, y_val, tokenizer, label_encoder

# --- Побудова моделі ---
def build_ipso_rnn_model(vocab_size, embedding_dim, max_length, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE):
    text_input=Input(shape=(max_length,), name='text_input'); reliability_input=Input(shape=(1,), name='reliability_input')
    embedding_layer=Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')(text_input)
    lstm_layer=LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='lstm')(embedding_layer)
    concatenated=Concatenate(name='concatenate')([lstm_layer, reliability_input])
    x=Dense(128, activation='relu', name='dense_1')(concatenated); x=BatchNormalization()(x); x=Dropout(dropout_rate)(x)
    x=Dense(64, activation='relu', name='dense_2')(x); x=BatchNormalization()(x); x=Dropout(dropout_rate)(x)
    output=Dense(1, activation='sigmoid', name='output')(x)
    model=Model(inputs=[text_input, reliability_input], outputs=output, name='IPSO_RNN_LSTM_Binary_Model')
    optimizer=tf.keras.optimizers.Adam(); model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print("Модель LSTM для бінарної класифікації побудована:")
    model.summary(line_length=100)
    return model

# --- Навчання моделі ---
def train_ipso_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, model_weights_file=MODEL_WEIGHTS_FILE, class_weights=None):
    print("\n--- Початок навчання моделі ---")
    train_reliability=np.full((X_train.shape[0], 1), INITIAL_RELIABILITY, dtype='float32'); val_reliability=np.full((X_val.shape[0], 1), INITIAL_RELIABILITY, dtype='float32')
    if not os.path.exists(MODEL_WEIGHTS_DIR): print(f"Створення директорії: {MODEL_WEIGHTS_DIR}"); os.makedirs(MODEL_WEIGHTS_DIR)
    early_stopping=EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    model_checkpoint=ModelCheckpoint(model_weights_file, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
    print(f"Параметри: епохи={epochs}, батч={batch_size}")
    if class_weights: print(f"Використовуються ваги класів: {class_weights}")
    else: print("Зважування класів не використовується.")
    print(f"Збереження ваг у: {model_weights_file}"); print("-" * 30)
    history = model.fit([X_train, train_reliability], y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_val, val_reliability], y_val), callbacks=[early_stopping, model_checkpoint], class_weight=class_weights, verbose=1)
    print("-" * 30); print(f"Навчання завершено.")
    if os.path.exists(model_weights_file): print(f"Найкращі ваги збережено: {model_weights_file}")
    else: print(f"УВАГА: Файл ваг {model_weights_file} не збережено.")
    return model, history

# --- Функція передбачення---
def predict_single_message(model, tokenizer, label_encoder, db_conn, text, source_id, max_length=MAX_LENGTH, threshold=0.5):
    if not text or not isinstance(text, str): print("[Predict] Помилка: Порожній текст."); r = get_source_reliability(db_conn, source_id); return False, r
    cleaned_text = clean_text(text)
    if not cleaned_text: print("[Predict] Помилка: Текст порожній після очищення."); r = get_source_reliability(db_conn, source_id); return False, r
    current_reliability = get_source_reliability(db_conn, source_id); message_processed = False; message_hash = None
    if source_id and db_conn:
        try:
            hash_input = f"{str(source_id)}::{cleaned_text}"; message_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            cursor = db_conn.cursor(); cursor.execute("SELECT 1 FROM processed_messages WHERE message_hash = ?", (message_hash,)); result = cursor.fetchone()
            if result: message_processed = True; print(f"[Predict] Повідомлення від {source_id} (хеш: {message_hash[:8]}...) вже оброблено.")
        except Exception as e_hash_check: print(f"[Predict] Помилка перевірки хешу: {e_hash_check}")
    try: sequence = tokenizer.texts_to_sequences([cleaned_text]); padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    except Exception as e_tok: print(f"[Predict] Помилка токенізації: {e_tok}"); return False, current_reliability
    reliability_input = np.array([[current_reliability]], dtype='float32'); is_ipso = False; confidence = 0.0
    try:
        if padded_sequence.shape != (1, max_length): raise ValueError(f"Форма тексту: {padded_sequence.shape}")
        if reliability_input.shape != (1, 1): raise ValueError(f"Форма надійності: {reliability_input.shape}")
        prediction_prob_ipso = model.predict([padded_sequence, reliability_input], verbose=0)[0][0]
        confidence = prediction_prob_ipso; is_ipso = prediction_prob_ipso >= threshold
    except Exception as e_pred: print(f"[Predict] Помилка передбачення: {e_pred}"); return False, current_reliability
    final_reliability = current_reliability
    if not message_processed and source_id is not None:
        updated_reliability_val = update_source_reliability(db_conn, source_id, is_ipso)
        final_reliability = updated_reliability_val if updated_reliability_val is not None else current_reliability
        if message_hash and db_conn:
             try: cursor = db_conn.cursor(); cursor.execute("INSERT OR IGNORE INTO processed_messages (message_hash) VALUES (?)", (message_hash,)); db_conn.commit()
             except Exception as e_hash_insert: print(f"[Predict] Помилка збереження хешу: {e_hash_insert}")
    elif message_processed: final_reliability = current_reliability
    print(f"[Predict] Аналіз: Текст='{cleaned_text[:50]}...', Джерело={source_id}, Ймовірн_IPSO={confidence:.3f}, IS_IPSO={is_ipso}(Поріг={threshold}), Надійність={final_reliability:.3f}")
    return is_ipso, final_reliability

# # --- Функція побудови графіків---
# def plot_training_history(history, model_name="Model"):
#     if history is None or not hasattr(history, 'history') or not history.history: print("Неможливо побудувати графіки."); return
#     history_dict=history.history; acc=history_dict.get('accuracy'); val_acc=history_dict.get('val_accuracy'); loss=history_dict.get('loss'); val_loss=history_dict.get('val_loss')
#     required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss'];
#     if not all(key in history_dict for key in required_keys): print(f"Відсутні метрики: {[k for k in required_keys if k not in history_dict]}."); return
#     if not acc or not val_acc or not loss or not val_loss: print("Списки метрик порожні."); return
#     epochs_range = range(1, len(acc)+1); plt.figure(figsize=(14,6))
#     plt.subplot(1,2,1); plt.plot(epochs_range, loss, 'bo-', label='Втрати Навчання'); plt.plot(epochs_range, val_loss, 'ro-', label='Втрати Валідації'); plt.title(f'{model_name} - Втрати'); plt.xlabel('Епохи'); plt.ylabel('Втрати'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
#     plt.subplot(1,2,2); plt.plot(epochs_range, acc, 'bo-', label='Точність Навчання'); plt.plot(epochs_range, val_acc, 'ro-', label='Точність Валідації'); plt.title(f'{model_name} - Точність'); plt.xlabel('Епохи'); plt.ylabel('Точність'); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
#     plt.suptitle(f'Історія навчання: {model_name}', fontsize=14); plt.tight_layout(rect=[0,0.03,1,0.95])
#     try: plot_filename = f"{model_name.replace(' ','_')}_binary_30k_history.png"; plt.savefig(plot_filename); print(f"Графіки збережено: {plot_filename}") # <--- Оновлено ім'я файлу
#     except Exception as e_plot: print(f"Помилка збереження графіків: {e_plot}")
#     plt.show()

# --- Функція завантаження ресурсів---
def load_ai_resources(
        model_weights_path=MODEL_WEIGHTS_FILE,
        tokenizer_path=TOKENIZER_FILE,
        encoder_path=LABEL_ENCODER_FILE,
        db_path=DB_FILE):
    print("\n[AI Loader] Завантаження ресурсів ШІ...")
    loaded_model, loaded_tokenizer, loaded_encoder, db_conn = None, None, None, None
    try:
        required = {'Ваги': model_weights_path, 'Токенізатор': tokenizer_path, 'Енкодер': encoder_path}
        print("[AI Loader] Перевірка файлів..."); missing = [n for n, p in required.items() if not os.path.exists(p)]
        if missing: raise FileNotFoundError(f"Відсутні файли: {', '.join(missing)}")
        print("[AI Loader] Файли знайдено.")
        db_conn = init_db(db_path);
        if db_conn is None: raise ConnectionError(f"Не вдалося підключитися до БД: {db_path}")
        print(f"[AI Loader] Завантаження {tokenizer_path}...");
        with open(tokenizer_path, 'r', encoding='utf-8') as f: loaded_tokenizer = tokenizer_from_json(f.read())
        print(f"[AI Loader] Завантаження {encoder_path}...");
        with open(encoder_path, 'rb') as f: loaded_encoder = pickle.load(f)
        print("[AI Loader] Токенізатор та енкодер завантажено.")
        if list(loaded_encoder.classes_) != BINARY_LABELS: print("!!! ПОПЕРЕДЖЕННЯ: Класи в енкодері не бінарні!")
        build_vocab_size = VOCAB_SIZE # <--- Використовуємо константу
        print(f"[AI Loader] Побудова моделі (словник={build_vocab_size})...");
        loaded_model = build_ipso_rnn_model(build_vocab_size, EMBEDDING_DIM, MAX_LENGTH, LSTM_UNITS, DROPOUT_RATE) # <--- Використовуємо звичайний LSTM
        print(f"[AI Loader] Завантаження ваг {model_weights_path}...")
        loaded_model.load_weights(model_weights_path)
        print("[AI Loader] Прогрів моделі...");
        _ = loaded_model.predict([np.zeros((1, MAX_LENGTH)), np.zeros((1, 1))], verbose=0)
        print("[AI Loader] Модель завантажена та готова.")
        return loaded_model, loaded_tokenizer, loaded_encoder, db_conn
    except Exception as e:
        print(f"[AI Loader] КРИТИЧНА ПОМИЛКА завантаження: {e}")
        if db_conn: db_conn.close()
        return None, None, None, None

# --- Основний блок ---
if __name__ == "__main__":
    db_connection = init_db(DB_FILE);
    if db_connection is None: print("Помилка БД. Вихід."); sys.exit(1)
    tokenizer, label_encoder, model, history = None, None, None, None
    load_successful = False;

    # Спроба завантажити компоненти
    if os.path.exists(TOKENIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE) and os.path.exists(MODEL_WEIGHTS_FILE):
        print("Виявлено компоненти. Завантаження...");
        try:
            model, tokenizer, label_encoder, db_connection_loaded = load_ai_resources(
                MODEL_WEIGHTS_FILE, TOKENIZER_FILE, LABEL_ENCODER_FILE, DB_FILE
            )
            if model and tokenizer and label_encoder and db_connection_loaded:
                db_connection = db_connection_loaded
                load_successful = True
                print("Ресурси успішно завантажено.")
                # # ---> Оцінка завантаженої моделі на валідації <---
                # print("\nЗавантаження валідаційних даних для оцінки...")
                # try:
                #      X_val = np.load('X_val_binary.npy')
                #      y_val = np.load('y_val_binary.npy')
                #      val_reliability = np.load('val_reliability_binary.npy')
                #      print(f"Валідаційні дані завантажено: X({X_val.shape}), y({y_val.shape})")

                #      print("\n--- Оцінка завантаженої моделі на валідаційній вибірці (поріг=0.5) ---")
                #      y_pred_probs_val = model.predict([X_val, val_reliability], batch_size=BATCH_SIZE, verbose=0)
                #      if y_pred_probs_val.ndim > 1: y_pred_probs_val = y_pred_probs_val.flatten()
                #      y_pred_val_binary = (y_pred_probs_val >= 0.5).astype(int)

                #      print("Матриця помилок:")
                #      cm = confusion_matrix(y_val, y_pred_val_binary)
                #      print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))
                #      print("\nЗвіт класифікації:")
                #      print(classification_report(y_val, y_pred_val_binary, target_names=label_encoder.classes_, zero_division=0))
                # except FileNotFoundError:
                #      print("ПОМИЛКА: .npy файли валідаційних даних не знайдено. Оцінка неможлива.")
                # except Exception as e_eval_load:
                #      print(f"Помилка під час оцінки завантаженої моделі: {e_eval_load}")
                # # ---> КІНЕЦЬ ОЦІНКИ <---

            else: print("Функція load_ai_resources повернула помилку."); load_successful = False
        except Exception as e_load_func: print(f"Помилка виклику load_ai_resources: {e_load_func}"); load_successful = False
    else: print("Збережені компоненти не знайдені.")

    # Якщо не завантажили -> Навчання
    if not load_successful:
        print("\n--- Запуск обробки даних та навчання моделі ---")
        X_train, X_val, y_train, y_val, tokenizer, label_encoder = load_and_preprocess_data(
            MESSAGES_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, BINARY_LABELS, VOCAB_SIZE # Передаємо новий VOCAB_SIZE
        )
        if tokenizer is None or label_encoder is None or y_train is None:
            print("\nПОМИЛКА обробки даних. Вихід.");
            if db_connection: db_connection.close(); sys.exit(1)
        else: print("\nДані для навчання підготовлено.")

        # Розрахунок ваг класів
        class_weights_dict = None
        try:
            print("\nРозрахунок ваг класів..."); y_integers = y_train
            unique_classes, counts = np.unique(y_integers, return_counts=True)
            print(f"Розподіл класів у y_train: {dict(zip(label_encoder.inverse_transform(unique_classes), counts))}")
            if len(unique_classes) == 2:
                weights = class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_integers)
                class_weights_dict = dict(zip(unique_classes, weights)); print(f"Ваги розраховано: {class_weights_dict}")
            else: print("К-сть класів не 2, ваги не розраховано.")
        except Exception as e_cw: print(f"ПОМИЛКА розрахунку ваг: {e_cw}."); class_weights_dict = None

        build_vocab_size = VOCAB_SIZE; print(f"\nРозмір словника для моделі: {build_vocab_size}")
        try:
            model = build_ipso_rnn_model(build_vocab_size, EMBEDDING_DIM, MAX_LENGTH, LSTM_UNITS, DROPOUT_RATE)
            model, history = train_ipso_model(model, X_train, y_train, X_val, y_val,
                                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                                              model_weights_file=MODEL_WEIGHTS_FILE,
                                              class_weights=class_weights_dict)
            if model is None: raise ValueError("Навчання повернуло None.")
            print("Нову модель навчено.")

            # Збереження компонентів
            print("\nЗбереження токенізатора та кодувальника міток...")
            try:
                with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f: f.write(tokenizer.to_json())
                print(f"Токенізатор збережено: {TOKENIZER_FILE}")
                with open(LABEL_ENCODER_FILE, 'wb') as f: pickle.dump(label_encoder, f)
                print(f"Кодувальник міток збережено: {LABEL_ENCODER_FILE}")
                # ---> Збереження валідаційних даних <---
                print("\nЗбереження валідаційних даних для аналізу порогу...")
                np.save('X_val_binary.npy', X_val)
                np.save('y_val_binary.npy', y_val)
                val_reliability_to_save = np.full((X_val.shape[0], 1), INITIAL_RELIABILITY, dtype='float32')
                np.save('val_reliability_binary.npy', val_reliability_to_save)
                print("Валідаційні дані збережено у .npy файли.")
            except Exception as e_save: print(f"ПОМИЛКА збереження: {e_save}")

            # # Побудова графіків
            # if history: plot_training_history(history, model_name="IPSO_RNN_LSTM_Binary_30k")

            # # ---> Оцінка моделі ПІСЛЯ НАВЧАННЯ <---
            # if model and X_val is not None and y_val is not None:
            #     print("\n--- Оцінка моделі на валідаційній вибірці (поріг=0.5) ---")
            #     try:
            #         val_reliability = np.full((X_val.shape[0], 1), INITIAL_RELIABILITY, dtype='float32')
            #         y_pred_probs_val = model.predict([X_val, val_reliability], batch_size=BATCH_SIZE, verbose=0)
            #         if y_pred_probs_val.ndim > 1: y_pred_probs_val = y_pred_probs_val.flatten()
            #         y_pred_val_binary = (y_pred_probs_val >= 0.5).astype(int)

            #         print("Матриця помилок:")
            #         cm = confusion_matrix(y_val, y_pred_val_binary)
            #         print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))
            #         print("\nЗвіт класифікації:")
            #         print(classification_report(y_val, y_pred_val_binary, target_names=label_encoder.classes_, zero_division=0))
            #     except Exception as e_eval:
            #         print(f"Помилка під час оцінки моделі: {e_eval}")
            # ---> КІНЕЦЬ ОЦІНКИ <---

        except Exception as e_train:
             print(f"\nКРИТИЧНА ПОМИЛКА навчання: {e_train}")
             if db_connection: db_connection.close(); print("\nБД закрито."); sys.exit(1)

    # # Тестові виклики (використовуємо стандартний поріг 0.5)
    # if model and tokenizer and label_encoder:
    #     print(f"\n--- Тестові виклики функції передбачення (поріг=0.5) ---")
    #     # Блок тестових викликів залишається без змін
    #     try:
    #         test_text_1="Срочно! Распространите! Наши герои..."; test_source_id_1="channel_test_b_1"
    #         print(f"\nТест 1: Текст='{test_text_1[:50]}...', Джерело='{test_source_id_1}'")
    #         is_ipso1, reliability1 = predict_single_message(model,tokenizer,label_encoder,db_connection,test_text_1,test_source_id_1)
    #         print(f"Результат 1: is_ipso={is_ipso1}, reliability={reliability1:.3f}")

    #         test_text_2="Сьогодні чудовий день для роботи."; test_source_id_2 = test_source_id_1
    #         print(f"\nТест 2: Текст='{test_text_2[:50]}...', Джерело='{test_source_id_2}' (SAFE)")
    #         is_ipso2, reliability2 = predict_single_message(model,tokenizer,label_encoder,db_connection,test_text_2,test_source_id_2)
    #         print(f"Результат 2: is_ipso={is_ipso2}, reliability={reliability2:.3f}")

    #         test_text_3="Срочно! Распространите! Наши герои..."; test_source_id_3 = test_source_id_1
    #         print(f"\nТест 3: Текст='{test_text_3[:50]}...', Джерело='{test_source_id_3}' (Дублікат)")
    #         is_ipso3, reliability3 = predict_single_message(model,tokenizer,label_encoder,db_connection,test_text_3,test_source_id_3)
    #         print(f"Результат 3: is_ipso={is_ipso3}, reliability={reliability3:.3f}")

    #         test_text_4="Всі на вила! Терпіти більше не можна!"; test_source_id_4 = None
    #         print(f"\nТест 4: Текст='{test_text_4[:50]}...', Джерело=None (пряме)")
    #         is_ipso4, reliability4 = predict_single_message(model,tokenizer,label_encoder,db_connection,test_text_4,test_source_id_4)
    #         print(f"Результат 4: is_ipso={is_ipso4}, reliability={reliability4:.3f}")

    #         test_text_5="Який гарний котик."; test_source_id_5 = None
    #         print(f"\nТест 5: Текст='{test_text_5[:50]}...', Джерело=None (пряме)")
    #         is_ipso5, reliability5 = predict_single_message(model,tokenizer,label_encoder,db_connection,test_text_5,test_source_id_5)
    #         print(f"Результат 5: is_ipso={is_ipso5}, reliability={reliability5:.3f}")

    #     except Exception as e_pred: print(f"\nПомилка тестів: {e_pred}")
    # else: print("\nПропуск тестів: компоненти не завантажено/навчено.")

    # Закриття БД
    if db_connection:
        try: db_connection.close(); print("\nЗ'єднання з БД закрито.")
        except sqlite3.Error as e_close: print(f"Помилка закриття БД: {e_close}")

    print("\nРоботу скрипта завершено.")
    