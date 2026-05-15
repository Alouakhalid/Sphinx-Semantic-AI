import os
import json
import tensorflow as tf
import keras
from tensorflow.keras import layers
from transformers import BertTokenizer
import numpy as np
from datetime import datetime

@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(max_len, d_model)

    def call(self, x):
        seq_len = keras.ops.shape(x)[1]
        positions = keras.ops.arange(seq_len)
        return self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config

@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(4 * self.d_model, activation="gelu"),
            layers.Dense(self.d_model)
        ])
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()
        self.drop = layers.Dropout(self.dropout_rate)

    def call(self, x):
        a = self.att(x, x)
        x = self.ln1(x + self.drop(a))
        f = self.ffn(x)
        x = self.ln2(x + self.drop(f))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate
        })
        return config

@keras.saving.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, vocab_size, max_len, d_model, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers_count = num_layers
        self.dropout_rate = dropout

        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(max_len, d_model)
        self.blocks = [
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ]
        self.pool = layers.GlobalAveragePooling1D()
        self.proj = layers.Dense(d_model)

    def call(self, x):
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        for b in self.blocks:
            x = b(x)
        x = self.pool(x)
        x = self.proj(x)
        norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(x), axis=-1, keepdims=True) + 1e-12)
        return x / norm

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers_count,
            "dropout": self.dropout_rate,
        })
        return config

@keras.saving.register_keras_serializable()
class SiameseModel(keras.Model):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs):
        s1 = inputs["input_ids_1"]
        s2 = inputs["input_ids_2"]

        e1 = self.encoder(s1)
        e2 = self.encoder(s2)

        sim = keras.ops.sum(e1 * e2, axis=-1, keepdims=True)
        return sim

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": keras.utils.serialize_keras_object(self.encoder)
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder_cfg = config.pop("encoder")
        encoder = keras.utils.deserialize_keras_object(
            encoder_cfg,
            custom_objects={
                "Encoder": Encoder,
                "TransformerBlock": TransformerBlock,
                "PositionalEmbedding": PositionalEmbedding
            }
        )
        return cls(encoder=encoder, **config)

@keras.saving.register_keras_serializable()
def contrastive_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    margin = 0.5
    pos_loss = y_true * tf.square(1 - y_pred)
    neg_loss = (1 - y_true) * tf.square(tf.maximum(y_pred - margin, 0))
    return tf.reduce_mean(pos_loss + neg_loss)

VOCAB_SIZE = 30522
MAX_LEN = 128
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "semantic_bert.keras")
LOG_PATH = os.path.join(BASE_DIR, "training_log.txt")
DATASET_PATH = os.path.join(BASE_DIR, "training_data.jsonl")

def build_siamese_model():
    encoder = Encoder(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model = SiameseModel(encoder)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=contrastive_loss
    )
    return model, encoder

print("[Sphinx AI] Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("[Sphinx AI] Tokenizer loaded.")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "Encoder": Encoder,
                "TransformerBlock": TransformerBlock,
                "PositionalEmbedding": PositionalEmbedding,
                "SiameseModel": SiameseModel,
                "contrastive_loss": contrastive_loss
            }
        )
        print(f"[Sphinx AI] ✓ Loaded trained model from {MODEL_PATH}")
    except Exception as e:
        print(f"[Sphinx AI] ⚠ Could not load model: {e}")

if model is None:
    print("[Sphinx AI] Building Siamese Transformer model from scratch...")
    model, _ = build_siamese_model()
    dummy = {
        "input_ids_1": tf.zeros((1, MAX_LEN), dtype=tf.int32),
        "input_ids_2": tf.zeros((1, MAX_LEN), dtype=tf.int32),
        "attention_mask_1": tf.zeros((1, MAX_LEN), dtype=tf.int32),
        "attention_mask_2": tf.zeros((1, MAX_LEN), dtype=tf.int32)
    }
    model(dummy)

total_params = sum(int(tf.reduce_prod(v.shape)) for v in model.trainable_variables)
print(f"[Sphinx AI] Model ready — {total_params:,} trainable parameters.")

_training_data_count = 0
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "r") as f:
        _training_data_count = sum(1 for _ in f)
    print(f"[Sphinx AI] ✓ {_training_data_count} training samples in dataset (never forgotten).")

def predict_similarity(sentence1: str, sentence2: str) -> float:
    s1 = tokenizer(
        sentence1, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="np"
    )
    s2 = tokenizer(
        sentence2, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="np"
    )
    inputs = {
        "input_ids_1": tf.constant(s1["input_ids"]),
        "attention_mask_1": tf.constant(s1["attention_mask"]),
        "input_ids_2": tf.constant(s2["input_ids"]),
        "attention_mask_2": tf.constant(s2["attention_mask"])
    }
    sim = model(inputs, training=False)
    score = float(sim[0][0])
    return max(0.0, min(1.0, score))

def _save_training_pair(sentence1: str, sentence2: str, label: float):
    entry = {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": label,
        "timestamp": datetime.now().isoformat()
    }
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def _load_all_training_data():
    if not os.path.exists(DATASET_PATH):
        return []
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def retrain_from_all_data(epochs: int = 1, batch_size: int = 8):
    all_data = _load_all_training_data()
    if not all_data:
        return 0.0

    ids1_list, att1_list, ids2_list, att2_list, labels_list = [], [], [], [], []
    for entry in all_data:
        s1 = tokenizer(
            entry["sentence1"], padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="np"
        )
        s2 = tokenizer(
            entry["sentence2"], padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="np"
        )
        ids1_list.append(s1["input_ids"][0])
        att1_list.append(s1["attention_mask"][0])
        ids2_list.append(s2["input_ids"][0])
        att2_list.append(s2["attention_mask"][0])
        labels_list.append(entry["label"])

    ids1 = np.array(ids1_list)
    att1 = np.array(att1_list)
    ids2 = np.array(ids2_list)
    att2 = np.array(att2_list)
    labels = np.array(labels_list).reshape(-1, 1).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids_1": ids1,
            "attention_mask_1": att1,
            "input_ids_2": ids2,
            "attention_mask_2": att2
        },
        labels
    )).shuffle(len(all_data)).batch(batch_size)

    total_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        for batch_inputs, batch_labels in dataset:
            with tf.GradientTape() as tape:
                y_pred = model(batch_inputs, training=True)
                loss = contrastive_loss(batch_labels, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss)
            steps += 1

    avg_loss = total_loss / max(steps, 1)

    try:
        model.save(MODEL_PATH)
    except Exception as e:
        print(f"[Sphinx AI] ⚠ Could not save model: {e}")

    return avg_loss

def train_on_pair(sentence1: str, sentence2: str, label: float):
    _save_training_pair(sentence1, sentence2, label)
    loss_val = retrain_from_all_data(epochs=1)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f'[{timestamp}] TRAINED | loss={loss_val:.6f} | label={label:.2f} | s1="{sentence1[:50]}" | s2="{sentence2[:50]}"'
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")
    return loss_val

def get_model_stats() -> dict:
    total_params = sum(int(tf.reduce_prod(v.shape)) for v in model.trainable_variables)
    train_count = 0
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            train_count = sum(1 for line in f if "TRAINED" in line)
    dataset_size = 0
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r") as f:
            dataset_size = sum(1 for _ in f)
    return {
        "model_name": "Sphinx V1",
        "architecture": "Siamese Transformer",
        "vocab_size": VOCAB_SIZE,
        "max_length": MAX_LEN,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "parameters": total_params,
        "parameters_human": f"{total_params / 1_000_000:.2f}M",
        "train_sessions": train_count,
        "dataset_size": dataset_size,
        "tokenizer": "BERT (bert-base-uncased)",
        "loss_function": "Contrastive",
        "dataset": "GLUE STS-B + User Data",
        "version": "1.0.0"
    }

def get_training_logs() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]
