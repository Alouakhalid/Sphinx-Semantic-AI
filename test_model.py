import tensorflow as tf
import keras
from tensorflow.keras import layers
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = layers.Embedding(max_len, d_model)
    def call(self, x):
        positions = tf.range(tf.shape(x)[1])
        return self.pos_emb(positions)
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
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
        self.drop = layers.Dropout(self.dropout)
    def call(self, x):
        a = self.att(x, x)
        x = self.ln1(x + self.drop(a))
        f = self.ffn(x)
        x = self.ln2(x + self.drop(f))
        return x
    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)
class Encoder(keras.Model):
    def __init__(self, vocab_size, max_len, d_model, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
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
        return tf.nn.l2_normalize(x, axis=-1)
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
def cosine_sim(a, b):
    return tf.reduce_sum(a * b, axis=-1)
class SiameseModel(keras.Model):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
    def call(self, inputs):
        s1 = inputs["input_ids_1"]
        s2 = inputs["input_ids_2"]
        e1 = self.encoder(s1)
        e2 = self.encoder(s2)
        sim = tf.reduce_sum(e1 * e2, axis=-1, keepdims=True)
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
def contrastive_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    margin = 0.5
    pos_loss = y_true * tf.square(1 - y_pred)
    neg_loss = (1 - y_true) * tf.square(tf.maximum(y_pred - margin, 0))
    return tf.reduce_mean(pos_loss + neg_loss)
model = keras.models.load_model(
    "/content/semantic_bert.keras",
    custom_objects={
        "Encoder": Encoder,
        "TransformerBlock": TransformerBlock,
        "PositionalEmbedding": PositionalEmbedding,
        "SiameseModel": SiameseModel,
        "contrastive_loss": contrastive_loss
    }
)
def predict(sentence1, sentence2):
    s1 = tokenizer(
        sentence1,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    s2 = tokenizer(
        sentence2,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )
    inputs = {
        "input_ids_1": tf.constant(s1["input_ids"]),
        "input_ids_2": tf.constant(s2["input_ids"])
    }
    sim = model(inputs)
    return float(sim[0][0])
print(predict(
    "A man is playing guitar",
    "A person is playing guitar"
))
print(predict(
    "The sky is blue",
    "A dog is running in the park"
))
print(predict(
    "He bought a phone yesterday",
    "Yesterday he purchased a smartphone"
))
