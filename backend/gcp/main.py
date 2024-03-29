import tensorflow as tf
import pickle
from google.cloud import storage

image_features_extract_model = None

BUCKET_NAME = "medical-report-generation" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

embedding_dim = 256
units = 512
max_length = 50

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

class BahdanauAttention(tf.keras.Model):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


def evaluate(image, decoder, encoder, word_to_index, index_to_word, image_features_extract_model):

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[1]))
                                                #  img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)
    # print('Hi')
    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []
    

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)


        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result

def predict(request):

  if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)
  
  # Set CORS headers for the main request
  headers = {
        'Access-Control-Allow-Origin': '*'
    }

  global image_features_extract_model
  if image_features_extract_model is None:
        download_blob(
            BUCKET_NAME,
            "models/chexnet.3.0_weights.h5",
            "/tmp/chexnet.3.0_weights.h5",
        )
        image_model = tf.keras.applications.DenseNet121(include_top=False,
                                                weights=None, pooling="avg")
        predictions = tf.keras.layers.Dense(14, activation='sigmoid', name='predictions')(image_model.output)

        image_model = tf.keras.Model(inputs=image_model.input, outputs=predictions)
        image_model.load_weights("/tmp/chexnet.3.0_weights.h5")

        # (image_model.summary())

        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
  prefix = 'checkpoint/'
  dl_dir = '/tmp/checkpoint/'
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(BUCKET_NAME)
  blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
  for blob in blobs:
      filename = blob.name.replace('/', '_') 
      blob.download_to_filename(dl_dir + filename)  # Download
      
  download_blob(
          BUCKET_NAME,
          "models/tv_layer.pkl",
          "/tmp/tv_layer.pkl",
      )
  from_disk = pickle.load(open("/tmp/tv_layer.pkl", "rb"))
  new_v = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
  # You have to call `adapt` with some dummy data (BUG in Keras)
  new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
  new_v.set_weights(from_disk['weights'])

  # print(new_v.get_vocabulary())

  # Create mappings for words to indices and indicies to words.
  word_to_index = tf.keras.layers.StringLookup(
      mask_token="",
      vocabulary=new_v.get_vocabulary())
  index_to_word = tf.keras.layers.StringLookup(
      mask_token="",
      vocabulary=new_v.get_vocabulary(),
      invert=True)
  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, new_v.vocabulary_size())
  optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

  checkpoint_path = "/tmp/checkpoint/"
  ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)

  image = request.files["file"]
  prediction = ' '.join(evaluate(image, decoder, encoder, word_to_index, index_to_word, image_features_extract_model))
  
  return {
      'prediction': prediction,
  }