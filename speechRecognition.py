# from huggingsound import SpeechRecognitionModel
# #
# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
# audio_paths = ["./assess/ChineseConversationalSpeech.wav"]
# #
# transcriptions = model.transcribe(audio_paths)
#

# 从训练集中随机抽取10个样本，然后使用Wav2Vec2模型进行预测。这里使用的是中文语言模型，因此我们需要将所有的文本转换为大写。
# import torch
# import librosa
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#
# LANG_ID = "zh-CN"
# MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
# SAMPLES = 10
#
# test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")
#
# processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
#
# # Preprocessing the datasets.
# # We need to read the audio files as arrays
# def speech_file_to_array_fn(batch):
#     speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
#     batch["speech"] = speech_array
#     batch["sentence"] = batch["sentence"].upper()
#     return batch
#
# test_dataset = test_dataset.map(speech_file_to_array_fn)
# inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
#
# with torch.no_grad():
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#
# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)
#
# for i, predicted_sentence in enumerate(predicted_sentences):
#     print("-" * 100)
#     print("Reference:", test_dataset[i]["sentence"])
#     print("Prediction:", predicted_sentence)


# 指定音频识别
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "zh-CN"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

audio_file = "assess/ChineseConversationalSpeech.wav"
speech_array, sampling_rate = librosa.load(audio_file, sr=16_000)

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = processor.batch_decode(predicted_ids)

print(predicted_sentence[0])
#

# import pyaudio
# import torch
# import librosa
# from np.magic import np
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#
# LANG_ID = "zh-CN"
# MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
#
# # Set up audio recording
# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 16000
# CHUNK_SIZE = 1024
#
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
#
# # 获取所有可用的音频设备信息
# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get('deviceCount')
# for i in range(0, numdevices):
#         if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#             print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
#
# # 输入想要使用的音频设备编号
# device_index = input("Choose input device: ")
#
#
# # Create audio stream
# # 打开音频设备并开始记录
# p = pyaudio.PyAudio()
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 input_device_index=int(device_index),
#                 frames_per_buffer=CHUNK)
#
# # Load Wav2Vec2 model
# processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
# model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
#
# # Loop for real-time speech recognition
# while True:
#     # Read audio data from stream
#     audio_data = stream.read(CHUNK_SIZE)
#
#     # Convert audio data to array
#     speech_array = librosa.resample(librosa.util.buf_to_float(audio_data, n_bytes=2, dtype=np.int16), RATE, 16000)
#
#     # Preprocess audio data
#     inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
#
#     # Perform inference
#     with torch.no_grad():
#         logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#
#     # Convert logits to predicted text
#     predicted_ids = torch.argmax(logits, dim=-1)
#     predicted_sentence = processor.batch_decode(predicted_ids)[0]
#
#     # Print predicted text
#     print("Predicted Text:", predicted_sentence)
#
