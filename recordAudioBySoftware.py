# import pyaudio
# import wave
#
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# RECORD_SECONDS = 15
# WAVE_OUTPUT_FILENAME = "output.wav"
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
# # 打开音频设备并开始记录
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 input_device_index=int(device_index),
#                 frames_per_buffer=CHUNK)
#
# print("* recording")
#
# frames = []
#
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# print("* done recording")
#
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# # 将记录的声音保存到 WAV 文件中
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()




import pyaudio
import wave

import websocket as websocket

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"


# 获取所有可用的音频设备信息
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# 输入想要使用的音频设备编号
device_index = input("Choose input device: ")

# 打开音频设备并开始记录
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=int(device_index),
                frames_per_buffer=CHUNK)

print("* recording")
# 创建一个websocket服务器，请求地址为ws://localhost:8000/ws

# 打开 WAV 文件并设置相应的参数
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)

# 不断记录音频
while True:
    data = stream.read(CHUNK)
    wf.writeframes(data)

# 关闭音频设备和 WAV 文件
stream.stop_stream()
stream.close()
p.terminate()
wf.close()
