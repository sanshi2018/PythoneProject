import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

# 获取当前系统的特定软件的音频输出
# 通过命令行查看当前系统的音频设备
# $ pacmd list-sinks
# 通过命令行查看当前系统的音频设备的索引
# $ pacmd list-sinks | grep index
# 通过命令行查看当前系统的音频设备的名称
# $ pacmd list-sinks | grep device.description

speaker_info = p.get_device_info_by_host_api_device_index()

# 获取扬声器输出的帧大小
frames_per_buffer = speaker_info['defaultSampleRate']

# 打开输入流，从扬声器输出捕获声音
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=int(frames_per_buffer))

print("* recording")

frames = []

# 使用 try/except 语句来捕获 KeyboardInterrupt 异常
try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

# 在捕获到 KeyboardInterrupt 异常时停止记录
except KeyboardInterrupt:
    print("* stopped recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
