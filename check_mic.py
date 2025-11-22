import speech_recognition as sr

mics = sr.Microphone.list_microphone_names()
print("\n[MIC] AVAILABLE MICROPHONES:")
for i, mic_name in enumerate(mics):
    print(f"Index {i}: {mic_name}")