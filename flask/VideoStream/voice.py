import whisper

model = whisper.load_model("base")

def audioToText(audioFile):
    audio = whisper.load_audio(audioFile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text

