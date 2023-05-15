import whisper
from pydub import AudioSegment

print("Importation du modèle Whsiper pour le Voix->Texte")
model = whisper.load_model("base")

def audioToText(audioFile):
    if audioFile[-3:]=="ogx":
        wav_audio = AudioSegment.from_file(audioFile, format="ogg")
        audioFileMp3=audioFile[:-4]+".mp3"
        wav_audio.export(audioFileMp3, format="mp3")
        audioFile=audioFileMp3

    audio = whisper.load_audio(audioFile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text

