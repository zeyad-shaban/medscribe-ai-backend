import torch
from df.enhance import init_df, enhance
import librosa

def denoise_audio(waveform, sr):
    model, df_state, _ = init_df()

    if sr != df_state.sr():
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=df_state.sr())

    noisy_audio = torch.from_numpy(waveform).float().unsqueeze(0)
    enhanced_audio = enhance(model, df_state, noisy_audio)

    return enhanced_audio


def normalize_audio(audio_tensor, target_rms_db=-18.0, max_peak_db=-1.5):
    # 1. Normalize Average RMS energy
    rms = torch.sqrt(torch.mean(audio_tensor**2))
    target_rms = 10 ** (target_rms_db / 20.0)
    audio_tensor = audio_tensor * (target_rms / (rms + 1e-8))

    # 2. Prevent dynamic clipping via Peak normalization fallback
    max_peak = torch.max(torch.abs(audio_tensor))
    target_peak = 10 ** (max_peak_db / 20.0)
    if max_peak > target_peak:
        audio_tensor = audio_tensor * (target_peak / max_peak)

    return torch.clamp(audio_tensor, -1.0, 1.0)