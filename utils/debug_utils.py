import soundfile as sf
import torch
import io


def debug_save(data, sr, label):
    """
    Saves various audio formats to local files for inspection.
    """
    filename = f"debug_{label}.wav"

    # Handle Torch Tensors (Cleaned/Normalized)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().squeeze()

    # Handle BytesIO (The final buffer)
    if isinstance(data, io.BytesIO):
        with open(filename, "wb") as f:
            f.write(data.getvalue())
        print(f"Saved Buffer to {filename}")
        return

    # Handle Numpy (Waveform or Int16)
    # Note: soundfile handles both float32 and int16 automatically
    sf.write(filename, data, sr)
    print(f"Saved {label} to {filename}")
