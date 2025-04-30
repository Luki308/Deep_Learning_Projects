from pathlib import Path

import torchaudio
import os

from Project_2.src.utils.DataHelperForTransformers import read_list, assign_label


def split_and_save_audio(file_path, output_path, clip_duration_sec=1.0, frac_moved=0.0):
    wav, sr = torchaudio.load(file_path)
    total_samples = wav.shape[1]
    clip_samples=int(sr * clip_duration_sec)
    num_clips=int((total_samples-frac_moved*clip_samples) / clip_samples)
    os.makedirs(output_path, exist_ok=True)

    for i in range(num_clips):
        start =int(frac_moved*clip_samples) + i * clip_samples
        end = start + clip_samples
        clip = wav[:, start:end]

        clip_filename = os.path.join(output_path, f'clip{frac_moved}_{i}_{file_path.name.lower()}.wav')

        torchaudio.save(clip_filename, clip, sr)

def main():
    data_dir = '../../data/train'
    data_dir = Path(data_dir).resolve()

    dataset_path = data_dir / 'audio'
    background_path = dataset_path / '_background_noise_'
    for frac in range(0, 10, 2):
        for audio_path in background_path.rglob("*.wav"):
            split_and_save_audio(audio_path,f"{dataset_path}/silence", frac_moved=frac/10)


if __name__ == '__main__':
    main()