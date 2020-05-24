
import os
import random
import argparse
import json
import torch
import torch.utils.data
from utils.audio_processor import AudioProcessor
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--wavfile_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["audio"]
    mel2samp = AudioProcessor(**data_config)
    
    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    filepath = args.wavfile_path
    melspectrogram = mel2samp.get_mel_from_audio_path(filepath)

    filename = os.path.basename(filepath)
    new_filepath = os.path.join(args.output_dir, filename + '.pt')
    torch.save(melspectrogram, new_filepath)
    print("Spectogram with shape:",melspectrogram.shape, "Saved in", new_filepath)