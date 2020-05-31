
import os
import random
import argparse
import json
import torch
import torch.utils.data
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.generic_utils import load_config
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--wavfile_path", required=True)
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str, default='../',
                        help='Output directory')
    args = parser.parse_args()

    config = load_config(args.config)
    ap = AudioProcessor(config.audio)
    
    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    filepath = args.wavfile_path

    # extract spectrogram
    spectrogram = ap.get_spec_from_audio_path(filepath)
    
    
    # save spectrogram
    filename = os.path.basename(filepath)
    new_filepath = os.path.join(args.output_dir, filename + '.pt')
    torch.save(spectrogram, new_filepath)
    
    # reverse spectrogram for wave file using Griffin-Lim
    wav = ap.inv_spectrogram(spectrogram)
    ap.save_wav(wav, '../test-spec-exctraction.wav')

    print("Spectogram with shape:",spectrogram.shape, "Saved in", new_filepath)