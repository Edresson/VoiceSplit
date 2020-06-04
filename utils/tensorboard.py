"""
This model was adapted from: https://github.com/mindslab-ai/voicefilter
So part of the code in this file was made by @seungwonpark
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from tensorboardX import SummaryWriter

def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir, config):
        super(TensorboardWriter, self).__init__(logdir)
        self.audio_config = config

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, target_wav, est_wav,
                       mixed_spec, target_spec, est_spec, est_mask,
                       step):
        
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.audio_config['sample_rate'])
        self.add_audio('target_wav', target_wav, step, self.audio_config['sample_rate'])
        self.add_audio('estimated_wav', est_wav, step, self.audio_config['sample_rate'])

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('result/estimated_spectrogram',
            plot_spectrogram_to_numpy(est_spec), step, dataformats='HWC')
        self.add_image('result/estimated_mask',
            plot_spectrogram_to_numpy(est_mask), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_spec - target_spec)), step, dataformats='HWC')
