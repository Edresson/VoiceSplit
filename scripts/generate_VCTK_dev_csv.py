import os
import random
import pandas
import librosa
random.seed(0)
all_speakers = ['p262', 'p270', 'p236', 'p250', 'p376', 'p283', 'p278', 'p317', 'p260', 'p232', 'p339', 'p257', 'p266', 'p297', 'p240', 'p326', 'p304', 'p308', 'p315', 'p231', 'p340', 'p364', 'p323', 'p312', 'p294', 'p341', 'p241', 'p259', 'p374', 'p287', 'p254', 'p228', 'p307', 'p316', 'p248', 'p335', 'p252', 'p256', 'p276', 'p298', 'p268', 'p234', 'p237', 'p281', 'p244', 'p267', 'p363', 'p299', 'p288', 'p239', 'p243', 'p280', 'p295', 'p274', 'p318', 'p330', 'p225', 'p269', 'p284', 'p264', 'p273', 'p227', 'p313', 'p362', 'p334', 'p253', 'p261', 'p263', 'p329', 'p233', 'p246', 'p293', 'p255', 'p247', 'p343', 'p245', 'p303', 'p271', 'p311', 'p251', 'p277', 'p333', 'p302', 'p230', 'p347', 'p306', 'p301', 'p238', 'p310', 'p305', 'p292', 'p300', 'p282', 'p286', 'p249', 'p285', 'p336', 'p272', 'p226', 'p229', 'p345', 'p361', 'p279', 'p314', 'p360', 'p351', 'p265', 'p275', 'p258']
output_dir = "../datasets/VCTK/"

vctk_dir = '../../../datasets/VCTK-Corpus-removed-silence/'
vctk_wavs_dir = 'wav48/'
sample_list = []

sample_rate = 16000
audio_len= int(sample_rate * 3) #time for 3 seconds

for i in range(len(all_speakers)):
    speaker_clean = all_speakers[i].replace(' ', '')
    for j in range(i+1,len(all_speakers[i:])):
            interference_speaker = all_speakers[j].replace(' ', '')
            wav_samples = os.listdir(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean))
            
            clean_wav = random.choice(wav_samples)
            while librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,clean_wav), sr=sample_rate)[0].shape[0] < audio_len:
                clean_wav = random.choice(wav_samples)

            clean_wav = clean_wav.replace(speaker_clean, '')
            emb_wav = random.choice(wav_samples) # select one emb reference diferente then clean_wav
            while clean_wav == emb_wav and librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,emb_wav), sr=sample_rate)[0].shape[0] < audio_len: # its necessary for emb and clean not is same sample
                emb_wav = random.choice(wav_samples)
            emb_wav = emb_wav.replace(speaker_clean, '')
            # get samples interference samples
            wav_samples = os.listdir(os.path.join(vctk_dir,vctk_wavs_dir,interference_speaker))
            interference_wav = random.choice(wav_samples)
            while clean_wav == interference_wav and librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,interference_wav), sr=sample_rate)[0].shape[0] < audio_len: # its necessary for clean interference not is same text, its necessary because the texts in vctk is parallel
                interference_wav = random.choice(wav_samples)
            interference_wav = interference_wav.replace(interference_speaker, '')
            print(clean_wav,emb_wav,interference_wav)
            clean_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+clean_wav)
            emb_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+emb_wav)
            interference_ref = os.path.join(vctk_wavs_dir, interference_speaker, interference_speaker+interference_wav) 
            
            sample_list.append([clean_ref, emb_ref, interference_ref])


df = pandas.DataFrame(data=sample_list, columns=['clean_utterance','embedding_utterance','interference_utterance'])
df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)




    
