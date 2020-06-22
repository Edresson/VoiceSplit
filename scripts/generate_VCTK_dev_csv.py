import os
import random
import pandas

all_speakers = ['p262', 'p270', 'p236', 'p250', 'p376', 'p283', 'p278', 'p317', 'p260', 'p232', 'p339', 'p257', 'p266', 'p297', 'p240', 'p326', 'p304', 'p308', 'p315', 'p231', 'p340', 'p364', 'p323', 'p312', 'p294', 'p341', 'p241', 'p259', 'p374', 'p287', 'p254', 'p228', 'p307', 'p316', 'p248', 'p335', 'p252', 'p256', 'p276', 'p298', 'p268', 'p234', 'p237', 'p281', 'p244', 'p267', 'p363', 'p299', 'p288', 'p239', 'p243', 'p280', 'p295', 'p274', 'p318', 'p330', 'p225', 'p269', 'p284', 'p264', 'p273', 'p227', 'p313', 'p362', 'p334', 'p253', 'p261', 'p263', 'p329', 'p233', 'p246', 'p293', 'p255', 'p247', 'p343', 'p245', 'p303', 'p271', 'p311', 'p251', 'p277', 'p333', 'p302', 'p230', 'p347', 'p306', 'p301', 'p238', 'p310', 'p305', 'p292', 'p300', 'p282', 'p286', 'p249', 'p285', 'p336', 'p272', 'p226', 'p229', 'p345', 'p361', 'p279', 'p314', 'p360', 'p351', 'p265', 'p275', 'p258']
output_dir = "../datasets/VCTK/"

vctk_wavs_dir = 'wav48/'
sample_list = []

wav_samples = ['_001.wav', '_002.wav', '_003.wav', '_004.wav', '_005.wav', '_006.wav', '_007.wav', '_008.wav', '_009.wav', '_010.wav', '_011.wav', '_012.wav', '_013.wav', '_014.wav', '_016.wav', '_017.wav', '_018.wav', '_019.wav', '_020.wav', '_021.wav', '_022.wav', '_023.wav', '_024.wav', '_025.wav', '_026.wav', '_027.wav', '_028.wav', '_029.wav', '_030.wav', '_033.wav', '_035.wav', '_036.wav', '_037.wav', '_038.wav', '_039.wav', '_040.wav', '_044.wav', '_045.wav', '_046.wav', '_049.wav', '_051.wav', '_052.wav', '_053.wav', '_054.wav', '_056.wav', '_057.wav', '_058.wav', '_059.wav', '_060.wav', '_061.wav', '_062.wav', '_063.wav', '_064.wav', '_065.wav', '_066.wav', '_067.wav', '_070.wav', '_071.wav', '_072.wav', '_073.wav', '_081.wav', '_082.wav', '_083.wav', '_084.wav', '_086.wav', '_089.wav', '_090.wav', '_092.wav', '_094.wav', '_103.wav', '_104.wav', '_108.wav', '_109.wav', '_110.wav', '_111.wav', '_113.wav', '_114.wav', '_115.wav', '_116.wav', '_117.wav', '_118.wav', '_120.wav', '_121.wav', '_122.wav', '_123.wav', '_124.wav', '_126.wav', '_127.wav', '_128.wav', '_131.wav', '_133.wav', '_135.wav', '_136.wav', '_141.wav', '_142.wav', '_143.wav', '_144.wav', '_145.wav', '_147.wav', '_149.wav', '_150.wav', '_151.wav', '_152.wav', '_153.wav', '_156.wav', '_157.wav', '_158.wav', '_159.wav', '_165.wav', '_166.wav', '_169.wav', '_171.wav', '_172.wav', '_173.wav', '_174.wav', '_175.wav', '_176.wav', '_177.wav', '_179.wav', '_182.wav', '_191.wav', '_192.wav', '_193.wav', '_195.wav', '_196.wav', '_197.wav', '_199.wav', '_200.wav', '_201.wav', '_202.wav', '_203.wav', '_208.wav', '_210.wav', '_211.wav', '_212.wav', '_218.wav', '_219.wav', '_220.wav', '_221.wav', '_222.wav', '_223.wav', '_224.wav', '_225.wav', '_235.wav', '_236.wav', '_237.wav', '_238.wav', '_239.wav', '_240.wav', '_241.wav', '_242.wav', '_243.wav', '_244.wav', '_248.wav', '_253.wav', '_254.wav', '_257.wav', '_258.wav', '_264.wav', '_265.wav', '_266.wav', '_268.wav', '_273.wav', '_274.wav', '_275.wav', '_276.wav', '_277.wav', '_279.wav', '_280.wav', '_281.wav', '_282.wav', '_285.wav', '_286.wav', '_287.wav', '_289.wav', '_290.wav', '_291.wav', '_293.wav', '_294.wav', '_295.wav', '_296.wav', '_297.wav', '_298.wav', '_299.wav', '_300.wav', '_301.wav', '_302.wav', '_303.wav', '_305.wav', '_308.wav', '_309.wav', '_310.wav', '_312.wav', '_314.wav', '_315.wav', '_316.wav', '_317.wav', '_318.wav', '_319.wav', '_320.wav', '_322.wav', '_323.wav', '_324.wav', '_325.wav', '_326.wav', '_328.wav', '_329.wav', '_330.wav', '_331.wav', '_332.wav', '_334.wav', '_335.wav', '_336.wav', '_337.wav', '_346.wav', '_347.wav', '_348.wav', '_349.wav', '_350.wav', '_351.wav', '_352.wav', '_353.wav', '_354.wav', '_355.wav', '_356.wav', '_357.wav', '_358.wav', '_359.wav', '_363.wav', '_365.wav', '_366.wav']
# p225_001.wav
#clean_utterance,embedding_utterance,interference_utterance
for i in range(len(all_speakers)):
    speaker_clean = all_speakers[i]
    for j in range(i+1,len(all_speakers[i:])):
            interference_speaker = all_speakers[j]
            clean_wav = random.choice(wav_samples)
            aux_wav_samples = wav_samples[:]
            aux_wav_samples.remove(clean_wav)
            emb_wav = random.choice(aux_wav_samples) # select one emb reference diferente then clean_wav
            aux_wav_samples.remove(emb_wav) # its necessary because texts in vctk is parallel, and its not reflect real cenaries
            interference_wav = random.choice(aux_wav_samples)

            clean_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+clean_wav)
            emb_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+emb_wav)
            interference_ref = os.path.join(vctk_wavs_dir, interference_speaker, interference_speaker+interference_wav) 
            
            sample_list.append([clean_ref, emb_ref, interference_ref])



df = pandas.DataFrame(data=sample_list, columns=['clean_utterance','embedding_utterance','interference_utterance'])
df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)




    
