import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import sounddevice as sd

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/base_es_singlespeaker_22k.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("voices/odal.pth", net_g, None)

def run_tts(text, rate = 1, noise_s = 0.667, noise_scale_w = 0.8):
    text = text.replace("(", ",")
    text = text.replace(")", ",")
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=noise_s, noise_scale_w=noise_scale_w, length_scale=rate)[0][0,0].data.cpu().float().numpy()
    # play here:
    sd.play(audio, hps.data.sampling_rate)
print("Ingresa tu texto")
while True:
    try:
        print("-"*50)
        transcript = input()
        if transcript == "":
            continue
        run_tts(transcript, 1, 0.667, 0.8)
    except EOFError:
        break
    except KeyboardInterrupt:
        print("Stopping...")
        break