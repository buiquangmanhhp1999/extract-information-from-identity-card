import torch
import numpy as np
from PIL import Image
from torch.nn.functional import log_softmax
import cv2
from model.vocab import Vocab
from model.beam import Beam
import parameters as cfg
from OCR import OCR


def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        memories = model.transformer.forward_encoder(src)
        for i in range(memories.size(1)):
            memory = memories[:, i, :].repeat(1, beam_size, 1)  # TxNxE
            sent = beamsearch(memory, model, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents


def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)  # TxNxE
        sent = beamsearch(memory, model, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent


def beamsearch(memory, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # memory: Tx1xE
    model.eval()
    device = memory.device

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token,
                end_token_id=eos_token)

    with torch.no_grad():
        memory = memory.repeat(1, beam_size, 1)  # TxNxE

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCXHxW"""
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']

    model = OCR(len(vocab),
                config['seq2seq']['encoder_hidden'],
                config['seq2seq']['decoder_hidden'],
                config['seq2seq']['img_channel'],
                config['seq2seq']['decoder_embedded'],
                config['seq2seq']['dropout'])
    model = model.to(device)

    return model, vocab


def process_image(image):
    # convert to numpy array
    img = np.asarray(image)
    h, w = img.shape[:2]

    # padding image:
    padding_right = cfg.width_img - w
    img = cv2.resize(img, (w, cfg.height_img), cv2.INTER_AREA)

    if padding_right >= 0:
        img = cv2.resize(img, (w, cfg.height_img), cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, top=0, left=0, right=padding_right, bottom=0, borderType=cv2.BORDER_CONSTANT, value=0)
    elif padding_right < 0:
        img = cv2.resize(img, (cfg.width_img, cfg.height_img), cv2.INTER_AREA)

    img = img.transpose(2, 0, 1)
    img = img / 255.0

    return img


def process_input(image):
    img = process_image(image)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)

    return s
