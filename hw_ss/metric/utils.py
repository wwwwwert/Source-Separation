from editdistance import eval 

def calc_wer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    target_words = target_text.split(' ')
    pred_words = predicted_text.split(' ')
    return eval(target_words, pred_words) / len(target_words)


def calc_cer(target_text: str, predicted_text: str) -> float:
    if not target_text:
        if predicted_text:
            return 1
        return 0
    return eval(target_text, predicted_text) / len(target_text)


def calc_si_sdr(target_audio, pred_audio):
    ...

def calc_pesq():
    ...