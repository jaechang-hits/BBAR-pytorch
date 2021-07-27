from torch import FloatTensor

def binary_cross_entropy(predict, answer) :
    if answer == 1.0 :
        return (predict + 1e-12).log().neg().mean()
    elif answer == 0.0 :
        return (1 - predict + 1e-12).log().neg().mean()
    else :
        return (((predict + 1e-12).log() * answer).mean() + ((1. - predict + 1e-12).log() * (1. - answer)).mean()).neg()

fragment_predict_loss = binary_cross_entropy
