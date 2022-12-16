def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def froze_layer(layer):
    for para in layer.parameters():
        para.requires_grad = False

def activate_layer(layer):
    for para in layer.parameters():
        para.requires_grad = True