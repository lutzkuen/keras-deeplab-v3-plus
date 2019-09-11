from model import Deeplabv3

model = Deeplabv3(input_shape=(256, 256, 3), classes=4)

print(model.summary())
