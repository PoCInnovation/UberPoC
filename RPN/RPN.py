from keras.models import Input, Model
from keras.layers import Conv2D
from utils import parse_label, smoothL1, loss_cls


class RPN:
    def __init__(self):
        # Hyper parameters of our model
        self.batch_size = 512
        self.k_anch = 9

        feature_map_tile = Input(shape=(None, None, 1536))
        conv3x3 = Conv2D(
            filters=512,
            kernel_size=(3,3),
            padding='same',
            name="3x3"
        )(feature_map_tile)
        output_deltas = Conv2D(
            filters= 4 * self.k_anch,
            kernel_size=(1,1),
            activation='linear',
            kernel_initializer="uniform",
            name="deltas1"
        )(conv3x3)
        output_scores = Conv2D(
            filters=1 * self.k_anch,
            kernel_size=(1,1),
            activation="sigmoid",
            kernel_initializer="uniform",
            name="scores1"
        )(conv3x3)
        self.model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])
        self.model.compile(optimizer='adam', loss={'scores1': loss_cls, 'deltas1': smoothL1})

    def generate_input(self):
        pass

    def create_batch(self):
        pass

    def train(self):
        pass


category, gt_boxes, scale = parse_label("./TRAIN_DATA/stop_1.xml")
print(category)
print(gt_boxes)
print(scale)