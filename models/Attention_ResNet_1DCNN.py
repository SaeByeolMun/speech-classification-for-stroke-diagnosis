# SE-ResNeXt models for Keras.
# Reference for ResNext - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))
# Reference for SE-Nets - [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf))


import tensorflow as tf


class MHAttention(tf.keras.layers.Layer):
    def __init__(self, d_model:int, num_heads:int, batch_size:int=None):
#     def __init__(self, d_model:int, num_heads:int):
        super(MHAttention,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads
        self.batch_size = batch_size
        assert self.d_model % self.num_heads == 0

        self.sqrt_depth = tf.cast(self.depth, dtype=tf.float32)

        self.query = tf.keras.layers.Dense(self.d_model)
        self.key = tf.keras.layers.Dense(self.d_model)
        self.value = tf.keras.layers.Dense(self.d_model)
        self.outweight = tf.keras.layers.Dense(self.d_model)
        self.softmax = tf.keras.layers.Softmax(axis=-1)


    def split_heads(self, input):
        if self.batch_size == None:
            batch_size = tf.shape(input)[0]
        else: 
            batch_size = self.batch_size
        input = tf.reshape(input,(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(input, perm=[0,2,1,3])


    def __call__(self, input):
        if self.batch_size == None:
            batch_size = tf.shape(input)[0]
        else: 
            batch_size = self.batch_size
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        query_splitted = self.split_heads(query)
        key_splitted = self.split_heads(key)
        value_splitted = self.split_heads(value)

        q_mat_k = tf.matmul(query_splitted, key_splitted, transpose_b=True)
        q_mat_k = q_mat_k / self.sqrt_depth

        q_mat_k_soft = self.softmax(q_mat_k)
        attention_score = tf.matmul(q_mat_k_soft, value_splitted)
        attention_score = tf.transpose(attention_score, perm=[0,2,1,3])
        attention_score = tf.reshape(attention_score, (batch_size, -1, self.d_model))

        return self.outweight(attention_score)


class MyEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model:int, num_heads:int, batch_size:int=None):
        super(MyEncoder,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.multi_head_attention = MHAttention(self.d_model, self.num_heads, self.batch_size)

        self.dense1 = tf.keras.layers.Dense(d_model)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.act1 = tf.keras.layers.Activation('relu')

    def __call__(self, input):
        a = self.multi_head_attention(input)
        con = tf.concat([input,a], axis=-1)
        o1 = self.dense1(con)
        o1 = self.layer_norm(o1)
        o1 = self.act1(o1)       

        return o1

class Res1(tf.keras.layers.Layer):
# class Res1(tf.keras.Model):
    def __init__(self,filters:int,kernel_size:int,padding:str,activation:str, flag_res:bool=True):
        super(Res1,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.FLAG_RES = flag_res

        self.conv1 = tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation(self.activation)

        self.conv2 = tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation(self.activation)

        self.conv3 = tf.keras.layers.Conv1D(self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.act3 = tf.keras.layers.Activation(self.activation)

        self.pool = tf.keras.layers.MaxPool1D(strides=2)


    def __call__(self, input):
        x1 = self.conv1(input)
        x2 = self.batch1(x1)
        x3 = self.act1(x2)

        x4 = self.conv2(x3)
        x5 = self.batch2(x4)
        x6 = self.act2(x5)

        x7 = self.conv3(x6)
        x8 = self.batch3(x7)
        x9 = self.act3(x8)

        if self.FLAG_RES:
            x_added = tf.add(x9, x3)
            return self.pool(x_added)
        else :
            return self.pool(x9)


        
class MobileDense1(tf.keras.layers.Layer):
    ## the number of filters must be twice of input channels because the output will be concatenated. 
    def __init__(self,filters:int, kernel_size:int=3, padding:str='same', activation:str='relu', depth_mul:int=4):
        super(MobileDense1,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.depth_mul = depth_mul

        if self.filters & 1:
            raise Exception("from MobileDense1: the parameter named 'filters' must be even number")

        self.conv1 = tf.keras.layers.DepthwiseConv2D(self.kernel_size, depth_multiplier=self.depth_mul, padding=self.padding)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation(self.activation)
        self.conv2 = tf.keras.layers.Conv2D(self.filters//2, 1, padding=self.padding)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation(self.activation)
        self.pool = tf.keras.layers.MaxPool2D(strides=2)
        
    def __call__(self, input):
        if input.shape[-1] != self.filters//2:
            raise Exception("from MobileDense1: the input channels must be half of the 'filters' parameter")

        x1 = self.conv1(input)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        x6 = self.act2(x5)
        x7 = tf.concat([input, x6],axis=-1)
        
        return self.pool(x7)

        
        

class AttResNet:
    def __init__(self, length, num_channel, num_filters, batch_size, problem_type='Regression',
                 output_nums=1, activation_fn='elu'):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums

        ## RESNET + ATTENTION
        self.BATCH_SIZE = batch_size
        self.activation_fn = activation_fn
        self.num_filters = num_filters

    def AttResNet(self):


        input_layer = tf.keras.Input(shape=(self.length,self.num_channel))

        res1 = Res1(self.num_filters, 3, 'same', self.activation_fn)(input_layer)
        ## 32000, 32

        res2 = Res1(self.num_filters*2, 3, 'same', self.activation_fn)(res1)
        ## 16000, 64

        res3 = Res1(self.num_filters*4, 3, 'same', self.activation_fn)(res2)
        ## 8000, 128

        res4 = Res1(self.num_filters*8, 3, 'same', self.activation_fn)(res3)
        ## 4000, 256

        res5 = Res1(self.num_filters*16, 3, 'same', self.activation_fn)(res4)
        ## 2000, 512

        res6 = Res1(self.num_filters*8, 3, 'same', self.activation_fn)(res5)
        ## 1000, 256


        conv1 = tf.keras.layers.Conv1D(self.num_filters*4, kernel_size=3, padding='same')(res6)
        ## 1000, 128
        tr1 = tf.transpose(conv1,perm=[0,2,1])
        # print("input_layer, ", res10)
        enc1 = MyEncoder(1000,10,tf.shape(res1)[0])(tr1)

        flat1 = tf.keras.layers.Flatten()(enc1)
#         batch6 = BatchNormalization()(flat1)
#         dense1 = Dense(200, activation='relu')(batch6)
        dense1 = tf.keras.layers.Dense(200, activation='relu')(flat1)
        batch7 = tf.keras.layers.BatchNormalization()(dense1)
        dense2 = tf.keras.layers.Dense(20, activation='relu')(batch7)
        # dense2 = Dense(20, activation='relu')(dense1)
        if self.problem_type == 'Classification':
            output_layer = tf.keras.layers.Dense(self.output_nums, activation='softmax')(dense2)
        if self.problem_type == 'Binary':
            output_layer = tf.keras.layers.Dense(self.output_nums, activation='sigmoid')(dense2)

        model = tf.keras.Model(input_layer, output_layer)

        return model



if __name__ == '__main__':
    # Configurations
    length = 1024
    model_name = 'AttResNet'  # Modified DenseNet
    model_width = 16  # Width of the Initial Layer, subsequent layers start from here
    batch_size = 4
    num_channel = 1  # Number of Channels in the Model
    problem_type = 'Regression' # Classification or Regression
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    # Build, Compile and Print Summary

    Model = AttResNet(length, num_channel, model_width, batch_size, 
                      problem_type=problem_type, output_nums=output_nums).AttResNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
