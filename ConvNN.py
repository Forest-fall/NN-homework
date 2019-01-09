import numpy as np

class ReluActivator(object):
    '''输入feature map激活,是np矩阵'''
    @staticmethod
    def forward(w):
        w = np.where(w > 0, w, 0)
        return w

    @staticmethod
    def backward(output):
        output = np.where(output > 0, 1, 0)
        return output


# 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    '''从输入数组中获取本次卷积的区域,自动适配输入为2D和3D的情况 '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[start_i : start_i + filter_height, start_j : start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:, start_i : start_i + filter_height, start_j : start_j + filter_width]
       

# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j


# 计算卷积
def conv(input_array, kernel_array, output_array, stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    # channel_number = input_array.ndim
    output_width = int(output_array.shape[1])
    output_height = int(output_array.shape[0])
    kernel_width = int(kernel_array.shape[-1])
    kernel_height = int(kernel_array.shape[-2])
    for i in range(output_height):
        for j in range(output_width):
            conv_patch = get_patch(input_array, i, j, kernel_width, kernel_height, stride)
            output_array[i][j] = ( conv_patch * kernel_array).sum() + bias


# 为输入增加Zero padding
def padding(input_array, zp):
    '''添加Zero padding，自动适配输入为2D和3D的情况'''
    zp = int(zp)
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = int(input_array.shape[2])
            input_height = int(input_array.shape[1])
            input_depth = int(input_array.shape[0])
            padded_array = np.zeros((input_depth, input_height + 2 * zp, input_width + 2 * zp))
            padded_array[:, zp : zp + input_height, zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = int(input_array.shape[1])
            input_height = int(input_array.shape[0])
            padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
            padded_array[zp : zp + input_height, zp : zp + input_width] = input_array
            return padded_array

class Filter(object):
    def __init__(self, depth, height, width):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        '''计算w, b的梯度'''
        self.nabla_weights = np.zeros(self.weights.shape)
        self.nabla_bias = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s\n\n' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_nabla_weights(self):
        return self.nabla_weights

    def get_nabla_bias(self):
        return self.nabla_bias


    def update(self, learning_rate, sigma_nabla_w, sigma_nabla_b, mini_batch_size):
        '''mini batch'''
        self.weights -= (learning_rate / mini_batch_size) * sigma_nabla_w
        self.bias -= (learning_rate / mini_batch_size) * sigma_nabla_b
        # self.weights -= learning_rate * self.nabla_weights
        # self.bias -= learning_rate * self.nabla_bias

    
class ConvLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, 
                filter_number, zero_padding, stride, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(self.channel_number, filter_height, filter_width))
        self.learning_rate = learning_rate

    def forward(self, input_array):
        '''计算卷积层的输出, 输出结果保存在self.output_array中'''
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f], self.stride, filter.get_bias())
        # element_wise_op(self.output_array, self.activator.forward)
        self.output_array = ReluActivator.forward(self.output_array)
            
    def backward(self, sensitivity_map):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array中
        卷积核的梯度保存在Filter对象的nabla_weights中
        '''
        # self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_map)
        self.bp_gradient(sensitivity_map)
        return self.delta_array

    def update(self, sigma_w, sigma_b, mini_batch_size):
        '''按照梯度下降，更新权重'''
        for i in range(self.filter_number):
            self.filters[i].update(self.learning_rate, sigma_w[i], sigma_b[i], mini_batch_size)
        return self.filters

    def bp_sensitivity_map(self, sensitivity_map):
        '''
        sensitivity_map:本层的传入误差
        self.delta_array：本层的传出误差
        计算传递到上一层的误差sensitivity map, 也就是本层的self.delta_array
        '''

        '''要对sensitivity_map的大小作处理'''
        #第一次补0, 计算第一次扩充成stride=1后仍需补0的zero padding
        expanded_array_first = self.expand_SensitivityMap_forS1(sensitivity_map) # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_width = expanded_array_first.shape[2]
        #第二次补0, 第二次扩充成能与卷积核进行反向卷积的新array
        zp = int((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        expanded_array_second = padding(expanded_array_first, zp)

        '''初始化delta_array; 上一层的误差, 也是本层误差与卷积核卷积的结果'''
        self.delta_array = self.create_delta_array() 
 
        '''对于有多个filter的卷积层, 上一层的误差是所有filter, 和与其对应的本层的误差的卷积 之和'''
        '''正向传递时一个filter对应一个feature map, 反向传播时一个filter对应一个sensitivity map'''
        for f_n in range(self.filter_number):
            filter = self.filters[f_n]
            # 计算每个filter对应的delta_array
            delta_array = self.create_delta_array()  
            for channel in range(delta_array.shape[0]):
                '''用filter的每一层与sensitivity map的每一层作卷积，得到相应层的delta_array'''
                weight = filter.get_weights()[channel]
                ''''将卷积核翻转180度'''
                flipped_weight = np.rot90(weight, 2)
                conv(expanded_array_second[f_n], flipped_weight, delta_array[channel], 1, 0)
            self.delta_array += delta_array
       
        derivative_array = np.array(self.input_array)  
        derivative_array = ReluActivator.backward(derivative_array)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_map): 
        expanded_array = self.expand_SensitivityMap_forS1(sensitivity_map)
        for f in range(self.filter_number):
            filter = self.filters[f]
            '''当输入的channel为1时，就直接传递sensitivity map. 若此时按层数索引，则会传递边长，就会报错'''
            filter_channel = filter.weights.shape[0]
            if filter_channel == 1:
                conv(self.padded_input_array, expanded_array[f], filter.nabla_weights, 1, 0)
            else:
                for d in range(filter_channel):
                    conv(self.padded_input_array[d], expanded_array[f], filter.nabla_weights[d], 1, 0)

            '''计算偏置项的梯度, 即每个sensitivity_map的和'''
            filter.nabla_bias = expanded_array[f].sum()  

    def expand_SensitivityMap_forS1(self, sensitivity_map):
        '''确定第一次扩展后sensitivity map的大小, 计算stride为1时sensitivity map的大小'''
        depth = sensitivity_map.shape[0]
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = sensitivity_map[:,i,j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        output_size = (input_size - filter_size + 2 * zero_padding) / stride + 1
        return int(output_size)


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        self.input_array = input_array
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    pool_patch = get_patch(input_array[d], i, j, self.filter_width, self.filter_height, self.stride)
                    self.output_array[d,i,j] = pool_patch.max()

    def backward(self, sensitivity_map):
        self.delta_array = self.create_delta_array()
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    pool_patch = get_patch(self.input_array[d], i, j, self.filter_width, self.filter_height, self.stride)
                    k, l = get_max_index(pool_patch)
                    self.delta_array[d, i * self.stride + k, j * self.stride + l] = sensitivity_map[d, i, j]
        return self.delta_array
    
    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))