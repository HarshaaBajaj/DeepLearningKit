def conv_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), bn_init_scale=1):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False, init_scale=bn_init_scale, disable_regularization=True)(c)
    return r
	
Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)

(previousBuffer, currentCommandBuffer, previousShape) = 
createConvolutionLayerCached(layer, inputBuffer: previousBuffer, inputShape: previousShape, metalCommandQueue: metalCommandQueue, metalDefaultLibrary:metalDefaultLibrary, metalDevice:metalDevice, layer_data_caches: &layer_data_caches, blob_cache: &blob_cache, layer_number: layer_number, layer_string: layer_string)
                    	
createConvolutionLayerCached(layer: NSDictionary,
    inputBuffer: MTLBuffer,
    inputShape: [Float],
    metalCommandQueue: MTLCommandQueue, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice,
    inout layer_data_caches: [Dictionary<String,MTLBuffer>],
    inout blob_cache: [Dictionary<String,([Float],[Float])>],
    layer_number: Int,
    layer_string: String) -> (MTLBuffer, MTLCommandBuffer, [Float]) 
	
mu = 1/N*np.sum(h,axis =0) # Size (H,) 
sigma2 = 1/N*np.sum((h-mu)**2,axis=0)# Size (H,) 
hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
y = gamma*hath+beta 
