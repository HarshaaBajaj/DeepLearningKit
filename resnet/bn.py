# TODO: map_rank is broken. We should specify the #slowest-changing axes. E.g. 1 would work for images and vectors. Requires C++ change.
[docs]def BatchNormalization(map_rank=default_override_or(None),  # if given then normalize only over this many dimensions. E.g. pass 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=default_override_or(5000), blend_time_constant=0,
                       epsilon=default_override_or(0.00001), use_cntk_engine=default_override_or(False),
                       disable_regularization=default_override_or(False),
                       name=''):
    '''
    BatchNormalization(map_rank=None, init_scale=1, normalization_time_constant=5000, blend_time_constant=0, epsilon=0.00001, use_cntk_engine=False, disable_regularization=False, name='')

    Layer factory function to create a batch-normalization layer.

    Batch normalization applies this formula to every input element (element-wise):
    ``y = (x - batch_mean) / (batch_stddev + epsilon) * scale + bias``
    where ``batch_mean`` and ``batch_stddev`` are estimated on the minibatch and ``scale`` and ``bias`` are learned parameters.

    During operation, this layer also estimates an aggregate running mean and standard deviation for use in inference.

    A ``BatchNormalization`` layer instance owns its learnable parameter tensors and exposes them as attributes ``.scale`` and ``.bias``.
    The aggregate estimates are exposed as attributes ``aggregate_mean``, ``aggregate_variance``, and ``aggregate_count``.

    Example:
     >>> # BatchNorm on an image with spatial pooling
     >>> f = BatchNormalization(map_rank=1)
     >>> f.update_signature((3,480,640))
     >>> f.bias.shape, f.scale.shape  # due to spatial pooling (map_rank=1), there are only 3 biases and scales, shared across all pixel positions
         ((3,), (3,))

    Args:
     map_rank (1 or ``None``): passing 1 means spatially-pooled batch-normalization, where normalization values will be tied across all pixel positions; while ``None``
      will normalize all elements of the input tensor independently
     init_scale (float, default 1): initial value for the ``scale`` parameter
     normalization_time_constant (int, default 5000): time constant for smoothing the batch statistics in order to compute aggregate estimates for inference.
     epsilon (float, default 0.00001): epsilon added to the variance to avoid division by 0
     use_cntk_engine (bool, default ``False``): if ``True`` then use CNTK's own engine instead of NVidia's.
     disable_regularization (bool, default ``False``): if ``True`` then disable regularization in BatchNormalization
     name (str, optional): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the operation to it

    Todo:
       Add paper reference.
    '''

    map_rank                    = get_default_override(BatchNormalization, map_rank=map_rank)
    normalization_time_constant = get_default_override(BatchNormalization, normalization_time_constant=normalization_time_constant)
    epsilon                     = get_default_override(BatchNormalization, epsilon=epsilon)
    use_cntk_engine             = get_default_override(BatchNormalization, use_cntk_engine=use_cntk_engine)
    disable_regularization      = get_default_override(BatchNormalization, disable_regularization=disable_regularization)

    # for fp16 batch_normalization, we need to use fp32 statistics
    dtype = get_default_override(None, dtype=default_override_or(np.float32))
    stat_dtype = np.float32 if dtype == np.float16 or dtype == 'float16' else dtype
    
    # parameters bound to this Function
    norm_shape  = _INFERRED
    if map_rank is not None and map_rank != 1:
        UntestedBranchError("BatchNormalization map_rank can only be 1 or None for now")
    scale        = Parameter(norm_shape, init=init_scale, dtype=stat_dtype, name='scale')
    bias         = Parameter(norm_shape, init=0,          dtype=stat_dtype, name='bias')
    run_mean     = Constant(0, shape=norm_shape, dtype=stat_dtype, name='aggregate_mean')  # note: these are not really constants; they are updated differently
    run_variance = Constant(0, shape=norm_shape, dtype=stat_dtype, name='aggregate_variance')
    run_count    = Constant(0, shape=(),         dtype=stat_dtype, name='aggregate_count')

    # expression
    @BlockFunction('BatchNormalization', name)
    def batch_normalize(x):
        return batch_normalization(x, scale, bias, run_mean, run_variance, running_count=run_count,
                                   spatial=map_rank == 1, normalization_time_constant=normalization_time_constant, blend_time_constant=blend_time_constant, epsilon=epsilon,
                                   use_cudnn_engine=not use_cntk_engine, disable_regularization=disable_regularization)

    return batch_normalize

