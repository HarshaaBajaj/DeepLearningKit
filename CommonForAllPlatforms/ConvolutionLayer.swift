//
//  ConvolutionLayer.swift
//  MemkiteMetal
//
//  Created by Torb Morland & Amund Tveit on 12/12/15.
//  Copyright © 2015 Memkite. All rights reserved.
//

import Foundation
import Metal

// function name(input_param1 : data_type,input_param2: data_type ) -> (output_param1_dt output_param2_dt)

// utility function to create array from the input data
func getDataFromBlob(blob: NSDictionary) -> ([Float], [Float]) {
    print(" ==> getDataFromBlob")
    
    let shape = blob["shape"] as! NSDictionary // dimensions of the input
    let data = blob["data"] as! [Float] // get the input data
	// create float array from the input
    var FloatData = createFloatNumbersArray(data.count) // length of the array
    for i in 0 ..< data.count { // create the array
        FloatData[i] = data[i]
    }
    return (shape["dim"] as! [Float], FloatData) // return dims and input
}


// utility function for convolution operations

func createConvolutionLayerCached(layer: NSDictionary,
    inputBuffer: MTLBuffer,
    inputShape: [Float],
    metalCommandQueue: MTLCommandQueue, metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice,
    inout layer_data_caches: [Dictionary<String,MTLBuffer>],
    inout blob_cache: [Dictionary<String,([Float],[Float])>],
    layer_number: Int,
    layer_string: String) -> (MTLBuffer, MTLCommandBuffer, [Float]) {
        
        let start = NSDate()
        
        print("CREATECONVLAYERCACHED")
        
//        let metalCommandBuffer = metalCommandQueue.commandBuffer()
        let metalCommandBuffer = metalCommandQueue.commandBufferWithUnretainedReferences()
        
        var convolution_params_dict:NSDictionary = NSDictionary()
        var pad:Float = 0.0 // default padding 
        var kernel_size:Float = 1.0 // default filter size
        var stride:Float = 1.0 // default stride
        var blobs:[NSDictionary] = []
        var weights:[Float] = [] // float array for weights
        var weight_shape:[Float] = []
        var bias_data:[Float] = [] // float array for bias
        var h:Float = 0.0 // height of input
        var w:Float = 0.0 // width of input
        var result_shape:[Float] = [] // output dimensions
        var outputCount:Int = 0
        
		// create metal tensors 
        var input_dimensions:MetalTensorDimensions = MetalTensorDimensions(n: 0, channels: 0, width: 0, height:0) // w*h*d
        var weight_dimensions:MetalTensorDimensions = MetalTensorDimensions(n: 0, channels: 0, width: 0, height:0)
        var result_dimensions:MetalTensorDimensions = MetalTensorDimensions(n: 0, channels: 0, width: 0, height:0)
        var tensor_dimensions:[MetalTensorDimensions] = []
        var col_dimensions:MetalTensorDimensions = MetalTensorDimensions(n: 0, channels: 0, width: 0, height:0)
        var col_output:[Float] = []
		
		// assign convolution params
		// todo: check why kernel_size is 0 here
        var convolution_params:MetalConvolutionParameters = MetalConvolutionParameters(pad:0, kernel_size: 0, stride: 0)
        
        
            print("NOTCACHINGMODE")
            convolution_params_dict = layer["convolution_param"] as! NSDictionary
            pad = 0.0
            kernel_size = 1.0
            stride = 1.0
			
			// as? --> data type
			
            if let val = convolution_params_dict["pad"] as? Float { // if padding has been provided use that val else default to 0
                pad = val
            }
            if let val = convolution_params_dict["kernel_size"] as? Float { //if filter_kernel_size has been provided use it else default to 1
                kernel_size = val
            }
            
            let startblob = NSDate()

            
            if let tmpval = blob_cache[layer_number]["0"] {
                print("found blob key = 0 in cache")
                (weight_shape, weights) = tmpval
            } else {
                print("didnt find blob key = 0 in cache")
                blobs = layer["blobs"] as! [NSDictionary]
                (weight_shape, weights) = getDataFromBlob(blobs[0])
                blob_cache[layer_number]["0"] = (weight_shape, weights)
            }

// this can be optimized
            blobs = layer["blobs"] as! [NSDictionary]
            (_, bias_data) = getDataFromBlob(blobs[1])

            print("### Time to blob: \(NSDate().timeIntervalSinceDate(startblob))")

            
            /*
            let startblob = NSDate()
            blobs = layer["blobs"] as! [NSDictionary]
            (weight_shape, weights) = getDataFromBlob(blobs[0])
            (_, bias_data) = getDataFromBlob(blobs[1])
            print("### Time to blob: \(NSDate().timeIntervalSinceDate(startblob))")
*/

            # calculate the output dimensions
            h = (inputShape[2] + 2 * pad - kernel_size) / stride + 1 
            w = (inputShape[3] + 2 * pad - kernel_size) / stride + 1
            result_shape = [inputShape[0], weight_shape[0], h, w]
            outputCount = Int(result_shape.reduce(1, combine: *))
            
            // Create input and output vectors, and corresponding metal buffer
            input_dimensions = MetalTensorDimensions(n: inputShape[0], channels: inputShape[1], width: inputShape[2], height: inputShape[3])
            weight_dimensions = MetalTensorDimensions(n: weight_shape[0], channels: weight_shape[1], width: weight_shape[2], height: weight_shape[3])
            col_dimensions = MetalTensorDimensions(n: inputShape[0], channels: inputShape[1] * weight_shape[2] * weight_shape[3], width: inputShape[2], height: inputShape[3])
            result_dimensions = MetalTensorDimensions(n: result_shape[0], channels: result_shape[1], width: result_shape[2], height: result_shape[3])
            tensor_dimensions = [input_dimensions, weight_dimensions, col_dimensions, result_dimensions]
            
            // createFloatNumbersArray 
			array_len = int(input_shape[0]*h*w*weight_shape[0])
			temp_output = [float](array_len, repeat 0.0)
			
            col_output = createFloatNumbersArray(Int(col_dimensions.n * col_dimensions.channels * col_dimensions.height * col_dimensions.width))
            
            
            convolution_params = MetalConvolutionParameters(pad: pad, kernel_size: kernel_size, stride: stride)
            print("AFTER NOTCACHINGMODE")

        
        print("BEFORE THE BIG CALL")
        
        let resultBuffer = addConvolutionCommandToCommandBufferCached(metalCommandBuffer, inputBuffer: inputBuffer, im2ColCount: col_output.count, weights: weights, outputCount: outputCount, convolution_params: convolution_params, tensor_dimensions: tensor_dimensions, bias: bias_data, metalDefaultLibrary: metalDefaultLibrary, metalDevice:metalDevice, layer_data_caches: &layer_data_caches, layer_number: layer_number,layer_string: layer_string)
        //metalCommandBuffer.commit()
        
        print("AFTER BIG CALL")
        
        print("### Time to setup convolution layer: \(NSDate().timeIntervalSinceDate(start))")

        
        return (resultBuffer, metalCommandBuffer, result_shape)
        
}

func addConvolutionCommandToCommandBufferCached(commandBuffer: MTLCommandBuffer,
    inputBuffer: MTLBuffer,
    im2ColCount: Int,
    weights: [Float],
    outputCount: Int,
    convolution_params: MetalConvolutionParameters,
    tensor_dimensions: [MetalTensorDimensions],
    bias: [Float],
    metalDefaultLibrary:MTLLibrary, metalDevice:MTLDevice,
    inout layer_data_caches: [Dictionary<String,MTLBuffer>],
    layer_number: Int,
    layer_string: String) -> MTLBuffer {
        
        let start = NSDate()
        
        print("before output and col_output")
        
        var output:[Float] = []
        var col_output:[Float] = []
        
         output = createFloatNumbersArray(outputCount)
         col_output = createFloatNumbersArray(im2ColCount)
        
        print("before setupshaderinpipeline")
        
        let (_, im2colComputePipelineState, _) = setupShaderInMetalPipeline("im2col", metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
        
        let resultMetalBuffer = createOrReuseFloatMetalBuffer("resultMetalBuffer", data: output, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
        
        print("after resultmetalbuffer")
        
        let weightMetalBuffer = createOrReuseFloatMetalBuffer("weightMetalBuffer", data: weights, cache: &layer_data_caches, layer_number:layer_number, metalDevice: metalDevice)
        
        
        let convolutionParamsMetalBuffer = createOrReuseConvolutionParametersMetalBuffer("convolutionParamsMetalBuffer", data: convolution_params, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
        let tensorDimensionsMetalBuffer = createOrReuseTensorDimensionsVectorMetalBuffer("tensorDimensionsMetalBuffer", data: tensor_dimensions, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
        
        let colOutputMetalBuffer = createOrReuseFloatMetalBuffer("colOutputMetalBuffer", data: col_output, cache: &layer_data_caches, layer_number: layer_number, metalDevice: metalDevice)
        let biasMetalBuffer = createOrReuseFloatMetalBuffer("bias", data: bias, cache: &layer_data_caches, layer_number:layer_number, metalDevice: metalDevice)
        
        
        // Create Metal compute command encoder for im2col
        var metalComputeCommandEncoder = commandBuffer.computeCommandEncoder()
        metalComputeCommandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 0)
        metalComputeCommandEncoder.setBuffer(tensorDimensionsMetalBuffer, offset: 0, atIndex: 1)
        metalComputeCommandEncoder.setBuffer(convolutionParamsMetalBuffer, offset: 0, atIndex: 2)
        metalComputeCommandEncoder.setBuffer(colOutputMetalBuffer, offset: 0, atIndex: 3)
        //metalComputeCommandEncoder.setComputePipelineState(im2colComputePipelineState)
        
        
        // Set the shader function that Metal will use
        metalComputeCommandEncoder.setComputePipelineState(im2colComputePipelineState)
        
        // Set up thread groups on GPU
        var threadsPerGroup = MTLSize(width:im2colComputePipelineState.threadExecutionWidth,height:1,depth:1)
        // ensure at least 1 threadgroup
        print("before mtlsize 2")
        var numThreadgroups = MTLSize(width:(col_output.count-1)/im2colComputePipelineState.threadExecutionWidth + 1, height:1, depth:1)
        metalComputeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        print("after dispatch")
        
        // Finalize configuration
        metalComputeCommandEncoder.endEncoding()
        
        
        
        
        let (_, convolutionComputePipelineState, _) = setupShaderInMetalPipeline("convolution_layer", metalDefaultLibrary: metalDefaultLibrary, metalDevice: metalDevice)
        metalComputeCommandEncoder = commandBuffer.computeCommandEncoder()
        
        // Create Metal Compute Command Encoder and add input and output buffers to it
        metalComputeCommandEncoder.setBuffer(resultMetalBuffer, offset: 0, atIndex: 0)
        metalComputeCommandEncoder.setBuffer(weightMetalBuffer, offset: 0, atIndex: 1)
        metalComputeCommandEncoder.setBuffer(tensorDimensionsMetalBuffer, offset: 0, atIndex: 2)
        metalComputeCommandEncoder.setBuffer(colOutputMetalBuffer, offset: 0, atIndex: 3)
        metalComputeCommandEncoder.setBuffer(biasMetalBuffer, offset: 0, atIndex: 4)
        
        // Set the shader function that Metal will use
        metalComputeCommandEncoder.setComputePipelineState(convolutionComputePipelineState)
        
        // Set up thread groups on GPU
        threadsPerGroup = MTLSize(width:convolutionComputePipelineState.threadExecutionWidth,height:1,depth:1)
        // ensure at least 1 threadgroup
        numThreadgroups = MTLSize(width:(outputCount-1)/convolutionComputePipelineState.threadExecutionWidth + 1, height:1, depth:1)
        metalComputeCommandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        // Finalize configuration
        metalComputeCommandEncoder.endEncoding()
        
        print("after endencoding")
        
        print("#### Time to add convolution layer: \(NSDate().timeIntervalSinceDate(start))")

        
        return resultMetalBuffer
        
}

