import tensorflow as tf
import os
import numpy as np


def init(H, config=None):
    """
    Given an input hyperparameter config H, returns an object with properties:
    "W": two random variable arrays 
    "B": two random variable arrays 
    "weight_tensors": all weight and bias layers from GoogleNet
    "reuse_ops": all other useful ops from GoogleNet
    "input_op": the GoogleNet input op
    "W_norm": a weight decay term for all weights, W and weight_tensors
    """


    # No idea what this does
    if config is None:
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
    
    k = H['arch']['num_classes']
    
    # This might be the number of neurons in the last fully connected layers???
    features_dim = 1024
    
    features_layers = ['output/confidences', 'output/boxes']

    graph_def_orig_file = '%s/../data/googlenet.pb' % os.path.dirname(os.path.realpath(__file__))

    dense_layer_num_output = [k, 4]
    
    # Load in the GoogleNet graph from file
    googlenet_graph = tf.Graph()
    graph_def = tf.GraphDef()
    tf.set_random_seed(0)
    with open(graph_def_orig_file, "rb") as f:
        tf.set_random_seed(0)
        graph_def.MergeFromString(f.read())

    with googlenet_graph.as_default():
        tf.import_graph_def(graph_def, name='')
    
    # Grab the input op
    input_op = googlenet_graph.get_operation_by_name('input')
    
    # Get the weight ops
    weights_ops = [
        op for op in googlenet_graph.get_operations() 
        if any(op.name.endswith(x) for x in [ '_w', '_b'])
        and op.type == 'Const'
    ]
    
    # Get a list of other ops that we might want
    reuse_ops = [
        op for op in googlenet_graph.get_operations() 
        if op not in weights_ops + [input_op]
        and op.name != 'output'
    ]
    
    # Make a dict with the original weights and biases 
    with tf.Session(graph=googlenet_graph, config=config):
        weights_orig = {
            op.name: op.outputs[0].eval()
            for op in weights_ops
        }
    
    # Initialize a random numpy array of size features_dim (1024 currentlY) x num_output
    def weight_init(num_output):
        return 0.001 * np.random.randn(features_dim, num_output).astype(np.float32)
    
    # Initialize a random numpy array of size num_output
    def bias_init(num_output):
        return 0.001 * np.random.randn(num_output).astype(np.float32)
    
    # Currently, creates one weight layer for each of ['output/confidences', 'output/boxes'] 
    # confidences layer is [1024, num_classes]
    # boxes layer is [1024, 4]
    # Would do different things given different local variable values
    W = [
        tf.Variable(weight_init(dense_layer_num_output[i]), 
                    name='softmax/weights_{}'.format(i)) 
        for i in range(len(features_layers))
    ]
    
    # Same as above for the biases
    # confidences is [num_classes]
    # biases is [4]
    B = [
        tf.Variable(bias_init(dense_layer_num_output[i]),
                    name='softmax/biases_{}'.format(i)) 
        for i in range(len(features_layers))
    ]

    # Clone each weight/bias layer from GoogleNet into a new Variable
    weight_vars = {
        name: tf.Variable(weight, name=name)
        for name, weight in weights_orig.items()
    }
    
    # Make a copy weight_vars, but with the Variable op converted to a tensor (what does that do???)
    weight_tensors = {
        name: tf.convert_to_tensor(weight)
        for name, weight in weight_vars.items()
    }
    
    # Create weight decay terms for each weight/bias layer (should biases be decayed???)
    W_norm = [tf.nn.l2_loss(weight) for weight in list(weight_vars.values()) + W]
    
    # Sum up all of the weight decay terms
    W_norm = tf.reduce_sum(tf.pack(W_norm), name='weights_norm')
    
    # Create a summary of the weight decay term
    tf.scalar_summary(W_norm.op.name, W_norm)

    googlenet = {
        "W": W,
        "B": B,
        "weight_tensors": weight_tensors,
        "reuse_ops": reuse_ops,
        "input_op": input_op,
        "W_norm": W_norm,
        }
    return googlenet

# Returns the last convolutional layer from a new network created from GoogleNet
def model(x, googlenet, H):
    weight_tensors = googlenet["weight_tensors"]
    input_op = googlenet["input_op"]
    reuse_ops = googlenet["reuse_ops"]
    
    # Test if a layer name starts with any of the "early loss" layer name prefixes (what is an early loss layer???)
    def is_early_loss(name):
        early_loss_layers = ['head0', 'nn0', 'softmax0', 'head1', 'nn1', 'softmax1', 'output1']
        return any(name.startswith(prefix) for prefix in early_loss_layers)

    T = weight_tensors
    # What the hell does this do??? I don't think input_op.name is in T, so probably just makes a new key-value pair
    T[input_op.name] = x

    for op in reuse_ops:
        # Skip "early loss" layers
        if is_early_loss(op.name):
            continue
            
        # Create an average pooling layer with size [1, grid_height, grid_width, 1] with stride [1,1,1,1]
        # No idea why this needs to be part of this loop - doesn't use anything from the original op at all
        # Probably so this op doesn't fall into the "else" category - needs some special handling
        elif op.name == 'avgpool0':
            pool_op = tf.nn.avg_pool(T['mixed5b'], ksize=[1,H['arch']['grid_height'],H['arch']['grid_width'],1], strides=[1,1,1,1], padding='VALID', name=op.name)
            T[op.name] = pool_op
        
        # Otherwise, clone the layer into our new graph
        else:
            copied_op = x.graph.create_op(
                op_type = op.type, 
                inputs = [T[t.op.name] for t in list(op.inputs)], 
                dtypes = [o.dtype for o in op.outputs], 
                name = op.name, 
                attrs =  op.node_def.attr
            )

            T[op.name] = copied_op.outputs[0]
            #T[op.name] = tf.Print(copied_op.outputs[0], [tf.shape(copied_op.outputs[0]), tf.constant(op.name)])
    
    # Get the last layer in our new graph
    cnn_feat = T['mixed5b']
    # reshape to some weird semi-flat shape
    cnn_feat_r = tf.reshape(cnn_feat, [H['arch']['batch_size'] * H['arch']['grid_width'] * H['arch']['grid_height'], 1024])
    
    # Get "finegrain" op (why???)
    # These ops aren't referenced anywhere else here. I think they're useless and can be deleted???
    finegrain = T['mixed3b']
    finegrain_r = tf.reshape(cnn_feat, [H['arch']['batch_size'] * H['arch']['grid_width'] * H['arch']['grid_height'] * 4 * 4, 64])

    return cnn_feat_r
