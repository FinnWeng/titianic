import tensorflow as tf


class Classification_Net(tf.keras.Model):
    """Transformer MLP / feed-forward block."""

    # mlp_dim: int
    # dtype: Dtype = jnp.float32
    # out_dim: Optional[int] = None
    # dropout_rate: float = 0.1
    # kernel_init: Callable[[PRNGKey, Shape, Dtype],
    #                         Array] = nn.initializers.xavier_uniform()
    # bias_init: Callable[[PRNGKey, Shape, Dtype],
    #                     Array] = nn.initializers.normal(stddev=1e-6)
    
    def __init__(self, mlp_dim = 10, layer_num = 10, out_dim = 2, name = None):
        super(Classification_Net, self).__init__(name)
        # actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.layer_num = layer_num

        self.d1 = tf.keras.layers.Dense(self.mlp_dim)
        self.layer_list = []
        for i in range(layer_num):
            self.layer_list.append(tf.keras.layers.Dense(self.mlp_dim, activation = "tanh"))

        self.head = tf.keras.layers.Dense(self.out_dim)
        
    def call(self, inputs):
        x = inputs
        x = self.d1(x)
        for ly in self.layer_list:
            x = ly(x)
        output = self.head(x)
        return output
