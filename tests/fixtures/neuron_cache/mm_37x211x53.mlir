module @jit_mm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%arg0: tensor<37x211xf32>, %arg1: tensor<211x53xf32>) -> tensor<37x53xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<37x211xf32>, tensor<211x53xf32>) -> tensor<37x53xf32>
    return %0 : tensor<37x53xf32>
  }
}
