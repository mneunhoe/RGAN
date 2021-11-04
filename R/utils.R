torch_rand_ab <- function(shape, a = -1, b = 1, ...) {
  (a-b) * torch::torch_rand(shape, ...) + b
}
