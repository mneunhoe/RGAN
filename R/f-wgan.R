# We will use the kl GAN loss
# You can find the paper here: https://arxiv.org/abs/1910.09779
# And the original python implementation here: https://github.com/ermongroup/f-wgan

kl_real <- function(dis_real) {
  loss_real <- torch::torch_mean(torch::nnf_relu(1 - dis_real))

  return(loss_real)
}

kl_fake <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss_fake = torch::torch_mean(torch::nnf_relu(1. + dis_fake))

  return(loss_fake)
}

kl_gen <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss = -torch::torch_mean(dis_fake)
  return(loss)
}
