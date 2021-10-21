#' Create a GAN Discriminator
#'
#' Create a GAN discriminator. You can either input your own architecture or rely on the default definition.
#'
#' @param a Input a
#' @param b Input b
#'
#' @return
#' @export
#'
#' @examples
#' create_discriminator(a = "A", b = "B")
create_discriminator <- function(a = "Hello", b = "World") {
  paste(a, b)
}

