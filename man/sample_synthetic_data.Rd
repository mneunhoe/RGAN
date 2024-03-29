% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sample-synthetic-data.R
\name{sample_synthetic_data}
\alias{sample_synthetic_data}
\title{Sample Synthetic Data from a trained RGAN}
\usage{
sample_synthetic_data(trained_gan, transformer = NULL)
}
\arguments{
\item{trained_gan}{A trained RGAN object of class "trained_RGAN"}

\item{transformer}{The transformer object used to pre-process the data}
}
\value{
Function to sample from a
}
\description{
Provides a function that makes it easy to sample synthetic data from a Generator
}
\examples{
\dontrun{
# Before running the first time the torch backend needs to be installed
torch::install_torch()
# Load data
data <- sample_toydata()
# Build new transformer
transformer <- data_transformer$new()
# Fit transformer to data
transformer$fit(data)
# Transform data and store as new object
transformed_data <-  transformer$transform(data)
# Train the default GAN
trained_gan <- gan_trainer(transformed_data)
# Sample synthetic data from the trained GAN
synthetic_data <- sample_synthetic_data(trained_gan, transformer)
# Plot the results
GAN_update_plot(data = data,
synth_data = synthetic_data,
main = "Real and Synthetic Data after Training")
}
}
