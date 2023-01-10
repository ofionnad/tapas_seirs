# Libraries and packages
library(ggplot2)
library(DescTools)
library(Imap)

# R code to get seirs data

# create temporary environment
senegal_tmp.env <- new.env()
# load saved workspace to tmp env
load("senegal_66C_test-5Corr.RData", envir = senegal_tmp.env)
# Load specific objects from the tmp to the global env
ndvi_forecast <- get("forecast", pos=senegal_tmp.env)
ndvi_observed <- get("observed", pos=senegal_tmp.env)
# remove the tmp env
# rm(senegal_tmp.env)

# extract ndvi values only
ndvi_fc <- ndvi_forecast[,-(1:4)]
ndvi_ob <- ndvi_observed[,-(1:4)]

# differences between forecast and observed ndvi
delta_ndvi <- abs(ndvi_ob - ndvi_fc)
delta_ndvi_2 <- ndvi_ob - ndvi_fc

# histograms
ndvi_fc_vec <- as.numeric(ndvi_fc[200,])
ndvi_ob_vec <- as.numeric(ndvi_ob[200,])
delta_ndvi_vec <- as.numeric(delta_ndvi[200,])
delta_ndvi_2_vec <- as.numeric(delta_ndvi_2[200,])

# CDFs
ndvi_fc_cdf <- ecdf(ndvi_fc[200,])
ndvi_ob_cdf <- ecdf(ndvi_ob[200,])
delta_ndvi_cdf <- ecdf(delta_ndvi[200,])
delta_ndvi_2_cdf <- ecdf(delta_ndvi_2[200,])
