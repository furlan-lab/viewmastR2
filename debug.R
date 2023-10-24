roxygen2::roxygenize(".")



library(viewmastR2)
library(tidyr)
library(keras)
vm_demo(FUNC="softmax_regression", device = "CPU")



vm_demo(FUNC="keras", device = 0)
