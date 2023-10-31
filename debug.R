
rm(list=ls())
roxygen2::roxygenize(".")

library(viewmastR2)
test_backends()


library(Seurat)
seu<-readRDS("/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/experiments/patient_marrows/aggr/cds/indy/220831_TN2.RDS")
DimPlot(seu)
rna<-readRDS("/Users/sfurlan/Library/CloudStorage/OneDrive-SharedLibraries-FredHutchinsonCancerCenter/Furlan_Lab - General/datasets/Healthy_BM_greenleaf/230329_rnaAugmented_seurat.RDS")
DimPlot(rna)
vg<-common_variant_genes(rna, seu, top_n = 3000)
undebug(viewmastR)
seu<-viewmastR(seu, rna, ref_celldata_col = "SFClassification", query_celldata_col = "smr_pred_dense", selected_genes = vg, FUNC = "softmax_regression", use_sparse = F)
DimPlot(seu, group.by = "smr_pred_dense")
seu<-viewmastR(seu, rna, ref_celldata_col = "SFClassification", query_celldata_col = "smr_pred_sparse", selected_genes = vg, FUNC = "softmax_regression", use_sparse = T)
DimPlot(seu, group.by = "smr_pred_sparse")

seu<-viewmastR(seu, rna, ref_celldata_col = "SFClassification", query_celldata_col = "nn_pred_dense", selected_genes = vg, FUNC = "neural_network", use_sparse = F)
DimPlot(seu, group.by = "nn_pred_dense")
seu<-viewmastR(seu, rna, ref_celldata_col = "SFClassification", query_celldata_col = "nn_pred_sparse", selected_genes = vg, FUNC = "neural_network", use_sparse = T)
DimPlot(seu, group.by = "smr_pred_sparse")




library(tidyr)

args <- readRDS(file.path("~/Desktop/testlist.RDS"))
test_notsparse<-do.call(smr, args)

args[[1]] <- rbind(rep(1, ncol(args[[1]])), args[[1]]) %>% t() %>% as(., "RsparseMatrix")
args[[2]] <- rbind(rep(1, ncol(args[[2]])), args[[2]]) %>% t() %>% as(., "RsparseMatrix")
args[[6]] <- rbind(rep(1, ncol(args[[6]])), args[[6]]) %>% t() %>% as(., "RsparseMatrix")

test_sparse<-do.call(smr_sparse, args)

cts<-seu@assays$RNA@counts
class(x)
class(cts2)
x <- as(matrix(c(1, 0, 0, 2, 3,
                 0, 0, 1, 0, 2), 2, 5), 'dgRMatrix')
x1 <- as(matrix(c(1, 0, 0, 2, 3,
                  0, 0, 1, 0, 2), 2, 5), 'dgCMatrix')
x2<-as(x1, "RsparseMatrix")


cts3<-as(as.matrix(cts),  'dgRMatrix')
x2<-as(x, "RsparseMatrix")
cts2<-as(cts, "RsparseMatrix")
times_two(cts3)
times_two(x2)
x1
library('Matrix')
x <- as(matrix(c(1, 0, 0, 2, 3,
                 0, 0, 1, 0, 2), 2, 5), 'dgRMatrix')
viewmastR2::times_two(x)



library(tidyr)
library(keras)
vm_demo(FUNC="softmax_regression", device = "GPU")




DimPlot(seu, group.by = "smr_celltype", cols=sfc(21))

vm_demo(FUNC="keras", device = 0)