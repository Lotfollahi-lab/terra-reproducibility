##========================
## Installation (MacOS)
##========================
# SRTsim requirements
install.packages("rgl") # on MacOS this requires brew install xquartz, brew install mesa, brew install libpng
install.packages("units") # on MacOS, this requires brew install udunits
Sys.setenv(PATH = paste("/opt/homebrew/bin", Sys.getenv("PATH"), sep = ":"))
install.packages("sf") # on MacOS, this requires brew install gdal
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("S4Vectors")
install.packages("nloptr") # on MacOS, this requires brew install cmake
install.packages("lme4")
install.packages("ggpubr")

# SRTsim
install.packages('devtools')
devtools::install_github('xzhoulab/SRTsim')

# SingleCellExperiment
Sys.setenv(PATH = paste("/opt/gcc/bin", Sys.getenv("PATH"), sep = ":"))
Sys.setenv(PATH = paste("/opt/llvm/bin", Sys.getenv("PATH"), sep = ":"))
BiocManager::install(version = "3.19")
BiocManager::install("zellkonverter") # on MacOS, use gcc compiler
# brew install gcc
# mkdir -p ~/.R
# nano ~/.R/Makevars
# CC = /usr/local/bin/gcc-10
# CXX = /usr/local/bin/g++-10

##========================
## Installation (MacOS)
##========================
rm(list=ls())
library(dplyr)
library(SRTsim)
library(SingleCellExperiment)
library(zellkonverter)

isid = 1
foldchange = 10 # foldchange, 1, 3, 5, 10
set.seed(isid)

load(paste0("./starmap_1020_0410_seurat_filter_layer.rds"))
info2   <- info %>% select(c(x,y,label)) %>% 
				filter(label!="Smc") %>% as.data.frame()

region_celltype_df <- matrix(0.1,4,4)
diag(region_celltype_df) <- 0.7
rownames(region_celltype_df) <- paste0("Region",1:4)
colnames(region_celltype_df) <- paste0("Celltype",1:4)

nGene = 2000
nLoc = 5000

##========================
## simulate spatial spot
##========================
simLoc <- simNewLocs(newN= nLoc,lay_out="random",preLoc = info2)

## assign the region labels
simLoc %<>% mutate(region_label = case_when(
		x <= 0.5*median(x) ~ "Region1",
		x > 0.5*median(x) & x <= median(x)~ "Region2",
		x > median(x) & x <= 1.5*median(x)~ "Region3",
		TRUE ~ "Region4"
	)) %>% as.data.frame()


## load the Ligand-Receptor data
load("./LR_dat.RData")

##========================
## Data Generation
##========================

## reference free
example_CCI_free = srtsim_cci_free(
					zero_prop_in = 0,
					disper_in = Inf,
					mu_in = 1, 
					numGene = nGene,
					location_in  = simLoc[,c("x","y","region_label")],
					# region_label = region_label,
					region_cell_map = region_celltype_df,
					sim_seed = isid,
					fc = foldchange,
					LR_in = LR_dat)

## reference based
simSRT  <- createSRT(count_in=sp_count[,rownames(info2)],loc_in =info2)
simSRT1 <- srtsim_fit(simSRT,sim_schem="domain")

example_CCI_ref = srtsim_cci_ref(
					EstParam = simSRT1@EstParam,
					numGene = nGene,
					location_in  = simLoc[,c("x","y","region_label")],
					# region_label = region_label,
					region_cell_map = region_celltype_df,
					sim_seed = isid,
					fc = foldchange,
					LR_in = LR_dat)

sce_free <- SingleCellExperiment(
  list(counts=example_CCI_free@simCounts),
  colData=DataFrame(region_label=example_CCI_free@simcolData$region_label,
                    cell_type=example_CCI_free@simcolData$celltype,
                    x=example_CCI_free@simcolData$x,
                    y=example_CCI_free@simcolData$y))

rownames(sce_free) <- example_CCI_free@simCounts@Dimnames[[1]]
colnames(sce_free) <- example_CCI_free@simCounts@Dimnames[[2]]
writeH5AD(sce_free, file = "./simulated_free_1.h5ad")

sce_ref <- SingleCellExperiment(
  list(counts=example_CCI_ref@simCounts),
  colData=DataFrame(region_label=example_CCI_ref@simcolData$region_label,
                    cell_type=example_CCI_ref@simcolData$celltype,
                    x=example_CCI_ref@simcolData$x,
                    y=example_CCI_ref@simcolData$y))

rownames(sce_ref) <- example_CCI_ref@simCounts@Dimnames[[1]]
colnames(sce_ref) <- example_CCI_ref@simCounts@Dimnames[[2]]
writeH5AD(sce_ref, file = "./simulated_ref_1.h5ad")