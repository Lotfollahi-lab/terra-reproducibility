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
## Imports
##========================
rm(list=ls())
library(dplyr)
library(SRTsim)
library(SingleCellExperiment)
library(zellkonverter)

##========================
## Overwrite simulation functions to use CellPhoneDB v5.0.0 data as input (including protein complexes)
##========================
srtsim_cci_ref = function(
    EstParam = NULL,
    numGene = 1000,
    location_in,
    region_cell_map,
    cci_in,
    sim_seed = 1,
    numKNN = 6,
    numSingleCellType = 2000){	
  
  set.seed(sim_seed)
  if(is.null(EstParam)){stop("EstParam is NULL, consider srtsim_cci_free for reference free simulation!")}
  if(!is.list(EstParam)){stop("EstParam has to be a list!!")}
  if(length(EstParam)< ncol(region_cell_map)){
    warning("The length of EstParam is less than targeted celltype number.")
    idx1 				<- sample(1:length(EstParam),ncol(region_cell_map),replace=TRUE)
    newEstParam <- EstParam[idx1]
    names(newEstParam) <- paste0("Celltype",1:ncol(region_cell_map))
  }else if(length(EstParam)> ncol(region_cell_map)){
    idx1 <- sample(1:length(EstParam),ncol(region_cell_map),replace=FALSE)
    newEstParam <- EstParam[idx1]
    names(newEstParam) <- paste0("Celltype",1:ncol(region_cell_map))
    print(paste0(paste(names(EstParam)[idx1],collapse=",")," are selected for data generation"))
  }else if(length(EstParam)== ncol(region_cell_map)){
    newEstParam <- EstParam
    names(newEstParam) <- paste0("Celltype",1:ncol(region_cell_map))
    print(paste0(names(EstParam)," are renamed to ",paste0("Celltype",1:ncol(region_cell_map))))
  }
  
  
  numLoc 	= nrow(location_in)
  region_label = location_in$region_label
  
  # numEntry = numSingleCellType*numGene # 5000 gene x 2000 cells to be sampled for each cell type
  
  # assign cell types to regions
  print("Generate Gene Expression Data...")
  celltype_count_in = list()
  for(celltype in colnames(region_cell_map)){
    celltype_param 	<- newEstParam[[celltype]]$marginal_param1
    idx_gene 				<- sample(1:nrow(celltype_param),numGene,replace=TRUE)
    tmp_parm 				<- celltype_param[idx_gene,1:3] ## ignore the model
    
    count_simu 			<- t(apply(tmp_parm,1,function(param_in){rbinom(numSingleCellType, 1, 1-param_in[1]) * rnbinom(numSingleCellType, size = param_in[2], mu = param_in[3])}))
    
    celltype_count_in[[celltype]] = count_simu
    rm(celltype,count_simu)
  }
  
  # assign cell types to regions
  print("Generate Cell Types...")
  ztrue 	<- region_label
  c_simu 	<- rep(NA, length(ztrue))
  
  for(z in unique(region_label)){
    zi_idx <- ztrue == z # zi_idx is index for region z
    c_simu[zi_idx] <- sample(colnames(region_cell_map), sum(zi_idx), prob = region_cell_map[z,], replace = T)
  }
  
  # assign count data
  sim_cnt <- array(NA, c(numGene, numLoc))
  for(ct in unique(c_simu)){
    c_size <- sum(c_simu == ct)  # true number of cells in cell type c
    if(dim(celltype_count_in[[ct]])[2]<c_size){
      stop("Targeted cell number of ", ct, " is greater than prespecified cell number in background pool. Increase the numSingleCellType value")
    }else{
      cells_select <- sample(1:dim(celltype_count_in[[ct]])[2], c_size, replace = F)
    }
    
    # sample same number of group c cells in real data from generated cells
    sim_cnt[, c_simu == ct] <- as.matrix(celltype_count_in[[ct]][, cells_select])
    # for positions of original cell type c cells, assign generated counts of group c
  }
  colnames(sim_cnt) <- paste0("Cell" ,1:numLoc)
  rownames(sim_cnt) <- paste0("Gene" ,1:numGene)
  
  knn.list = FNN::get.knn(as.matrix(location_in[,c("x","y")]), k = numKNN)[[1]]
  gene_names = unique(unlist(cci_in[,1:2]))
  
  simu_count = sim_cnt
  # gene_names = unique(unlist(LR_dat))
  
  # sample genes to keep from GP genes
  if(dim(simu_count)[1] < length(gene_names)){
    gene_names <- sample(gene_names, dim(simu_count)[1], replace = FALSE)
  }
  
  rownames(simu_count)[1:length(gene_names)] = gene_names
  # mat_foldchange_record = matrix(0, length(gene_names), dim(simu_count)[2])
  
  print("Construct Communications...")
  for(celltype_A in unique(cci_in$celltypeA)){
    cellid_A = which(c_simu==celltype_A)
    for(cellid in cellid_A){
      nearest_cells = knn.list[cellid,]
      for(nearid in nearest_cells){
        nearest_celltypeB = c_simu[nearid]
        for(fc in unique(cci_in$fold_change)){
          tmp_GP = cci_in[which(cci_in$celltypeA == celltype_A &
                                  cci_in$fold_change == fc &
                                  cci_in$celltypeB %in% nearest_celltypeB),]
          source_gene_ind=na.omit(match(c(unlist(tmp_GP$sources)),rownames(simu_count)))
          target_gene_ind=na.omit(match(c(unlist(tmp_GP$targets)),rownames(simu_count)))
          if(length(source_gene_ind)>0)
            simu_count[target_gene_ind,cellid] = fc*sim_cnt[target_gene_ind,cellid]
          if(length(target_gene_ind)>0)
            simu_count[source_gene_ind,nearid] = fc*sim_cnt[source_gene_ind,nearid]
        }
      }
    }
  }
  
  simInfo <- cbind.data.frame(location_in,celltype=c_simu)
  simsrt 	<- new(
    Class = "simSRT",
    simCounts = as(as.matrix(simu_count),"sparseMatrix"),
    simcolData = as(simInfo,"DataFrame"),
    metaParam = list(
      simSeed = sim_seed,
      simCCIKNN= numKNN,
      simParam = list(background_param = newEstParam),
      simType = "CCI_REF")
  )
  return(simsrt)
}

srtsim_cci_free = function(
    # param_in = c(0,1,1),
  zero_prop_in = 0,
  disper_in = Inf,
  mu_in = 1, 
  numGene = 1000,
  location_in,
  region_cell_map,
  cci_in,
  sim_seed = 1,
  numKNN = 6,
  numSingleCellType = 2000){	
  
  
  param_in <- c(zero_prop_in,disper_in,mu_in)
  
  set.seed(sim_seed)
  numLoc 	= nrow(location_in)
  region_label = location_in$region_label
  
  numEntry = numSingleCellType*numGene # 5000 gene x 2000 cells to be sampled for each cell type
  
  # assign cell types to regions
  print("Generate Gene Expression Data...")
  celltype_count_in = list()
  for(celltype in colnames(region_cell_map)){
    count_simu = rbinom(numEntry, 1, 1-param_in[1]) * rnbinom(numEntry, size = param_in[2], mu = param_in[3])
    celltype_count_in[[celltype]] = matrix(count_simu, numGene, numSingleCellType)
    rm(celltype)
  }
  
  # assign cell types to regions
  print("Generate Cell Types...")
  ztrue 	<- region_label
  c_simu 	<- rep(NA, length(ztrue))
  
  for(z in unique(region_label)){
    zi_idx <- ztrue == z # zi_idx is index for region z
    c_simu[zi_idx] <- sample(colnames(region_cell_map), sum(zi_idx), prob = region_cell_map[z,], replace = T)
  }
  
  # assign count data
  sim_cnt <- array(NA, c(numGene, numLoc))
  for(ct in unique(c_simu)){
    c_size <- sum(c_simu == ct)  # true number of cells in cell type c
    if(dim(celltype_count_in[[ct]])[2]<c_size){
      stop("Targeted cell number of ", ct, " is greater than prespecified cell number in background pool. Increase the numSingleCellType value")
    }else{
      cells_select <- sample(1:dim(celltype_count_in[[ct]])[2], c_size, replace = F)
    }
    
    # sample same number of group c cells in real data from generated cells
    sim_cnt[, c_simu == ct] <- as.matrix(celltype_count_in[[ct]][, cells_select])
    # for positions of original cell type c cells, assign generated counts of group c
  }
  colnames(sim_cnt) <- paste0("Cell" ,1:numLoc)
  rownames(sim_cnt) <- paste0("Gene" ,1:numGene)
  
  knn.list = FNN::get.knn(as.matrix(location_in[,c("x","y")]), k = numKNN)[[1]]
  gene_names = unique(unlist(cci_in[,1:2]))
  
  simu_count = sim_cnt
  # gene_names = unique(unlist(LR_dat))
  
  # sample genes to keep from GP genes
  if(dim(simu_count)[1] < length(gene_names)){
    gene_names <- sample(gene_names, dim(simu_count)[1], replace = FALSE)
  }
  
  rownames(simu_count)[1:length(gene_names)] = gene_names
  # mat_foldchange_record = matrix(0, length(gene_names), dim(simu_count)[2])
  
  print("Construct Communications...")
  for(celltype_A in unique(cci_in$celltypeA)){
    cellid_A = which(c_simu==celltype_A)
    for(cellid in cellid_A){
      nearest_cells = knn.list[cellid,]
      for(nearid in nearest_cells){
        nearest_celltypeB = c_simu[nearid]
        for(fc in unique(cci_in$fold_change)){
          tmp_GP = cci_in[which(cci_in$celltypeA == celltype_A &
                                  cci_in$fold_change == fc &
                                  cci_in$celltypeB %in% nearest_celltypeB),]
          source_gene_ind=na.omit(match(c(unlist(tmp_GP$sources)),rownames(simu_count)))
          target_gene_ind=na.omit(match(c(unlist(tmp_GP$targets)),rownames(simu_count)))
          if(length(source_gene_ind)>0)
            simu_count[target_gene_ind,cellid] = fc*sim_cnt[target_gene_ind,cellid]
          if(length(target_gene_ind)>0)
            simu_count[source_gene_ind,nearid] = fc*sim_cnt[source_gene_ind,nearid]
        }
      }
    }
  }
  
  simInfo <- cbind.data.frame(location_in,celltype=c_simu)
  simsrt 	<- new(
    Class = "simSRT",
    simCounts = as(as.matrix(simu_count),"sparseMatrix"),
    simcolData = as(simInfo,"DataFrame"),
    metaParam = list(
      simSeed = sim_seed,
      simCCIKNN= numKNN,
      simParam = list(background_param = param_in),
      simType = "CCI")
  )
  return(simsrt)
}

unlockBinding("srtsim_cci_ref", asNamespace("SRTsim"))
assign("srtsim_cci_ref", srtsim_cci_ref, envir = asNamespace("SRTsim"))
lockBinding("srtsim_cci_ref", asNamespace("SRTsim"))

unlockBinding("srtsim_cci_free", asNamespace("SRTsim"))
assign("srtsim_cci_free", srtsim_cci_ref, envir = asNamespace("SRTsim"))
lockBinding("srtsim_cci_free", asNamespace("SRTsim"))

##========================
## Config
##========================
st_data_bronze_folder_path = "./" # "../datasets/st_data/bronze"
sim_folder_path = "./" # "../datasets/cci_data/simulations"

isid = 1
set.seed(isid)

r = 8
c = 8
nGene = 1354
nLoc = 10000
p = 0.2
fc = 3

region_celltype_df <- matrix(0.1,r,c)
diag(region_celltype_df) <- 0.7
rownames(region_celltype_df) <- paste0("Region",1:8)
colnames(region_celltype_df) <- paste0("Celltype",1:8)
region_celltype_df[1, 5] <- 0.
region_celltype_df[1, 6] <- 0.
region_celltype_df[1, 7] <- 0.
region_celltype_df[1, 8] <- 0.
region_celltype_df[2, 5] <- 0.
region_celltype_df[2, 6] <- 0.
region_celltype_df[2, 7] <- 0.
region_celltype_df[2, 8] <- 0.
region_celltype_df[3, 3] <- 0.5
region_celltype_df[3, 7] <- 0.
region_celltype_df[3, 8] <- 0.
region_celltype_df[4, 4] <- 0.5
region_celltype_df[4, 7] <- 0.
region_celltype_df[4, 8] <- 0.
region_celltype_df[5, 1] <- 0.
region_celltype_df[5, 2] <- 0.
region_celltype_df[5, 3] <- 0.
region_celltype_df[5, 4] <- 0.
region_celltype_df[5, 5] <- 1
region_celltype_df[5, 6] <- 0.
region_celltype_df[5, 7] <- 0.
region_celltype_df[5, 8] <- 0.
region_celltype_df[6, 1] <- 0.
region_celltype_df[6, 2] <- 0.
region_celltype_df[6, 3] <- 0.
region_celltype_df[6, 4] <- 0.
region_celltype_df[6, 5] <- 0.
region_celltype_df[6, 6] <- 1
region_celltype_df[6, 7] <- 0.
region_celltype_df[6, 8] <- 0.
region_celltype_df[7, 1] <- 0.125
region_celltype_df[7, 2] <- 0.125
region_celltype_df[7, 3] <- 0.125
region_celltype_df[7, 4] <- 0.125
region_celltype_df[7, 5] <- 0.125
region_celltype_df[7, 6] <- 0.125
region_celltype_df[7, 7] <- 0.125
region_celltype_df[7, 8] <- 0.125
region_celltype_df[8, 1] <- 0.
region_celltype_df[8, 2] <- 0.
region_celltype_df[8, 3] <- 0.
region_celltype_df[8, 4] <- 0.
region_celltype_df[8, 5] <- 0.25
region_celltype_df[8, 6] <- 0.25
region_celltype_df[8, 7] <- 0.25
region_celltype_df[8, 8] <- 0.25

##========================
## Load reference data
##========================
load(paste0(st_data_bronze_folder_path, "starmap_mouse_visual_cortex_simulation_reference/starmap_1020_0410_seurat_filter_layer.rds"))
info2   <- info %>% select(c(x,y,label)) %>% 
  filter(label!="Smc") %>% as.data.frame()

##========================
## Simulate spatial locations
##========================
simLoc <- simNewLocs(newN= nLoc,lay_out="random",preLoc = info2)

## assign the region labels
simLoc %<>% mutate(region_label = case_when(
  x <= 0.5*median(x) & y <= median(y) ~ "Region1",
  x <= 0.5*median(x) & y > median(y) ~ "Region2",
  x > 0.5*median(x) & x <= median(x) & y <= median(y) ~ "Region3",
  x > 0.5*median(x) & x <= median(x) & y > median(y) ~ "Region4",
  x > median(x) & x <= 1.5*median(x) & y <= median(y) ~ "Region5",
  x > median(x) & x <= 1.5*median(x) & y > median(y) ~ "Region6",
  x > 1.5*median(x) & y <= median(y) ~ "Region7",
  TRUE ~ "Region8"
)) %>% as.data.frame()

##========================
## Load CCI data
##========================
cci_file_path <- paste(sim_folder_path, "simulated_cci_", p*100, "p_", fc, "fc.csv", sep = "")
cci_df <- read_csv(cci_file_path)

cci_df$sources <- lapply(cci_df$sources, function(x) unlist(strsplit(x, ",")))
cci_df$targets <- lapply(cci_df$targets, function(x) unlist(strsplit(x, ",")))

##========================
## Generate reference-based data
##========================
simSRT  <- createSRT(count_in=sp_count[,rownames(info2)],loc_in =info2)
simSRT1 <- srtsim_fit(simSRT,sim_schem="domain")

example_CCI_ref = srtsim_cci_ref(
  EstParam = simSRT1@EstParam,
  numGene = nGene,
  location_in  = simLoc[,c("x","y","region_label")],
  # region_label = region_label,
  region_cell_map = region_celltype_df,
  sim_seed = isid,
  cci_in = cci_df,
  numSingleCellType = nLoc/5*2)

##========================
## Generate reference-free data
##========================
example_CCI_free = srtsim_cci_free(
  zero_prop_in = 0,
  disper_in = Inf,
  mu_in = 1, 
  numGene = nGene,
  location_in  = simLoc[,c("x","y","region_label")],
  # region_label = region_label,
  region_cell_map = region_celltype_df,
  sim_seed = isid,
  cci_in = cci_df,
  numSingleCellType = nLoc/5*2)

cci_df$sources <- sapply(cci_df$sources, function(x) paste(x, collapse = ","))
cci_df$targets <- sapply(cci_df$targets, function(x) paste(x, collapse = ","))

##========================
## Save to h5ad file
##========================
sce_ref <- SingleCellExperiment(
  list(counts=example_CCI_ref@simCounts),
  colData=DataFrame(region_label=example_CCI_ref@simcolData$region_label,
                    cell_type=example_CCI_ref@simcolData$celltype,
                    x=example_CCI_ref@simcolData$x,
                    y=example_CCI_ref@simcolData$y))

rownames(sce_ref) <- example_CCI_ref@simCounts@Dimnames[[1]]
colnames(sce_ref) <- example_CCI_ref@simCounts@Dimnames[[2]]
save_path <- paste(st_data_bronze_folder_path, "sim_ref_", nGene, "fts_", nLoc, "locs_", r, "r_", c, "c_", p*100, "p_", fc, "fc", "/sim_ref_", nGene, "fts_", nLoc, "locs_", r, "r_", c, "c_", p*100, "p_", fc, "fc", ".h5ad", sep = "")
writeH5AD(sce_ref, file = save_path)

sce_free <- SingleCellExperiment(
  list(counts=example_CCI_free@simCounts),
  colData=DataFrame(region_label=example_CCI_free@simcolData$region_label,
                    cell_type=example_CCI_free@simcolData$celltype,
                    x=example_CCI_free@simcolData$x,
                    y=example_CCI_free@simcolData$y))

rownames(sce_free) <- example_CCI_free@simCounts@Dimnames[[1]]
colnames(sce_free) <- example_CCI_free@simCounts@Dimnames[[2]]
save_path <- paste(st_data_bronze_folder_path, "sim_free_", nGene, "fts_", nLoc, "locs_", r, "r_", c, "c_", p*100, "p_", fc, "fc", "/sim_free_", nGene, "fts_", nLoc, "locs_", r, "r_", c, "c_", p*100, "p_", fc, "fc", ".h5ad", sep = "")
writeH5AD(sce_free, file = save_path)