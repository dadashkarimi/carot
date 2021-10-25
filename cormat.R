library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(vcd)

dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["brainnetome"]] <- "Btm246"

get_ot_results <- function(filename,atlas) {
  atlas_data <- read.csv(paste("data/",file = filename,sep=""))
  if (!inherits(atlas_data, 'try-error')) atlas_data
  inp1 <- round(atlas_data$X1,digits=2)
  return(inp1)
}

get_data <- function(source,target,task) {
  f11 <-paste(source,target,task,"orig-orig","id",sep="_")
  f12 <-paste(source,target,task,"orig-ot","id",sep="_")
  f13 <-paste(source,target,task,"ot-orig","id",sep="_")
  f14 <-paste(source,target,task,"ot-ot","id",sep="_")
  
  c_data11 <- get_ot_results(paste(f11,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data12 <- get_ot_results(paste(f12,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data13 <- get_ot_results(paste(f13,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data14 <- get_ot_results(paste(f14,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  # c_data <- data.frame(c(c_data11,c_data12,c_data13,c_data14))#, ncol = 2, nrow = 2))
  # colnames(c_data) <- c("Orig","Ot")
  return(c(c_data11,c_data12,c_data13,c_data14))
  # m <- matrix(c(c_data11,c_data12,c_data13,c_data14),ncol=2)
  # colnames(m) <- c("Orig","Ot")
  # rownames(m) <- c("Orig","Ot")
  # 
  # return(as.table(m))
}

my_plot<-function(source,target,task1,task2){
  c_data11 <- get_data(source,target,task1)
  c_data12 <- get_data(source,target,task2)
  c_data21 <- get_data(target,source,task1)
  c_data22 <- get_data(target,source,task2)
  
  # target <- toupper(target[1:2]) 
  # source <-toupper(source[1:2])
  # task1<-toupper(task1)
  # task2 <-toupper(task2)
  
  ar <- array(1:16, c(2, 2, 4))
  
  ar[,,1] <-c_data11
  ar[,,2] <-c_data12
  ar[,,3] <-c_data21
  ar[,,4] <-c_data22

  c_table <- array(data     = ar,
                   dim      = c(2, 2, 4),
                   dimnames = list(c("orig","ot"),c("orig","ot"),c(paste(dict[[target]],"-",toupper(task1)),
                                                                   paste(dict[[target]],"-",toupper(task2)),
                                                                   paste(dict[[source]],"-",toupper(task1)),
                                                                   paste(dict[[source]],"-",toupper(task2))))
  )
  
  names(dimnames(c_table)) <- c("row", "col", "Test")
  p<-fourfold(c_table, fontsize = 17, color = c("#99CCFF", "#6699CC", "#FFA0A0", "#A0A0FF", "#FF0000", "#000080"))
  return(p)
}
task1 = "rest1"
task2 = "rest2"

source = "shen"
target = "brainnetome"




Cairo::Cairo(40,40,
             file = paste(paste("figs/",source,target,task1,task2,"cormat",sep="_"), ".png", sep = ""),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)

p <-my_plot(source,target,task1,task2)

p
dev.off()

