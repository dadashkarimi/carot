library('Ternary')
library(hrbrthemes)
library(gcookbook)
library(tidyverse)
library("wesanderson")
library(scales)
library(ggpubr)

# current verison

dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["shen368"]] <- "Shen368"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["craddock"]] <- "Crad"
dict[["brainnetome"]] <- "Btm246"
dict[["power"]] <- "Power"
dict[["dosenbach"]] <- "Dsnbch"
dict[["relational"]] <- "rel"
dict[["emotion"]] <- "emtn"
dict[["wm"]] <- "wm"
dict[["motor"]] <- "motor"
dict[["gambling"]] <- "gam"
dict[["lang"]] <- "lang"
dict[["social"]] <- "soc"
dict[["rest1"]] <- "rest1"


task_list <-c("rest1","gambling","wm","motor","emotion","relational","lang","social")

# atlas_list <- c("shen","shen368","schaefer","craddock","craddock400","brainnetome","power","dosenbach")
atlas_list <- c("dosenbach","schaefer","brainnetome")


get_ot_results <- function(filename,target,task,alpha,beta) {
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
  c_data <- data.frame(mean(atlas_data$pearson),sd(atlas_data$pearson),target, dict[[task]],alpha,beta)
  colnames(c_data) <-c("value","std","target","task","alpha","beta")
  if(alpha!=beta){
    c_data2 <- data.frame(mean(atlas_data$pearson),sd(atlas_data$pearson),target, dict[[task]],beta,alpha)
    colnames(c_data2) <-c("value","std","target","task","alpha","beta")
    c_data <-rbind(c_data,c_data2)
  }
  
  return(c_data)
}

get_data <- function(task,target) {
  my_data <- list()
  # i<-2
  data_index =1
  for(alpha in seq(0, 11, 2)){
    for(beta in seq(alpha, 11, 2)){
      if(alpha+beta<=10){
        f <-paste("simplex",target,task,alpha,beta,"iq",sep="_")
        my_data[[data_index]] <- get_ot_results(paste(f,".csv",sep=""),target,task,alpha,beta)
        data_index <-data_index+1
      }
      
    }
  }
  cdata <- bind_rows(my_data)
  return(cdata)
  
}

generate_data <- function(cdata){
  nPoints <- floor(cdata1[1, "value"]*100)
  sd <-floor(cdata[1, "std"]*100)
  alpha <-cdata[1, "alpha"]
  beta <-cdata[1, "beta"]
  gamma<-1-(alpha+beta)
  
  coordinates <-  cbind(abs(rnorm(nPoints, alpha, sd)),
                        abs(rnorm(nPoints, beta, sd)),
                        abs(rnorm(nPoints, gamma, sd)))
  
  for( i in 2:length(cdata[,1])){
    nPoints <- floor(cdata[1, "value"]*100)
    sd <-floor(cdata[1, "std"]*100)
    alpha <-cdata[1, "alpha"]
    beta <-cdata[1, "beta"]
    gamma<-1-(alpha+beta)
    coordinates <- rbind(coordinates,
                         cbind(abs(rnorm(nPoints, alpha, sd)),
                               abs(rnorm(nPoints, beta, sd)),
                               abs(rnorm(nPoints, gamma, sd))))
  }
  colnames(coordinates) <-c("atlas 1","atlas 2","atlas3")
  return(coordinates)
}

tasks <-c("gambling","motor","wm","rest1","emotion","relational","lang")
target <- "shen"

Cairo::Cairo(20,6,
             file = paste("simplex","all",".png"),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)
spectrum <- viridisLite::viridis(256L, alpha = 0.6)
par(mfrow=c(1, length(tasks)), mar=rep(0.3, 4))

for(task in tasks){
  cdata<- get_data(task,target)
  coordinates <- generate_data(cdata)
  TernaryPlot(clab=task,
              atip = expression(A),
              btip = expression(C),
              ctip = expression(B))
  
  ColourTernary(TernaryDensity(coordinates, resolution = 10L))
  TernaryDensityContour(coordinates, resolution = 36L)
}

legend('topright', 
       pch=21, pt.cex=1.8,
       pt.bg=c(spectrum[255], 
               spectrum[100],
               spectrum[1]),
       legend=c('High', 'Medium', 'Low'), 
       cex=0.8, bty='n')
dev.off()