library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)

dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["brainnetome"]] <- "Btm246"


get_ot_results <- function(filename,atlas) {
  atlas_data <- read.csv(file = filename)
  inp1 <- data.frame(atlas_data$orig,rep("2-orig",100),rep("task",100))
  inp2 <- data.frame(atlas_data$ot,rep("3-ot",100),rep("task",100))
  colnames(inp1) <-c("value","method","task")
  colnames(inp2) <-c("value","method","task")
  c_data <- rbind(inp1,inp2)
  c_data <- data.frame(c_data,rep(atlas,200))
  colnames(c_data) <-c("value","method","task","atlas")
  
  return(c_data)
}

get_data <- function(source,target,task) {
  f1 <-paste(source,target,task,"iq",sep="_")
  f2 <-paste(target,source,task,"iq",sep="_")
  
  c_data1 <- get_ot_results(paste(f1,".csv",sep=""),paste(source,"to",target,sep = " "))
  c_data2 <- get_ot_results(paste(f2,".csv",sep=""),paste(target,"to",source,sep = " "))
  
  inp1 <- data.frame(c_data2[c_data2$method=="2-orig",]$value,rep(paste("1-",dict[[source]]),100),rep(task,100),rep(paste(source,"to",target,sep = " "),100))
  colnames(inp1) <-c("value","method","task","atlas")
  
  inp2 <- data.frame(c_data1[c_data1$method=="2-orig",]$value,rep(paste("1-",dict[[target]]),100),rep(task,100),rep(paste(target,"to",source,sep = " "),100))
  colnames(inp2) <-c("value","method","task","atlas")
  
  c_data <- rbind(c_data1,c_data2,inp1,inp2)
  return(c_data)
}

my_plot<-function(c_data,pal,col0,col1){
  
  p<-grouped_ggbetweenstats(
    data = c_data,
    x = method,
    y = value,
    grouping.var = atlas,#grouping variable
    outlier.tagging = TRUE, # whether outliers need to be tagged
    outlier.coef = 2,
    ggsignif.args = list(textsize = 4, tip_length = 0.01),
    p.adjust.method = "bonferroni", # method for adjusting p-values for multiple comparisons
    ggplot.component = list(ggplot2::scale_color_manual(values = c(col0,col1,col1)),
                            ggplot2::theme_bw(base_size=15),
                            ggplot2::scale_y_continuous(limits = c(-0.25, 0.6)
                                                        ,sec.axis = ggplot2::dup_axis())),
    palette = pal,
    package = "ggsci",
    plotgrid.args = list(nrow = 1),
    annotation.args = list(title = "")
  )
  
  return(p)
}
task1 = "rest1"
task2 = "wm"

source = "shen"
target = "brainnetome"

source = "schaefer"
target = "craddock400"
c_data1 <- get_data(source,target,task1)
c_data2 <- get_data(source,target,task2)


p1<-my_plot(c_data1,"default_jama","red","blue")
p2<-my_plot(c_data1,"default_jama","red","blue")   

#jpeg(paste(source,target,task1,task2,".jpg",sep="_"),width = 90)
#pdf(paste(source,target,task1,task2,".pdf",sep="_"),width = 10)
Cairo::Cairo(45,30,
             file = paste(paste(source,target,task1,task2,sep="_"), ".png", sep = ""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc 
)

ggstatsplot::combine_plots(
  list(p1,p2),
  plotgrid.args = list(nrow = 2),
  annotation.args = list(
    title = paste("Between Atlas Optimal Transport: ",toupper(task1)," (TOP), ",toupper(task2)," (Bottom)"))
)

dev.off()
