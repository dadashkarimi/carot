library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)

get_ot_results <- function(filename,atlas) {
  atlas_data <- read.csv(file = filename)
  inp1 <- data.frame(atlas_data$orig,rep("orig",100),rep("task",100))
  colnames(inp1) <-c("value","method","task")
  c_data <- data.frame(inp1$value,rep("task",100),rep(atlas,100))
  colnames(c_data) <-c("value","task","atlas")
  
  return(c_data)
}

get_data <- function(source,target,task,cross_task) {
  f01 <-paste(source,target,task,"iq",sep="_")
  f1 <-paste(source,target,task,"cross_from",cross_task,"iq",sep="_")
  
  f02 <-paste(target,source,task,"iq",sep="_")
  f2 <-paste(target,source,task,"cross_from",cross_task,"iq",sep="_")
  
  c_data01 <- get_ot_results(paste(f01,".csv",sep=""),paste(source,"to",target,sep = " "))
  c_data1 <- get_ot_results(paste(f1,".csv",sep=""),paste(source,"to",target,sep = " "))
  delta1 <-data.frame(-c_data01$value+c_data1$value,c_data01$task,c_data01$atlas)
  colnames(delta1) <-c("value","task","atlas")
  
  
  c_data02 <- get_ot_results(paste(f02,".csv",sep=""),paste(target,"to",source,sep = " "))
  c_data2 <- get_ot_results(paste(f2,".csv",sep=""),paste(target,"to",source,sep = " "))
  delta2 <-data.frame(-c_data02$value+c_data1$value,c_data02$task,c_data02$atlas)
  colnames(delta2) <-c("value","task","atlas")
  
  c_data <- rbind(delta1,delta2)
  return(c_data)
}
my_plot<-function(c_data,pal){
  
  
  
  # plot
  p<- grouped_gghistostats(
    data = c_data,
    x = value,
    test.value.line=TRUE,
    test.value.size = 1.5,
    type = "nonparametric",
    results.subtitle=FALSE,
    ggplot.component = ggplot2::theme_bw(base_size=20),
    grouping.var = atlas, # grouping variable
    title.prefix="Fig ",
    normal.curve = TRUE, # superimpose a normal distribution curve
    normal.curve.args = list(color = "red", size = 1),
    ggtheme = ggthemes::theme_tufte(),
    # modify the defaults from `ggstatsplot` for each plot
    plotgrid.args = list(nrow = 1),
    annotation.args = list(title = "dddd")
  )
  
  return(p)
}
task1 = "rest1"
cross_task1 = "wm"

task2 = "wm"
cross_task2 = "rest1"


source = "craddock400"
target = "schaefer"
# 
source = "shen"
target = "brainnetome"

c_data1 <- get_data(source,target,task1,cross_task1)
c_data2 <- get_data(source,target,task2,cross_task2)



p1<-my_plot(c_data1,"default_jama")
p2<-my_plot(c_data2,"nrc_npg")


Cairo::Cairo(30,18,
             file = paste(paste(source,target,task1,task2,"cross_hist",sep="_"), ".png", sep = ""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc
)

ggstatsplot::combine_plots(
  list(p1, p2),
  title.size = 10,
  title.text = paste("Between Atlas, Cross-Task Optimal Transport: "),
  plotgrid.args = list(nrow = 2),
  annotation.args = list(
    title = paste("Between Atlas, Cross-Task Optimal Transport: ",
                  toupper(task1)," (TOP), ",toupper(task2)," (Bottom)"))
)+ggplot2::theme_bw(base_size=25)

dev.off()
