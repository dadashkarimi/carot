library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(latex2exp)


dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["craddock"]] <- "Crad"
dict[["brainnetome"]] <- "Btm246"

get_ot_results_hist <- function(filename,atlas) {
  atlas_data <- read.csv(paste("data/",file = filename,sep=""))
  inp1 <- data.frame(atlas_data$orig,rep("orig",100),rep("task",100))
  inp2 <- data.frame(atlas_data$ot,rep("ot",100),rep("task",100))
  colnames(inp1) <-c("value","method","task")
  colnames(inp2) <-c("value","method","task")
  delta <- inp1$value-inp2$value
  
  #print(delta)
  c_data <- data.frame(delta,rep("task",100),rep(atlas,100))
  # <- data.frame(c_data,rep(atlas,200))
  colnames(c_data) <-c("value","task","atlas")
  
  return(c_data)
}

get_data_hist <- function(source,target,task) {
  f1 <-paste(source,target,task,"iq",sep="_")
  print(f1)
  #f2 <-paste(target,source,task,"iq",sep="_")
  
  c_data1 <- get_ot_results_hist(paste(f1,".csv",sep=""),paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  return(c_data1)
}

get_ot_results_cross_hist <- function(filename,atlas) {
  atlas_data <- read.csv(paste("data/",file = filename,sep=""))
  inp1 <- data.frame(atlas_data$ot,rep("ot",100),rep("task",100))
  colnames(inp1) <-c("value","method","task")
  c_data <- data.frame(inp1$value,rep("task",100),rep(atlas,100))
  colnames(c_data) <-c("value","task","atlas")
  
  return(c_data)
}

get_data_hist_cross <- function(source,target,task,cross_task) {

  f01 <-paste(source,target,task,"iq",sep="_")
  f1 <-paste(source,target,task,"cross_from",cross_task,"iq",sep="_")
  # print(f1)
  
  # f02 <-paste(target,source,task,"iq",sep="_")
  # f2 <-paste(target,source,task,"cross_from",cross_task,"iq",sep="_")
  # 
  c_data01 <- get_ot_results_cross_hist(paste(f01,".csv",sep=""),paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  c_data1 <- get_ot_results_cross_hist(paste(f1,".csv",sep=""),paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  delta1 <-data.frame(c_data01$value-c_data1$value,c_data01$task,c_data01$atlas)
  colnames(delta1) <-c("value","task","atlas")
  return(delta1)
}

get_ot_results <- function(filename,atlas) {
  atlas_data <- read.csv(paste("data/",file = filename,sep=""))
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
  
  c_data1 <- get_ot_results(paste(f1,".csv",sep=""),paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  c_data2 <- get_ot_results(paste(f2,".csv",sep=""),paste(dict[[target]],"to",dict[[source]],paste("(",toupper(task),")"),sep = " "))
  
  inp1 <- data.frame(c_data2[c_data2$method=="2-orig",]$value,rep(paste("1-",dict[[source]]),100),rep(task,100),rep(paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "),100))
  colnames(inp1) <-c("value","method","task","atlas")
  
  inp2 <- data.frame(c_data1[c_data1$method=="2-orig",]$value,rep(paste("1-",dict[[target]]),100),rep(task,100),rep(paste(dict[[target]],"to",dict[[source]],paste("(",toupper(task),")"),sep = " "),100))
  colnames(inp2) <-c("value","method","task","atlas")
  
  c_data <- rbind(c_data1,c_data2,inp1,inp2)
  c_data <- rbind(c_data1,inp1)
  
  return(c_data)
}



get_ot_results_cross <- function(filename,task,atlas) {
  atlas_data <- read.csv(paste("data/",file = filename,sep=""))
  inp1 <- data.frame(atlas_data$orig,rep("1-orig",100),rep("task",100))
  inp2 <- data.frame(atlas_data$ot,rep(paste("2-",task,sep=""),100),rep("task",100))
  colnames(inp1) <-c("value","method","task")
  colnames(inp2) <-c("value","method","task")
  c_data <- rbind(inp1,inp2)
  c_data <- data.frame(c_data,rep(atlas,200))
  colnames(c_data) <-c("value","method","task","atlas")
  
  return(c_data)
}

get_data_cross <- function(source,target,task,cross_task) {
  f01 <-paste(source,target,task,"iq",sep="_")
  f1 <-paste(source,target,task,"cross_from",cross_task,"iq",sep="_")
  
  f02 <-paste(target,source,task,"iq",sep="_")
  f2 <-paste(target,source,task,"cross_from",cross_task,"iq",sep="_")
  
  c_data01 <- get_ot_results_cross(paste(f01,".csv",sep=""),task,
                                   paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  c_data1 <- get_ot_results_cross(paste(f1,".csv",sep=""),task,
                                  paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "))
  
  inp1 <- data.frame(c_data01[101:200,]$value,rep(paste("3-",cross_task),100),rep(task,100),rep(paste(dict[[source]],"to",dict[[target]],paste("(",toupper(task),")"),sep = " "),100))
  colnames(inp1) <-c("value","method","task","atlas")
  
  c_data <- rbind(c_data1,inp1)
  return(c_data)
}




my_plot<-function(c_data,pal,col0,col1){
  
  p<-grouped_ggbetweenstats(
    data = c_data,
    x = method,
    y = value,
    plot.type="box",
    grouping.var = atlas,#grouping variable
    outlier.tagging = TRUE, # whether outliers need to be tagged
    outlier.coef = 2,
    p.adjust.method = "bonferroni", # method for adjusting p-values for multiple comparisons
    ggplot.component = list(ggplot2::scale_color_manual(values = c(col0,col1,col1)),
                            ggplot2::theme_bw(base_size=18)+ggplot2::theme(
                              axis.text.x = element_text(size = 20,family="Decima WE"),
                              plot.margin=unit(c(0,0,0,0), "cm"),
                              axis.title.y.right=element_blank(),
                              axis.text.y.right=element_blank()),
                            ggplot2::scale_y_continuous(limits = c(-0.25, 0.8)
                                                        ,sec.axis = ggplot2::dup_axis())),
    ggsignif.args = list(textsize =3 , tip_length = 0.01),
    palette = pal,
    package = "ggsci",
    results.subtitle = FALSE,
    plotgrid.args = list(nrow = 1),
    annotation.args = list(title = "")
  )+xlab("")+ylab("IQ Prediction")+theme(legend.position="none")+ggplot2::labs(caption = NULL)
  
  return(p)
}

my_plot_hist<-function(c_data,pal){
  p<- grouped_gghistostats(
    data = c_data,
    x = value,
    test.value.line=TRUE,
    test.value.size = 1.5,
    type = "nonparametric",
    results.subtitle=FALSE,
    ggplot.component = list(ggplot2::scale_y_discrete(limits = c(0, 25)),ggplot2::theme(plot.margin=unit(c(0,0,0,0), "cm")
                                           )),
    grouping.var = atlas, # grouping variable
    title.prefix="Fig ",
    normal.curve = TRUE, # superimpose a normal distribution curve
    normal.curve.args = list(color = "red", size = 0.5),
    ggtheme = ggthemes::theme_tufte(),
    # modify the defaults from `ggstatsplot` for each plot
    plotgrid.args = list(nrow = 1),
    annotation.args = list(title = "")
  )+ylab("")+xlab(TeX("$\\Delta(orig -ot)$"))
  
  return(p)
}

task1 = "wm"
cross_task1 = "gambling"

task2 = "gambling"
cross_task2 = "wm"

source = "shen"
target = "craddock"

target = "craddock"
source = "shen"


c_data1_hist <- get_data_hist(source,target,task1)
c_data2_hist <- get_data_hist(source,target,task2)

p1_hist<-my_plot_hist(c_data1_hist,"default_jama")
p2_hist<-my_plot_hist(c_data2_hist,"nrc_npg")

c_data1_hist_cross <- get_data_hist_cross(source,target,task1,cross_task1)
c_data2_hist_cross <- get_data_hist_cross(source,target,task2,cross_task2)

p1_hist_cross<-my_plot_hist(c_data1_hist_cross,"default_jama")
p2_hist_cross<-my_plot_hist(c_data2_hist_cross,"nrc_npg")

c_data1 <- get_data(source,target,task1)
c_data2 <- get_data(source,target,task2)
c_data3 <- get_data(target,source,task1)
c_data4 <- get_data(target,source,task2)
p1<-my_plot(c_data1,"default_jama","red","blue")
p2<-my_plot(c_data2,"default_jama","red","blue")  
p3<-my_plot(c_data3,"default_jama","red","blue")
p4<-my_plot(c_data4,"default_jama","red","blue")   

c_data1_cross <- get_data_cross(source,target,task1,cross_task1)
c_data2_cross <- get_data_cross(source,target,task2,cross_task2)
p1_cross<-my_plot(c_data1_cross,"default_jama","blue","blue")
p2_cross<-my_plot(c_data2_cross,"default_jama","blue","blue")  


Cairo::Cairo(30,22,
             file = paste("figs/",paste(source,target,task1,task2,"chart","hist",sep="_"), ".png", sep = ""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc 
)

ggarrange(ggarrange(p1,p2,nrow=2),ggarrange(p3,p4,nrow=2)) 
                                           # First row with scatter plot
# ggarrange(ggarrange(p1,p2,nrow=2),
#           ggarrange(ggarrange(p1_hist+coord_flip()+ ggtitle(""), heights = c(3,1),nrow=2),
#                     ggarrange(p2_hist+coord_flip()+ ggtitle(""), heights = c(3,1),nrow=2)
#                     ,nrow=2),
#           widths = c(3,1),
#           nrow=1,ncol=2) # Second row with box and dot plots
#ggarrange(ggarrange(p1_cross,p2_cross,nrow=2),ggarrange(p1_hist_cross+coord_flip()+ ggtitle(""),p2_hist_cross+coord_flip()+ ggtitle(""),nrow=2),widths = c(3,1),nrow=1,ncol=2) # Second row with box and dot plots
# ggsave(file="whatever.pdf", g)

dev.off()
