library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(latex2exp)
library(ggplot2)
library("wesanderson")
library(latex2exp)
library(rlang)

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

atlas_list <- c("shen","shen368","schaefer","craddock","brainnetome","power","dosenbach","craddock400")

pal <- wes_palette("Rushmore1")[1:3]

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




my_plot<-function(c_data,source,target,task,within_atlas,id){
  c_data <- c_data[order(c_data$method),]
  

  p<-ggboxplot(c_data,x = "method",y = "value",
                fill =pal)+theme_void(base_size = 14)+
    theme(axis.text.x = element_text(angle = 30),
          axis.title=element_text(size=15,face="bold"),
          plot.subtitle=element_text(size=12))+xlab("method")+
    labs(subtitle =paste("Task: ",dict[[task]]))+
    theme(legend.position="none",
          axis.text.y = element_text(color = "black",size = 14, angle = 40),
          axis.text.x = element_text(color = "black",size = 14),
          axis.title.y  = element_text(color = "black",size = 16,angle = 90),
          axis.title.x  = element_text(color = "black",size = 16),
          axis.line.y = element_line(color="black", size = 0.5))
  
  if(id!=0){
    t <- dict[[target]]
    p<-p+labs(title= bquote(T["#"~ .(dict[[source]])] ~ "=" ~.(dict[[target]])),caption="", y = "Pearson")
  }else{
    p<-p+theme(legend.position="none")
  }
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

within_atlas <- FALSE
task1 = "wm"
cross_task1 = "gambling"

task2 = "gambling"
cross_task2 = "wm"
#atlas_list<-c("craddock")
source = "shen"
target = "shen"
tasks<-task_list

if(!within_atlas){
 tasks<-c("wm") 
 atlas_list <- c("schaefer","craddock","brainnetome","power","dosenbach","craddock400")
}




my_plots <- list()
p_index <-1

for(source in atlas_list){
  if(source!=target){
    for(task in tasks){
      print(task)
      c_data <- get_data(source,target,task)
      p<-my_plot(c_data,source,target,task,within_atlas,p_index)
      my_plots[[p_index]]<-p
      p_index <- p_index+1
    }
  }
}

# 

Cairo::Cairo(40,10,
             file = paste("figs/",paste("all",target,task,"boxplot",sep="_"), ".png", sep = ""),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc 
)
plot.new()



ggarrange(plotlist = my_plots,nrow=1,ncol=6)
# 
# legend("topright",
#        pch=21, pt.cex=1.8,
#        pt.bg=c(pal[3], 
#                pal[2],
#                pal[1]),
#        legend=c('ot', 'orig', 'Source'), 
#        cex=0.8, bty='n')

dev.off()
