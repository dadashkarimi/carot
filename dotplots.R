set.seed(123)
library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(gridExtra)
library(ggpubr)


dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["shen368"]] <- "Shen368"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["craddock"]] <- "Crad"
dict[["brainnetome"]] <- "Btm246"
dict[["power"]] <- "Power"
dict[["dosenbach"]] <- "Dsnbch"


atlas_list <- c("shen","shen368","schaefer","craddock","craddock400","brainnetome","power","dosenbach")

get_ot_results <- function(filename,atlas,target,task) {
  print(paste("data/",filename,sep=""))
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
  
  inp1 <- data.frame(mean(atlas_data$ot),sd(atlas_data$ot),mean(atlas_data$orig),sd(atlas_data$orig),task,target)
  colnames(inp1) <-c("value","std","orig","orig_std","task","target")

  c_data <- data.frame(inp1,rep(atlas,1))
  colnames(c_data) <-c("value","std","orig","orig_std","task","target","atlas")
  
  return(c_data)
}

get_data <- function(target,task) {
  # f0 <-paste(source,target,task,"iq",sep="_")
  # c_data0 <- get_ot_results(paste(f0,".csv",sep=""),paste(dict[[source]],"to",dict[[target]],sep = " "))
  # 
  f1 <-paste(atlas_list[1],target,task,"iq",sep="_")
  c_data1 <- get_ot_results(paste(f1,".csv",sep=""),paste(dict[[atlas_list[1]]],sep = " "),target,task)
  
  f2 <-paste(atlas_list[2],target,task,"iq",sep="_")
  c_data2 <- get_ot_results(paste(f2,".csv",sep=""),paste(dict[[atlas_list[2]]],sep = " "),target,task)
  
  f3 <-paste(atlas_list[3],target,task,"iq",sep="_")
  c_data3 <- get_ot_results(paste(f3,".csv",sep=""),paste(dict[[atlas_list[3]]],sep = " "),target,task)
  
  f4 <-paste(atlas_list[4],target,task,"iq",sep="_")
  c_data4 <- get_ot_results(paste(f4,".csv",sep=""),paste(dict[[atlas_list[4]]],sep = " "),target,task)
  
  f5 <-paste(atlas_list[5],target,task,"iq",sep="_")
  c_data5 <- get_ot_results(paste(f5,".csv",sep=""),paste(dict[[atlas_list[5]]],sep = " "),target,task)
  
  f6 <-paste(atlas_list[6],target,task,"iq",sep="_")
  c_data6 <- get_ot_results(paste(f6,".csv",sep=""),paste(dict[[atlas_list[6]]],sep = " "),target,task)
  
  f7 <-paste(atlas_list[7],target,task,"iq",sep="_")
  c_data7 <- get_ot_results(paste(f7,".csv",sep=""),paste(dict[[atlas_list[7]]],sep = " "),target,task)
  
  f8 <-paste(atlas_list[8],target,task,"iq",sep="_")
  c_data8 <- get_ot_results(paste(f8,".csv",sep=""),paste(dict[[atlas_list[8]]],sep = " "),target,task)
  
  
  c_data <-rbind(c_data1,c_data2,c_data3,c_data4,c_data5,c_data6,c_data7,c_data8)
  return(c_data)
}

my_plot<-function(c_data,target,task){
  paletter_vector <- paletteer::paletteer_d(palette = "palettetown::venusaur",
                                            n = nlevels(as.factor(atlas_list)),type = "discrete")
  # data<-c_data[order(c_data$value),]
  # pl <-paletter_vector[order(c_data$value)]
  # plot
  # print(head(data))
  p<-p<-ggplot(c_data, aes(value,atlas)) + 
    geom_point(aes(size = std),color=paletter_vector)+
    geom_point(aes(size = std),color="black",pch=21)+
    geom_vline(xintercept = mean(c_data$orig),alpha=0.8,color="red")+
    geom_vline(xintercept = mean(c_data$orig),color="red",
                                                                 ,size= 3+100*sd(c_data$orig),alpha=0.06)+
    scale_x_continuous(limits = c(0.0, 0.4))+theme_bw()+
                            theme(legend.position="none",axis.text.y = element_text(color = "black",size = 10, angle = 40),
                                  axis.text.x = element_text(color = "black",size = 10),
                                  axis.title.y = element_blank())+
    scale_x_continuous(name=task, limits=c(0, 0.4))+ggtitle(dict[[target]])
  # p<-ggdotplotstats(
  #   data = data,
  #   x = value,
  #   y = atlas,
  #   xlab = paste("",task),
  #   ylab = "",
  #   test.value = 15.5,
  #   point.args = list(
  #     shape = 19,
  #     color = pl,
  #     size = 6
  #   ),
  #   results.subtitle=FALSE,
  #   title = paste(""),
  #   ggplot.component = list(ggplot2::theme_bw(base_size=15),
  #                           ggplot2::scale_x_continuous(limits = c(0.0, 0.4)),
  #                           theme( axis.text.y = element_text(color = "black", 
  #                                                                                             size = 10, angle = 40))),
  #   subtitle = paste(dict[[source]]),
  #   ggtheme = hrbrthemes::theme_ipsum()
  # )
  return(p)
}
task1 = "rest1"
task2 = "gambling"
task3 = "wm"
task4 = "motor"
task5 = "emotion"
task6 = "relational"
task7 = "lang"
task8 = "social"

target = "dosenbach"

c_data1 <- get_data(target,task1)
c_data2 <- get_data(target,task2)
c_data3 <- get_data(target,task3)
c_data4 <- get_data(target,task4)
c_data5 <- get_data(target,task5)
c_data6 <- get_data(target,task6)
c_data7 <- get_data(target,task7)
c_data8 <- get_data(target,task8)




# p1<-my_plot(c_data1)
p1<-my_plot(c_data1,target,task1)
p2<-my_plot(c_data2,target,task2)
p3<-my_plot(c_data3,target,task3)
p4<-my_plot(c_data4,target,task4)
p5<-my_plot(c_data5,target,task5)
p6<-my_plot(c_data6,target,task6)
p7<-my_plot(c_data7,target,task7)
p8<-my_plot(c_data8,target,task8)

Cairo::Cairo(30,20,
             file = paste("",paste(target,"all","dotplot",sep="_"), ".png", sep = ""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc 
)

ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,nrow=2,ncol=4)
# removing factor level with very few no. of observations
# df <- dplyr::filter(ggplot2::mpg, cyl %in% c("4", "6"))

dev.off()
