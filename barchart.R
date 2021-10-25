set.seed(123)
library(ggplot2)
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


get_ot_results <- function(filename,atlas) {
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
  if (!inherits(atlas_data, 'try-error')) atlas_data
  inp1 <- round(atlas_data$X1,digits=2)
  return(inp1)
}

get_data <- function(target,task) {
  my_data <- list()
  for (i in 1:8) {
    if(atlas_list[i]!=target){
      source = atlas_list[i]
  f11 <-paste(source,target,task,"orig-orig","id",sep="_")
  f12 <-paste(source,target,task,"orig-ot","id",sep="_")
  f13 <-paste(source,target,task,"ot-orig","id",sep="_")
  f14 <-paste(source,target,task,"ot-ot","id",sep="_")

  c_data11 <- get_ot_results(paste(f11,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data12 <- get_ot_results(paste(f12,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data13 <- get_ot_results(paste(f13,".csv",sep=""),paste(source,"to",target,sep = " ")) 
  c_data14 <- get_ot_results(paste(f14,".csv",sep=""),paste(source,"to",target,sep = " ")) 

  # 
  d1 <- c(rep('correct',round(100*c_data11)),rep('wrong',100-round(100*c_data11)))#rnorm(100,mean=100*c_data11, sd = 5)
  d2 <- c(rep('correct',round(100*c_data12)),rep('wrong',100-round(100*c_data12)))
  d3 <- c(rep('correct',round(100*c_data13)),rep('wrong',100-round(100*c_data13)))
  d4 <- c(rep('correct',round(100*c_data14)),rep('wrong',100-round(100*c_data14)))

  c_data1 <-data.frame(c(d1,d2,d3,d4),c(rep('orig/orig',100),rep('orig/ot',100),
                                        rep('ot/orig',100),rep('ot/ot',100)),rep(paste(dict[[source]]),400))

  colnames(c_data1) <-c("correct","direction","atlas")
  my_data[[atlas_list[i]]] <- c_data1
    }
  }
  return(my_data)
}


myplot<-function(c_data,source,target,task,id){
  cols = palette("Set2")
  p<-ggbarstats(
    data = c_data,
    x = correct,
    y = direction,
    grouping.var = atlas,
    title = paste(source,sep=''),
    xlab = "",
    legend.title = "",
    results.subtitle = FALSE,
    ggtheme = hrbrthemes::theme_ipsum_pub(),
    ggplot.component = list(ggplot2::theme_minimal(base_size=20),
                            ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(n.dodge = 2)),
                            ggplot2::scale_fill_manual(values=c(cols[2],cols[1])))
  )+ggtitle(source)
  if(id==7){
    # p<-p+theme(legend.position="none")
  # }else{
    p<-p+theme(legend.position="bottom")
  }else{
    p<-p+theme(legend.position="none")
   }
  return(p)
}
# plot
task1 = "rest1"
task2 = "rest2"

source1 = "schaefer"
target = "shen"

# source = "shen"
# target = "brainnetome"


c_data <-get_data(target,task1)

# c_data2 <-get_data(target,source,task1)
# 
# c_data3 <-get_data(source,target,task2)
# c_data4 <-get_data(target,source,task2)
# 
Cairo::Cairo(40,28,
             file = paste("figs/",paste("all",target,task1,task2,"cormat",sep="_"), ".png", sep = ""),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)


p1 <-myplot(c_data$shen368,c_data$shen368$atlas,target,task1,1)
p2 <-myplot(c_data$schaefer,c_data$schaefer$atlas,target,task1,2)
p3 <-myplot(c_data$power,c_data$power$atlas,target,task1,3)
p4 <-myplot(c_data$dosenbach,c_data$dosenbach$atlas,target,task1,4)
p5 <-myplot(c_data$craddock400,c_data$craddock400$atlas,target,task1,5)
p6 <-myplot(c_data$craddock,c_data$craddock$atlas,target,task1,6)
p7 <-myplot(c_data$brainnetome,c_data$brainnetome$atlas,target,task1,7)
p8 <-myplot(c_data$shen,c_data$shen$atlas,target,task1,8)



# p_list <- list()
# for(i in 1:7){
#   p_list[[i]]<-myplot(c_data[[atlas_list[i]]],source,target,task1,1)
# }

# p1<- myplot(c_data1,source,target,task1,1)
# p2<- myplot(c_data2,target,source,task1,2)
# p3<- myplot(c_data3,source,target,task2)
# p4<- myplot(c_data4,target,source,task2)


p<-ggarrange(ggarrange(p1,p2,p3,p4,nrow=2,ncol=4,widths = c(1,1,1,1),heights=c(5.5,1)),
             ggarrange(ggarrange(p4,p5,p6,nrow=2,ncol=3,heights=c(5.5,1)),p7,nrow=1,ncol=2,widths =c(3,1)),nrow=2) # Second row with box and dot plots

# p<-ggarrange(ggarrange(p1,p2,p3,p4,nrow=2,ncol=2),ggarrange(p5,p6,nrow=2),widths = c(1.4,1)) # Second row with box and dot plots
# p<-ggarrange(p1,p2,p3,p4,p5,p6,nrow=2,ncol=3) # Second row with box and dot plots


p
dev.off()