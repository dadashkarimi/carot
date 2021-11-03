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
  cols = c("#4682B4","orange")
# wes_palette("Rushmore1")#palette("Set1")
  c_data<-c_data[c_data$correct=="correct",]
  cols <-wes_palette("Darjeeling2")
  # mylabel<-round(sum(c_data$correct=="correct")/length(c_data))
  # print(mylabel)
  p<-ggplot(c_data, aes(x=direction, fill=correct))+ geom_bar(color = "black")+theme_void(base_size=18)+
    theme(axis.text.y = element_text(color = "black",size = 14, angle = 40),
                                              axis.text.x = element_text(color = "black",size = 14,angle = 40),
                                              axis.title.y  = element_blank(),
                                              axis.line.y = element_line(color="black", size = 0.5))+
          geom_text(
    aes(label=after_stat(count)),
    stat='count',
    nudge_y=9.125,
    va='bottom'
  )+scale_fill_manual(values=cols[2])+labs(title=source)
  # p<-ggplot(c_data) +
  #   aes(x = direction, fill = forcats::fct_rev(correct)) +theme_void(base_size=18)+
  #   geom_bar(position = "fill",color = "black")+theme(axis.text.y = element_text(color = "black",size = 14, angle = 40),
  #                                     axis.text.x = element_text(color = "black",size = 14,angle = 40),
  #                                     axis.title.y  = element_blank(),
  #                                     axis.line.y = element_line(color="black", size = 0.5))+
  #   scale_fill_manual(values=c("white",cols[2]))+labs(title=source)
    # geom_text(aes(label = round(ave_lifeExp, 1)), 
    #           position = position_dodge(0.9),
    #           color="white",vjust = 0.5,hjust = 1)
  
  # p<-ggbarstats(
  #   data = c_data,
  #   x = correct,
  #   y = direction,
  #   grouping.var = atlas,
  #   title = paste(source,sep=''),
  #   xlab = "",
  #   legend.title = "",
  #   results.subtitle = FALSE, results.title = FALSE,
  #   bf.message = FALSE,
  #   subtitle = "",
  #   paired = TRUE,
  #   ggtheme = hrbrthemes::theme_ipsum_pub(),
  #   ggplot.component = list(ggplot2::theme_void(base_size=5),ggplot2::theme(axis.text.y = element_text(color = "black",size = 14, angle = 40),
  #                                                                               axis.text.x = element_text(color = "black",size = 14),
  #                                                                               axis.title.y  = element_text(color = "black",size = 16,angle = 90),
  #                                                                               axis.title.x  = element_text(color = "black",size = 16)),
  #                           ggplot2::scale_x_discrete(guide = ggplot2::guide_axis(n.dodge = 2)),
  #                           ggplot2::scale_fill_manual(values=c("white",cols[1])))
  # )+ggtitle(source)
  # if(id==7){
  #   p<-p#+theme(legend.position="bottom")
  # }else{
    p<-p+theme(legend.position="none")
   # }
  return(p)
}
# plot
task1 = "rest1"
task2 = "rest2"

source1 = "schaefer"
target = "shen"

c_data <-get_data(target,task1)

# 
Cairo::Cairo(40,7,
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



# p<-ggarrange(ggarrange(p1,p2,p3,p4,nrow=2,ncol=4,widths = c(1,1,1,1),heights=c(5.5,1)),
#              ggarrange(ggarrange(p5,p6,p7,nrow=2,ncol=3,heights=c(5.5,1)),p8,nrow=1,ncol=2,widths =c(3,1)),nrow=2) # Second row with box and dot plots

p<-ggarrange(p1,p2,p3,p4,p5,p6,p7,nrow=1,ncol=7)
# p<-ggarrange(ggarrange(p1,p2,p3,p4,nrow=2,ncol=2),ggarrange(p5,p6,nrow=2),widths = c(1.4,1)) # Second row with box and dot plots
# p<-ggarrange(p1,p2,p3,p4,p5,p6,nrow=2,ncol=3) # Second row with box and dot plots

plot.new()
p
# legend("bottomright",
#        pch=21, pt.cex=2.9,
#        pt.bg=c('#4682B4','white'),
#        legend=c('correct', 'wrong'),
#        cex=1.5, bty='n')

dev.off()