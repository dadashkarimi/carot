library(RColorBrewer)


library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(vcd)
# library(RColorBrewer)
library(wesanderson)

dict <- new.env(hash = TRUE, parent = emptyenv(), size = NA)
dict[["shen"]] <- "Shen"
dict[["schaefer"]] <- "Shae"
dict[["craddock400"]] <- "Crad400"
dict[["brainnetome"]] <- "Btm246"

get_ot_results <- function(filename,atlas) {
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
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
  c_data <- data.frame(c("orig-orig","orig-ot","ot-orig","ot-ot"),c(c_data11,c_data12,c_data13,c_data14))
  colnames(c_data) <- c("name","value")
  return(c_data)
}


my_bar_plot<-function(c_data,atlas,task){
  # return(ggplot(c_data, aes(x=name, y=value, fill=name))+ 
  #          geom_bar(stat="identity", fill="white",color="black")+
  #          scale_fill_manual(brewer.pal(4, "Set2"))+values=coul))+
  #          labs(title=paste(dict[[atlas]],"-",toupper(task)),
  #               x ="method", y = "ID Rate")+
  #   theme_bw(base_size = 25))
print(c_data)
  p<-barplot(table(c_data$name,c_data$value),
            main="Age Count of 10 Students",
            xlab="Age",
            ylab="Count",
            border="red",
            col="blue",
            density=10
  )
return(p)
  # coul <- (4, "Set2") 
  # p<-barplot(table(c_data$name, c_data$value), names=c_data$value, col=coul )
  # return(p)
  # 
  # +
  #   theme(axis.line = element_line(colour = "black"),
  #         panel.grid.major = element_blank(),
  #         panel.grid.minor = element_blank(),
  #         panel.border = element_blank(),
  #         panel.background = element_blank())
}


my_plot<-function(source,target,task1,task2){
  c_data11 <- get_data(source,target,task1)
  c_data12 <- get_data(source,target,task2)
  c_data21 <- get_data(target,source,task1)
  c_data22 <- get_data(target,source,task2)
  coul <- brewer.pal(5, "Set2") 
  
  p1<-my_bar_plot(c_data11,target,task1)
  p2<-my_bar_plot(c_data12,target,task2)
  p3<-my_bar_plot(c_data21,source,task1)
  p4<-my_bar_plot(c_data22,source,task2)
  
  p<-ggstatsplot::combine_plots(
    list(p1, p2,p3,p4),
    plotgrid.args = list(nrow = 2),
    title.size = 27,
    title.text = ""
    # annotation.args = list(
    #   title = paste("Identification rates: ",toupper(task1)," (TOP), ",toupper(task2)," (Bottom)"))
  )
  
  return(p)
}
task1 = "rest1"
task2 = "rest2"

source = "schaefer"
target = "craddock400"

source = "shen"
target = "brainnetome"

c_data <- get_data(source,target,task1)

# 
Cairo::Cairo(40,40,
             file = paste(paste("fig/",source,target,task1,task2,"cormat",sep="_"), ".png", sep = ""),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)
# 
# p<-barplot(table(c_data$name,c_data$value),
#         main="Age Count of 10 Students",
#         xlab="Age",
#         ylab="Count",
#         border="red",
#         col="blue",
#         density=10)
p <-my_plot(source,target,task1,task2)
# 
dev.off()


