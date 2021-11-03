set.seed(123)
library(Rcpp)
library(ggstatsplot)
library(palmerpenguins)
library(tidyverse)
library(gridExtra)
library(ggpubr)
library(viridis)
library(ggplot2)
library(gridExtra)
library(grid)
require(lattice)
library(viridisLite)
library(latex2exp)
library(RColorBrewer)

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

pal <- brewer.pal(11,"Spectral")
task_list <-c("rest1","gambling","wm","motor","emotion","relational","lang","social")

atlas_list <- c("shen","schaefer","craddock","brainnetome","power","dosenbach")
# atlas_list <- c("dosenbach","schaefer","brainnetome")


get_ot_results <- function(filename,target,task) {
  cdata <- try(read.csv(file = paste("data/",filename,sep="")))
  return(cdata)
}

get_data <- function(source, target,task) {
  f <-paste("G",source,target,task,"iq",sep="_")
  cdata<-get_ot_results(paste(f,".csv",sep=""),target,task)
  cdata$X<-NULL
  colnames(cdata)<-1:length(cdata[1,])
  
  return(cdata)
}

get_data_all<-function(task_list,source,target){
  my_data <- list()
  data_index =1
  for(task in task_list){
    cdata <-get_data(source,target,task)
    my_data[[data_index]] <- cdata
    data_index <-data_index+1
  }
  return(Reduce("+", my_data)/length(my_data))
}

my_plot<-function(entries,source,target,task,source_size,target_Size){
  a<-paste('$T: ','R^{',source_size,' \\times ',target_Size,'}$}',sep="")
  print(a)
  p<-ggplot(entries, aes(x=x, y=y) ) +
    stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
    scale_fill_distiller(palette= "Spectral", direction=-1) +
    theme(
      legend.position='none'
    )+scale_x_continuous(name=source)+
    labs(title=TeX(a),subtitle = paste("Task:",task))+ylab(target)+theme_bw(base_size = 10)+ 
    scale_x_continuous(name=source,expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme(legend.position="none",
          axis.text.y = element_text(color = "black",size = 8, angle = 40),
          axis.text.x = element_text(color = "black",size = 8,angle = 40),
          axis.title.y  = element_text(color = "black",size = 12,angle = 90),
          axis.title.x  = element_text(color = "black",size = 12),
          axis.line.y = element_line(color="black", size = 0.1),
          axis.line.x = element_line(color="black", size = 0.1))
}
source="craddock"
target="shen"
task="rest1"
average = FALSE

my_plots <- list()
plot_sizes <- list()
p_index <-1


if(average){
  for(source in atlas_list){
    if(source!=target){
      cdata<-get_data_all(task_list,source,target)
      entries <-which(cdata>0.001,arr.ind = T)
      entries<-data.frame(x=entries[,1],y=entries[,2])
      
      p<-my_plot(entries,source,target,"all",length(cdata[1,]),length(cdata[,1]))
      my_plots[[p_index]]<-p
      plot_sizes[[p_index]]<-length(cdata[,1])
      p_index <- p_index+1
    }
  }
  task = "average"
}else{
  task_list<-c("rest1")
  task<-task_list[1]
  for(task in task_list){
    for(source in atlas_list){
      if(source!=target){
        cdata<-get_data(source, target,task)
        entries <-which(cdata>0.001,arr.ind = T)
        entries<-data.frame(x=entries[,1],y=entries[,2])
        
        p<-my_plot(entries,source,target,task,length(cdata[1,]),length(cdata[,1]))
        my_plots[[p_index]]<-p
        plot_sizes[[p_index]]<-length(cdata[,1])
        p_index <- p_index+1
      }
    }
  }
}

plot_sizes[[p_index]]<-200

Cairo::Cairo(28,6.5,
             file = paste(paste("all",target,task,sep="_"),"_heatpam.png",sep=""),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc
)

plot.new()

ggarrange(plotlist = my_plots,widths=array(as.numeric(plot_sizes)),nrow=1,ncol=6)
legend('topright', 
       pch=21, pt.cex=1.8,
       pt.bg=c(pal[1], 
              pal[2],
               pal[3],
              pal[5],
              pal[5]),
       legend=c('Very High', 'High','Medium', 'Low','Zero'), 
       cex=0.8, bty='n')

dev.off()


