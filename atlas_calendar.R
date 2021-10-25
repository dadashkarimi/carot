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


get_ot_results <- function(filename,atlas1,atlas2,target,task,task_id) {
  # print(paste("data/",filename,sep=""))
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
  
  c_data <- data.frame(mean(atlas_data$ot),mean(atlas_data$orig),atlas1,atlas2,target, dict[[task]],task_id)
  colnames(c_data) <-c("value","orig","atlas1","atlas2","target","task","task_id")
  
  return(c_data)
}

get_data <- function(targets) {
  my_data <- list()
  # i<-2
  data_index =1
  for (i in 1:3) {
    for (j in 1:length(atlas_list)) {
      for(target in targets){
        print(targets)
        for (k in 1:length(task_list)) {
          if((atlas_list[i]!=target) && (atlas_list[j]!=target) ){
            if(atlas_list[i] == atlas_list[j]){
              f <-paste(atlas_list[i],target,task_list[k],"iq",sep="_")
              my_data[[data_index]] <- get_ot_results(paste(f,".csv",sep=""),
                                                      dict[[atlas_list[i]]],dict[[atlas_list[j]]],
                                                      target,task_list[k],k)
            }else{
              f <-paste(atlas_list[i],atlas_list[j],target,task_list[k],"iq",sep="_")
              if(!(file.exists(paste("data/",f,".csv",sep="")))){
                f <-paste(atlas_list[j],atlas_list[i],target,task_list[k],"iq",sep="_")
              }
              my_data[[data_index]] <- get_ot_results(paste(f,".csv",sep=""),dict[[atlas_list[i]]],dict[[atlas_list[j]]],target,task_list[k],k)
            }
            data_index = data_index + 1
          }else{ # one of the atlases in source equals tarrget
            
            f <-paste(atlas_list[i],target,task_list[k],"iq",sep="_")
            if(!(file.exists(paste("data/",f,".csv",sep="")))){
              f <-paste(atlas_list[j],target,task_list[k],"iq",sep="_")
            }
            my_data[[data_index]] <- get_ot_results(paste(f,".csv",sep=""),
                                                    dict[[atlas_list[i]]],dict[[atlas_list[j]]],
                                                    target,task_list[k],k)
          }
        }
      }
    }
  }
  cdata <- bind_rows(my_data)
  return(cdata)
}


task1 = "rest1"
task2 = "gambling"
task3 = "wm"
task4 = "motor"
task5 = "emotion"
task6 = "relational"
task7 = "lang"
task8 = "social"

target = "shen"
targets <-c("shen","power","craddock","shen368","craddock400")
c_data1 <- get_data(targets)
Cairo::Cairo(22,19,
             file = paste("calendar_plot.png"),
             type = "png",bg = "transparent", dpi = 300, units = "cm" #you can change to pixels etc
)

paletter_vector <- viridis_pal(option="C")(4)
ggplot(c_data1, aes(task, atlas2, fill = value)) + 
  geom_tile(colour = "white") + 
  facet_grid(target~atlas1) + 
  scale_fill_viridis_c()+
  labs(x="",
       y="",
       # title = "Merging Atlases with Optimal Transport", 
       subtitle="Merging Atlases with Optimal Transport",
       fill="ot")+ theme_bw( base_size = 15)+
  # geom_point(aes(color=orig),size=3)+scale_colour_gradient(low = paletter_vector[4], high = paletter_vector[1],limits=c(0,0.4),name="Orig(dots)")+
  geom_text(aes(label=paste(round(orig,2)),color=orig),size=1.8)+scale_colour_distiller(palette="Spectral")+
  theme(axis.text.y = element_text(color = "black", angle = 0),
        axis.text.x = element_text(color = "black", angle = 45),
        legend.title=element_text(size=11),
        legend.key.size = unit(0.3, "cm"),
        strip.background = element_blank())
# geom_text(data=c_data1,
#                                                                  label=c(round(c_data1$value,digit=2),
#                                                                          round(c_data1$orig,digit=2)))
dev.off()
# dev.off()
