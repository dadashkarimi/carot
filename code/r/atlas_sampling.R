library(hrbrthemes)
library(gcookbook)
library(tidyverse)
library("wesanderson")
library(scales)
library(ggpubr)

# current verison

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


get_ot_results <- function(filename,target,task,sample) {
  atlas_data <- try(read.csv(file = paste("data/",filename,sep="")))
  #atlas_data <-atlas_data[sample(nrow(atlas_data), 10), ]
  c_data <- data.frame(mean(atlas_data$pearson),sd(atlas_data$pearson),target, dict[[task]],sample)
  colnames(c_data) <-c("value","std","target","task","sample")
  return(c_data)
}

get_data <- function(tasks,target) {
  my_data <- list()
  # i<-2
  data_index =1
  for(i in 0:4){
    for(task in tasks) {
      f <-paste("all",target,task,paste("sample",i,sep = ""),"iq",sep="_")
      my_data[[data_index]] <- get_ot_results(paste(f,".csv",sep=""),target,task,i)
      data_index <-data_index+1
    }
  }
  cdata <- bind_rows(my_data)
  return(cdata)
  
}

my_plot<-function(cdata,task,target,id){
  paletter_vector <- paletteer::paletteer_d(palette = "palettetown::venusaur",n = 4)
  # fill = wes_palette("Rushmore1")[1:5] geom bar
  p <- ggplot(cdata, aes(x=sample, y=value)) + 
    geom_bar(stat='identity',colour="black",fill = wes_palette("Rushmore1")[1:5])+
    geom_errorbar( aes(x=sample, ymin=value-std, ymax=value+std), width=0.4, colour="black",size=.3)+
    theme_void(base_size = 14)+
    scale_y_continuous(limits=c(0,0.9))+
    labs(title="",subtitle= bquote(T["#"~ .("n-k")] ~ "=" ~.(dict[[target]])), x="k", y = "Pearson")+
    theme(legend.position="none",
          axis.text.y = element_text(color = "black",size = 14, angle = 40),
          axis.text.x = element_text(color = "black",size = 14),
          axis.title.y  = element_text(color = "black",size = 16,angle = 90),
          axis.title.x  = element_text(color = "black",size = 16),
          axis.line.y = element_line(color="black", size = 0.5))

  if(id==1){
    p<-p+labs(title="Leave k-atlas out")
  }
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

packageVersion("hrbrthemes")
tasks <-c("relational","social","rest1","gambling","wm","emotion","motor","lang")
targets <- c("shen","schaefer","craddock","brainnetome","dosenbach","power")
task <-"rest1"

Cairo::Cairo(40,7,
             file = paste("atlas_sample_","all",".png"),
             type = "png",bg = "white", dpi = 300, units = "cm" #you can change to pixels etc
)

my_plots <- list()
p_index <-1
for(target in targets){
  # for(target in targets){
  if(target!="all"){
    cdata<- get_data(task,target)
    p<-my_plot(cdata,task,target,p_index)
    my_plots[[p_index]]<-p
    p_index <- p_index+1
  }
}


ggarrange(plotlist = my_plots,nrow=1,ncol=6)


dev.off()